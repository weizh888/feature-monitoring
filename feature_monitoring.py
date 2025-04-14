import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
import argparse
import yaml
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, ArrayType, IntegerType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureDriftMonitor:
    def __init__(self):
        """Initialize the feature drift monitor with Spark session and fixed date column."""
        self.spark = SparkSession.builder.appName("FeatureDriftMonitor").enableHiveSupport().getOrCreate()
        self.date_column = "grass_date"

    def calculate_psi(self, expected: Union[pd.Series, Dict[str, float]], actual: pd.Series, 
                     buckets: int = 10, breakpoints: Optional[List[float]] = None) -> Tuple[float, Dict[str, Any]]:
        """Calculate Population Stability Index (PSI) between two distributions.
        
        Args:
            expected: Either a pandas Series with the expected distribution or a dictionary of bucket:probability
            actual: Actual data as a pandas Series
            buckets: Number of buckets to use for continuous variables (ignored if breakpoints provided)
            breakpoints: List of breakpoints for continuous variables (optional)
            
        Returns:
            Tuple containing:
            - PSI value (float)
            - Dictionary containing bucket information
        """
        # Track missing values
        actual_missing = actual.isna()
        actual_non_missing = actual[~actual_missing]
        if len(actual_non_missing) == 0: return np.nan, {}
        
        bucket_info = {'expected_prob': {}, 'actual_prob': {}, 'breakpoints': breakpoints}

        if isinstance(expected, dict):
            # For categorical data, include missing values as a separate category
            bucket_info['expected_prob'] = expected
            actual_buckets = pd.cut(actual_non_missing, breakpoints, include_lowest=True) if breakpoints else actual_non_missing
        else:
            # For numeric data, calculate expected distribution including missing values
            expected_missing = expected.isna()
            expected_non_missing = expected[~expected_missing]
            if len(expected_non_missing) == 0: return np.nan, {}
            
            if pd.api.types.is_numeric_dtype(expected):
                if expected_non_missing.nunique() == 1: return 0.0, bucket_info
                breakpoints = np.unique(breakpoints if breakpoints else np.percentile(expected_non_missing, np.linspace(0, 100, buckets + 1)))
                if len(breakpoints) < 2: return 0.0, bucket_info
                expected_buckets = pd.cut(expected_non_missing, breakpoints, include_lowest=True)
                actual_buckets = pd.cut(actual_non_missing, breakpoints, include_lowest=True)
                bucket_info['breakpoints'] = breakpoints.tolist()
            else:
                expected_buckets, actual_buckets = expected_non_missing, actual_non_missing
            
            # Calculate expected probabilities including missing values
            total_expected = len(expected)
            expected_missing_prob = expected_missing.sum() / total_expected
            expected_non_missing_probs = expected_buckets.value_counts(normalize=True) * (1 - expected_missing_prob)
            bucket_info['expected_prob'] = {**expected_non_missing_probs.to_dict(), 'NULL': expected_missing_prob}

        # Calculate actual probabilities including missing values
        epsilon = 1e-10
        total_actual = len(actual)
        actual_missing_prob = actual_missing.sum() / total_actual
        actual_non_missing_probs = (actual_buckets.value_counts(normalize=True) * (1 - actual_missing_prob)).to_dict()
        bucket_info['actual_prob'] = {**actual_non_missing_probs, 'NULL': actual_missing_prob}

        # Ensure all buckets exist in both distributions
        all_buckets = set(bucket_info['expected_prob'].keys()) | set(bucket_info['actual_prob'].keys())
        for bucket in all_buckets:
            if bucket not in bucket_info['expected_prob']:
                bucket_info['expected_prob'][bucket] = epsilon
            if bucket not in bucket_info['actual_prob']:
                bucket_info['actual_prob'][bucket] = epsilon

        # Calculate PSI including missing values
        psi = sum((actual_prob - expected_prob) * np.log(actual_prob / expected_prob) 
                 for bucket, expected_prob in bucket_info['expected_prob'].items() 
                 for actual_prob in [bucket_info['actual_prob'].get(bucket, epsilon)])

        return psi, bucket_info

    def load_input_data(self, input_table: str, date_value: str) -> pd.DataFrame:
        """Load data from either a Hive table or parquet file.
        
        Args:
            input_table: Path to parquet file or Hive table name
            date_value: Date value to filter data
            
        Returns:
            DataFrame containing the loaded data
        """
        return pd.read_parquet(input_table) if input_table.endswith('.parquet') else \
               self.spark.sql(f"SELECT * FROM {input_table} WHERE {self.date_column} = '{date_value}'").toPandas()

    def load_expected_distribution_from_file(self, config_path: str) -> Dict[str, Dict[str, Any]]:
        """Load expected distribution configuration from a YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Dictionary containing expected distributions for each column
        """
        with open(config_path, 'r') as f:
            return {column: {'expected_prob': settings['expected_prob'], 'breakpoints': settings.get('breakpoints')} 
                   for column, settings in yaml.safe_load(f)['columns'].items()}

    def load_expected_distribution_from_table(self, output_bucket_table: str, reference_date: str) -> Dict[str, Dict[str, Any]]:
        """Load expected distribution from the expected bucket Hive table.
        
        Args:
            output_bucket_table: Name of the Hive table containing bucket information
            reference_date: Date to use as reference for expected distribution
            
        Returns:
            Dictionary containing expected distributions for each column
        """
        df = self.spark.sql(f"SELECT column_name, column_type, bucket_value, expected_prob FROM {output_bucket_table} WHERE grass_date = '{reference_date}'").toPandas()
        if df.empty: raise ValueError(f"No expected distributions found for reference date: {reference_date}")
        
        expected_distributions = {}
        for column, column_data in df.groupby('column_name'):
            # Convert bucket values to probabilities dictionary
            expected_prob = dict(zip(column_data['bucket_value'], column_data['expected_prob']))
            
            # Generate breakpoints from bucket values if numeric
            breakpoints = None
            if column_data['column_type'].iloc[0] == 'numeric':
                # Get all bucket values except NULL
                bucket_values = [val for val in column_data['bucket_value'] if val != 'NULL']
                if all(isinstance(val, str) and '(' in val and ']' in val for val in bucket_values):
                    try:
                        # Extract numbers from interval strings
                        numbers = []
                        for val in bucket_values:
                            # Handle special cases for infinity
                            if 'inf' in val.lower():
                                if '-inf' in val.lower():
                                    numbers.append(float('-inf'))
                                else:
                                    numbers.append(float('inf'))
                            else:
                                # Extract numbers from string like "(0.0, 1.0]"
                                nums = [float(x) for x in val.replace('(', '').replace(']', '').split(',')]
                                numbers.extend(nums)
                        breakpoints = sorted(set(numbers))
                    except (ValueError, IndexError):
                        breakpoints = None
            
            expected_distributions[column] = {
                'expected_prob': expected_prob,
                'breakpoints': breakpoints
            }
        
        return expected_distributions

    def save_bucket_results_to_table(self, output_table: str, column_name: str, bucket_info: Dict[str, Any], 
                                   reference_date: str, target_date: str) -> None:
        """Save bucket distribution results to a Hive table.
        
        Args:
            output_table: Name of the Hive table to save results
            column_name: Name of the column being analyzed
            bucket_info: Dictionary containing bucket information
            reference_date: Reference date used for expected distribution
            target_date: Target date being analyzed
        """
        schema = StructType([
            StructField("column_name", StringType(), False),
            StructField("column_type", StringType(), False),
            StructField("bucket_index", IntegerType(), False),
            StructField("bucket_value", StringType(), False),  # Always store as string
            StructField("expected_prob", DoubleType(), False),
            StructField("actual_prob", DoubleType(), False),
            StructField("grass_date", StringType(), False)
        ])
        
        # Determine column type based on breakpoints
        column_type = 'numeric' if bucket_info['breakpoints'] else 'categorical'
        
        # Create data with bucket index
        data = []
        for idx, (bucket, expected_prob) in enumerate(bucket_info['expected_prob'].items()):
            # Convert bucket value to string, handling different types
            if isinstance(bucket, str):
                bucket_str = bucket
            elif pd.isna(bucket):
                bucket_str = 'NULL'
            else:
                bucket_str = str(bucket)
            
            data.append({
                'column_name': column_name,
                'column_type': column_type,
                'bucket_index': idx,
                'bucket_value': bucket_str,
                'expected_prob': float(expected_prob),
                'actual_prob': float(bucket_info['actual_prob'].get(bucket, 0.0)),
                'grass_date': target_date
            })
        
        self.spark.createDataFrame(data, schema).write.mode("append").partitionBy("grass_date").saveAsTable(output_table)
        logger.info(f"Saved bucket results to partitioned Hive table: {output_table}")

    def save_psi_results_to_table(self, output_table: str, drift_results: Dict[str, Dict[str, Any]], target_date: str) -> None:
        """Save PSI results to a Hive table.
        
        Args:
            output_table: Name of the Hive table to save results
            drift_results: Dictionary containing PSI results for each column
            target_date: Target date being analyzed
        """
        schema = StructType([StructField("column_name", StringType(), False), StructField("target_date", StringType(), False),
                           StructField("psi", DoubleType(), True), StructField("target_count", IntegerType(), False),
                           StructField("target_missing", IntegerType(), False)])
        
        data = [{'column_name': column, 'target_date': target_date, 'psi': float(result['psi']) if result['psi'] is not None else None,
                'target_count': int(result['target_date_count']), 'target_missing': int(result['target_date_missing'])}
                for column, result in drift_results.items() if 'error' not in result]
        
        self.spark.createDataFrame(data, schema).write.mode("append").partitionBy("target_date").saveAsTable(output_table)
        logger.info(f"Saved PSI results to partitioned Hive table: {output_table}")


    def monitor_feature_drift(self, input_table: str, target_date: str, 
                            expected_config: Optional[str] = None, reference_date: Optional[str] = None,
                            output_bucket_table: Optional[str] = None, output_psi_table: Optional[str] = None
                            ) -> Dict[str, Dict[str, Any]]:
        """Monitor feature drift between expected and actual distributions.
        
        Args:
            input_table: Path to input data (Hive table or parquet file)
            target_date: Date to analyze
            expected_config: Path to YAML file with expected distributions
            reference_date: Reference date to use for expected distribution
            output_bucket_table: Name of the Hive table to save bucket results
            output_psi_table: Name of the Hive table to save PSI results
            
        Returns:
            Dictionary containing drift results for each column
        """
        try:
            expected_distributions = self.load_expected_distribution_from_file(expected_config) if expected_config else \
                                   self.load_expected_distribution_from_table(output_bucket_table, reference_date) if reference_date and output_bucket_table else \
                                   None
            if not expected_distributions: raise ValueError("Either expected_config or reference_date with output_bucket_table must be provided")

            target_data = self.load_input_data(input_table, target_date)
            if target_data.empty: raise ValueError("No data found for target date")

            drift_results = {}
            for column in target_data.columns:
                if column != self.date_column:
                    try:
                        if column in expected_distributions:
                            expected = expected_distributions[column]['expected_prob']
                            breakpoints = expected_distributions[column].get('breakpoints')
                            psi, bucket_info = self.calculate_psi(expected, target_data[column], breakpoints=breakpoints)
                            
                            drift_results[column] = {'psi': psi, 'target_date_count': len(target_data[column]),
                                                   'target_date_missing': target_data[column].isna().sum(),
                                                   'bucket_info': bucket_info}
                            
                            if output_bucket_table:
                                self.save_bucket_results_to_table(output_bucket_table, column, bucket_info,
                                                                "expected" if expected_config else reference_date, target_date)
                            

                        else:
                            logger.warning(f"No expected distribution found for column: {column}")
                            drift_results[column] = {'psi': None, 'error': f"No expected distribution found for column: {column}"}
                            
                    except Exception as e:
                        logger.warning(f"Error calculating PSI for column {column}: {str(e)}")
                        drift_results[column] = {'psi': None, 'error': str(e)}

            if output_psi_table: self.save_psi_results_to_table(output_psi_table, drift_results, target_date)
            return drift_results

        except Exception as e:
            logger.error(f"Error monitoring feature drift: {str(e)}")
            raise

    def __del__(self):
        """Clean up Spark session when the object is destroyed."""
        if hasattr(self, 'spark'): self.spark.stop()

def main():
    """Main function to run feature drift monitoring."""
    parser = argparse.ArgumentParser(description='Monitor feature drift using PSI')
    parser.add_argument('--input-table', type=str, required=True, help='Input Hive table name or parquet file path')
    parser.add_argument('--output-bucket-table', type=str, help='Name of the Hive table to save bucket results')
    parser.add_argument('--output-psi-table', type=str, help='Name of the Hive table to save PSI results')
    parser.add_argument('--target-date', type=str, required=True, help='Target date to analyze (format: YYYY-MM-DD)')
    parser.add_argument('--reference-date', type=str, help='Reference date to use for expected distribution')
    parser.add_argument('--expected-config', type=str, help='Path to YAML file with expected distributions')
    
    args = parser.parse_args()
    if args.expected_config and args.reference_date:
        raise ValueError("Cannot specify both --expected-config and --reference-date")
    
    monitor = FeatureDriftMonitor()
    try:
        drift_results = monitor.monitor_feature_drift(input_table=args.input_table, target_date=args.target_date,
                                                    expected_config=args.expected_config, reference_date=args.reference_date,
                                                    output_bucket_table=args.output_bucket_table, output_psi_table=args.output_psi_table)
        
        print("\nFeature Drift Analysis Results")
        print("=" * 100)
        print(f"{'Column':<30} {'PSI':<10} {'Target Count':<12} {'Target Missing':<12}")
        print("-" * 100)
        
        for column, result in drift_results.items():
            if 'error' in result:
                print(f"{column:<30} {'N/A':<10} {'N/A':<12} {'N/A':<12}")
            else:
                print(f"{column:<30} {result['psi']:.4f if result['psi'] is not None else 'N/A':<10} "
                      f"{result['target_date_count']:<12} {result['target_date_missing']:<12}")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise
    finally:
        if 'monitor' in locals(): monitor.__del__()

if __name__ == "__main__":
    main() 