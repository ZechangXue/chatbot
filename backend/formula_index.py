import pandas as pd
import logging
from typing import Dict, Any, Callable, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FormulaIndex:
    """
    A local formula index library for performing calculations on DataFrames
    """
    
    def __init__(self):
        """
        Initialize the formula index with predefined calculation functions
        """
        self.formulas = {
            'sum': self.calculate_sum,
            'mean': self.calculate_mean,
            'max': self.calculate_max,
            'min': self.calculate_min,
            'count': self.calculate_count,
            'median': self.calculate_median,
            'std': self.calculate_std,
            'variance': self.calculate_variance,
            'range': self.calculate_range
        }
    
    def get_available_formulas(self) -> List[str]:
        """
        Get a list of available formula names
        
        Returns:
        - List of formula names
        """
        return list(self.formulas.keys())
    
    def calculate(self, df: pd.DataFrame, formula_name: str, column: str = 'amount', group_by: str = None) -> Dict[str, Any]:
        """
        Apply a calculation formula to the DataFrame
        
        Parameters:
        - df: DataFrame to perform calculation on
        - formula_name: Name of the formula to apply
        - column: Column to perform calculation on
        - group_by: Column to group by before calculation
        
        Returns:
        - Dictionary with calculation results
        """
        logger.info(f"Calculating {formula_name} on column {column} with group_by={group_by}")
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"DataFrame columns: {df.columns.tolist()}")
        
        # 检查DataFrame是否为空
        if df.empty:
            logger.warning("DataFrame is empty, cannot perform calculation")
            return {"error": "No data available for calculation", "formula": formula_name, "column": column}
        
        # 检查列是否存在
        if column not in df.columns:
            logger.error(f"Column '{column}' not found in DataFrame")
            return {"error": f"Column '{column}' not found", "formula": formula_name, "available_columns": df.columns.tolist()}
        
        # 检查分组列是否存在
        if group_by and group_by not in df.columns:
            logger.error(f"Group by column '{group_by}' not found in DataFrame")
            return {"error": f"Group by column '{group_by}' not found", "formula": formula_name, "available_columns": df.columns.tolist()}
        
        if formula_name not in self.formulas:
            logger.error(f"Formula '{formula_name}' not found in index")
            return {"error": f"Formula '{formula_name}' not found", "available_formulas": list(self.formulas.keys())}
        
        try:
            formula_func = self.formulas[formula_name]
            result = formula_func(df, column, group_by)
            logger.info(f"Calculation result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error applying formula '{formula_name}': {str(e)}", exc_info=True)
            return {"error": f"Error applying formula: {str(e)}", "formula": formula_name, "column": column}
    
    def calculate_sum(self, df: pd.DataFrame, column: str = 'amount', group_by: str = None) -> Dict[str, Any]:
        """
        Calculate the sum of a column, optionally grouped by another column
        
        Parameters:
        - df: DataFrame to perform calculation on
        - column: Column to sum
        - group_by: Column to group by before summing
        
        Returns:
        - Dictionary with sum results
        """
        try:
            if group_by and group_by in df.columns:
                # Group by the specified column and calculate sum
                grouped = df.groupby(group_by)[column].sum().reset_index()
                result = {
                    "formula": "sum",
                    "column": column,
                    "group_by": group_by,
                    "result_type": "grouped",
                    "results": grouped.to_dict(orient='records')
                }
            else:
                # Calculate overall sum
                total_sum = df[column].sum()
                result = {
                    "formula": "sum",
                    "column": column,
                    "result_type": "single",
                    "result": float(total_sum)  # Convert to float for JSON serialization
                }
            
            logger.info(f"Sum calculation completed: {result}")
            return result
        except Exception as e:
            logger.error(f"Error calculating sum: {str(e)}")
            return {"error": f"Error calculating sum: {str(e)}"}
    
    def calculate_mean(self, df: pd.DataFrame, column: str = 'amount', group_by: str = None) -> Dict[str, Any]:
        """
        Calculate the mean of a column, optionally grouped by another column
        
        Parameters:
        - df: DataFrame to perform calculation on
        - column: Column to calculate mean for
        - group_by: Column to group by before calculating mean
        
        Returns:
        - Dictionary with mean results
        """
        try:
            if group_by and group_by in df.columns:
                # Group by the specified column and calculate mean
                grouped = df.groupby(group_by)[column].mean().reset_index()
                result = {
                    "formula": "mean",
                    "column": column,
                    "group_by": group_by,
                    "result_type": "grouped",
                    "results": grouped.to_dict(orient='records')
                }
            else:
                # Calculate overall mean
                mean_value = df[column].mean()
                result = {
                    "formula": "mean",
                    "column": column,
                    "result_type": "single",
                    "result": float(mean_value)  # Convert to float for JSON serialization
                }
            
            logger.info(f"Mean calculation completed: {result}")
            return result
        except Exception as e:
            logger.error(f"Error calculating mean: {str(e)}")
            return {"error": f"Error calculating mean: {str(e)}"}
    
    def calculate_max(self, df: pd.DataFrame, column: str = 'amount', group_by: str = None) -> Dict[str, Any]:
        """
        Calculate the maximum value of a column, optionally grouped by another column
        
        Parameters:
        - df: DataFrame to perform calculation on
        - column: Column to calculate max for
        - group_by: Column to group by before calculating max
        
        Returns:
        - Dictionary with max results
        """
        try:
            if group_by and group_by in df.columns:
                # Group by the specified column and calculate max
                grouped = df.groupby(group_by)[column].max().reset_index()
                result = {
                    "formula": "max",
                    "column": column,
                    "group_by": group_by,
                    "result_type": "grouped",
                    "results": grouped.to_dict(orient='records')
                }
            else:
                # Calculate overall max
                max_value = df[column].max()
                # Get the record with the maximum value
                max_record = df.loc[df[column] == max_value].iloc[0].to_dict()
                result = {
                    "formula": "max",
                    "column": column,
                    "result_type": "single",
                    "result": float(max_value),  # Convert to float for JSON serialization
                    "record": max_record  # Include the full record for context
                }
            
            logger.info(f"Max calculation completed: {result}")
            return result
        except Exception as e:
            logger.error(f"Error calculating max: {str(e)}")
            return {"error": f"Error calculating max: {str(e)}"}
    
    def calculate_min(self, df: pd.DataFrame, column: str = 'amount', group_by: str = None) -> Dict[str, Any]:
        """
        Calculate the minimum value of a column, optionally grouped by another column
        
        Parameters:
        - df: DataFrame to perform calculation on
        - column: Column to calculate min for
        - group_by: Column to group by before calculating min
        
        Returns:
        - Dictionary with min results
        """
        try:
            if group_by and group_by in df.columns:
                # Group by the specified column and calculate min
                grouped = df.groupby(group_by)[column].min().reset_index()
                result = {
                    "formula": "min",
                    "column": column,
                    "group_by": group_by,
                    "result_type": "grouped",
                    "results": grouped.to_dict(orient='records')
                }
            else:
                # Calculate overall min
                min_value = df[column].min()
                # Get the record with the minimum value
                min_record = df.loc[df[column] == min_value].iloc[0].to_dict()
                result = {
                    "formula": "min",
                    "column": column,
                    "result_type": "single",
                    "result": float(min_value),  # Convert to float for JSON serialization
                    "record": min_record  # Include the full record for context
                }
            
            logger.info(f"Min calculation completed: {result}")
            return result
        except Exception as e:
            logger.error(f"Error calculating min: {str(e)}")
            return {"error": f"Error calculating min: {str(e)}"}
    
    def calculate_count(self, df: pd.DataFrame, column: str = 'amount', group_by: str = None) -> Dict[str, Any]:
        """
        Calculate the count of a column, optionally grouped by another column
        
        Parameters:
        - df: DataFrame to perform calculation on
        - column: Column to count
        - group_by: Column to group by before counting
        
        Returns:
        - Dictionary with count results
        """
        try:
            if group_by and group_by in df.columns:
                # Group by the specified column and calculate count
                grouped = df.groupby(group_by)[column].count().reset_index()
                result = {
                    "formula": "count",
                    "column": column,
                    "group_by": group_by,
                    "result_type": "grouped",
                    "results": grouped.to_dict(orient='records')
                }
            else:
                # Calculate count without grouping
                count = len(df)
                result = {
                    "formula": "count",
                    "column": column,
                    "result_type": "single",
                    "result": count
                }
            
            logger.info(f"Count calculation completed: {result}")
            return result
        except Exception as e:
            logger.error(f"Error calculating count: {str(e)}", exc_info=True)
            return {"error": f"Error calculating count: {str(e)}", "formula": "count", "column": column}
    
    def calculate_median(self, df: pd.DataFrame, column: str = 'amount', group_by: str = None) -> Dict[str, Any]:
        """
        Calculate the median of a column, optionally grouped by another column
        
        Parameters:
        - df: DataFrame to perform calculation on
        - column: Column to calculate median
        - group_by: Column to group by before calculating median
        
        Returns:
        - Dictionary with median results
        """
        try:
            if group_by and group_by in df.columns:
                # Group by the specified column and calculate median
                grouped = df.groupby(group_by)[column].median().reset_index()
                result = {
                    "formula": "median",
                    "column": column,
                    "group_by": group_by,
                    "result_type": "grouped",
                    "results": grouped.to_dict(orient='records')
                }
            else:
                # Calculate median without grouping
                median = df[column].median()
                result = {
                    "formula": "median",
                    "column": column,
                    "result_type": "single",
                    "result": float(median)
                }
            
            logger.info(f"Median calculation completed: {result}")
            return result
        except Exception as e:
            logger.error(f"Error calculating median: {str(e)}", exc_info=True)
            return {"error": f"Error calculating median: {str(e)}", "formula": "median", "column": column}
    
    def calculate_std(self, df: pd.DataFrame, column: str = 'amount', group_by: str = None) -> Dict[str, Any]:
        """
        Calculate the standard deviation of a column, optionally grouped by another column
        
        Parameters:
        - df: DataFrame to perform calculation on
        - column: Column to calculate standard deviation
        - group_by: Column to group by before calculating standard deviation
        
        Returns:
        - Dictionary with standard deviation results
        """
        try:
            if group_by and group_by in df.columns:
                # Group by the specified column and calculate standard deviation
                grouped = df.groupby(group_by)[column].std().reset_index()
                result = {
                    "formula": "std",
                    "column": column,
                    "group_by": group_by,
                    "result_type": "grouped",
                    "results": grouped.to_dict(orient='records')
                }
            else:
                # Calculate standard deviation without grouping
                std = df[column].std()
                result = {
                    "formula": "std",
                    "column": column,
                    "result_type": "single",
                    "result": float(std)
                }
            
            logger.info(f"Standard deviation calculation completed: {result}")
            return result
        except Exception as e:
            logger.error(f"Error calculating standard deviation: {str(e)}", exc_info=True)
            return {"error": f"Error calculating standard deviation: {str(e)}", "formula": "std", "column": column}
    
    def calculate_variance(self, df: pd.DataFrame, column: str = 'amount', group_by: str = None) -> Dict[str, Any]:
        """
        Calculate the variance of a column, optionally grouped by another column
        
        Parameters:
        - df: DataFrame to perform calculation on
        - column: Column to calculate variance
        - group_by: Column to group by before calculating variance
        
        Returns:
        - Dictionary with variance results
        """
        try:
            if group_by and group_by in df.columns:
                # Group by the specified column and calculate variance
                grouped = df.groupby(group_by)[column].var().reset_index()
                result = {
                    "formula": "variance",
                    "column": column,
                    "group_by": group_by,
                    "result_type": "grouped",
                    "results": grouped.to_dict(orient='records')
                }
            else:
                # Calculate variance without grouping
                var = df[column].var()
                result = {
                    "formula": "variance",
                    "column": column,
                    "result_type": "single",
                    "result": float(var)
                }
            
            logger.info(f"Variance calculation completed: {result}")
            return result
        except Exception as e:
            logger.error(f"Error calculating variance: {str(e)}", exc_info=True)
            return {"error": f"Error calculating variance: {str(e)}", "formula": "variance", "column": column}
    
    def calculate_range(self, df: pd.DataFrame, column: str = 'amount', group_by: str = None) -> Dict[str, Any]:
        """
        Calculate the range (max - min) of a column, optionally grouped by another column
        
        Parameters:
        - df: DataFrame to perform calculation on
        - column: Column to calculate range
        - group_by: Column to group by before calculating range
        
        Returns:
        - Dictionary with range results
        """
        try:
            if group_by and group_by in df.columns:
                # Group by the specified column and calculate range
                grouped_max = df.groupby(group_by)[column].max().reset_index()
                grouped_min = df.groupby(group_by)[column].min().reset_index()
                grouped = pd.merge(grouped_max, grouped_min, on=group_by, suffixes=('_max', '_min'))
                grouped['range'] = grouped[f'{column}_max'] - grouped[f'{column}_min']
                result = {
                    "formula": "range",
                    "column": column,
                    "group_by": group_by,
                    "result_type": "grouped",
                    "results": grouped.to_dict(orient='records')
                }
            else:
                # Calculate range without grouping
                max_val = df[column].max()
                min_val = df[column].min()
                range_val = max_val - min_val
                result = {
                    "formula": "range",
                    "column": column,
                    "result_type": "single",
                    "result": float(range_val),
                    "max": float(max_val),
                    "min": float(min_val)
                }
            
            logger.info(f"Range calculation completed: {result}")
            return result
        except Exception as e:
            logger.error(f"Error calculating range: {str(e)}", exc_info=True)
            return {"error": f"Error calculating range: {str(e)}", "formula": "range", "column": column} 