import pandas as pd
import os
import re
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    # 定义所有关键词
    DATA_QUERY_KEYWORDS = {
        'general': ['data', 'query', 'calculate', 'transaction', 'transactions', 'group', 'grouped', 'grouping', 'group by', 
                   '数据', '查询', '计算', '交易', '金额', '查找', '统计', '分组', '按照', '根据'],
        'min': ['min', 'minimum', 'smallest', 'lowest', 'least', '最小', '最小值', '最低', '最少'],
        'max': ['max', 'maximum', 'largest', 'highest', 'greatest', 'most', '最大', '最大值', '最高', '最多'],
        'sum': ['sum', 'total', 'add up', 'summation', '总和', '合计', '求和', '加总'],
        'mean': ['average', 'mean', 'avg', '平均', '平均值', '均值'],
        'count': ['count', 'number', 'quantity', 'how many', '数量', '个数', '多少个', '计数'],
        'median': ['median', 'middle', '中位数', '中值'],
        'std': ['std', 'standard deviation', 'deviation', '标准差', '方差'],
        'variance': ['variance', 'var', '方差'],
        'percentile': ['percentile', 'quantile', '百分位', '分位数'],
        'range': ['range', 'span', 'difference between max and min', '范围', '区间'],
        'growth': ['growth', 'increase', 'rate', 'change', '增长', '增长率', '变化率']
    }

    def __init__(self, csv_path: str = "sample_txns.csv"):
        """
        Initialize the data processor with the path to the CSV file
        """
        self.csv_path = csv_path
        self.df = None
        self.load_data()
    
    def load_data(self) -> None:
        """
        Load data from CSV file with multiple retry attempts and paths
        """
        logger.info("Starting data loading process")
        
        # 定义可能的路径
        possible_paths = [
            self.csv_path,
            os.path.join(os.getcwd(), self.csv_path),
            os.path.join(os.getcwd(), "gambling_prediction_app_demo_api_regular_expressions", self.csv_path),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), self.csv_path)
        ]
        
        errors = []
        for path in possible_paths:
            try:
                logger.info(f"Attempting to load data from: {path}")
                if not os.path.exists(path):
                    logger.warning(f"File not found at: {path}")
                    continue
                    
                self.df = pd.read_csv(path)
                
                # 确保日期列是字符串类型
                self.df['dateObj'] = self.df['dateObj'].astype(str)
                
                logger.info(f"Successfully loaded data from {path}")
                logger.info(f"DataFrame shape: {self.df.shape}")
                logger.info(f"DataFrame columns: {self.df.columns.tolist()}")
                logger.info(f"Sample data:\n{self.df.head()}")
                return
                
            except Exception as e:
                error_msg = f"Error loading data from {path}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        # 如果所有尝试都失败了，抛出异常
        raise FileNotFoundError(f"Failed to load data from any path. Errors: {'; '.join(errors)}")
    
    def is_data_query(self, query: str) -> bool:
        """
        检查是否是数据查询
        """
        query_lower = query.lower()
        
        # 检查是否包含任何数据查询关键词
        has_keywords = any(
            keyword in query_lower 
            for keywords in self.DATA_QUERY_KEYWORDS.values() 
            for keyword in keywords
        )
        
        # 检查是否包含日期
        date_pattern = r'\d{4}-\d{2}-\d{2}'
        has_date = bool(re.search(date_pattern, query))
        
        # 检查是否包含分组查询的特定模式
        group_patterns = [
            r'group\s+by',
            r'grouped\s+by',
            r'按照',
            r'根据',
            r'分组'
        ]
        has_group = any(re.search(pattern, query_lower) for pattern in group_patterns)
        
        # 检查是否是简短的计算查询（如 "Sum of transactions"）
        calc_patterns = [
            r'^sum\s+of',
            r'^average\s+of',
            r'^mean\s+of',
            r'^min\s+of',
            r'^max\s+of',
            r'^count\s+of',
            r'^total\s+of',
            r'^median\s+of',
            r'^std\s+of',
            r'^variance\s+of',
            r'^range\s+of'
        ]
        is_calc_query = any(re.search(pattern, query_lower) for pattern in calc_patterns)
        
        logger.info(f"Query '{query}' has keywords: {has_keywords}, has date: {has_date}, has group: {has_group}, is calc query: {is_calc_query}")
        return has_keywords or has_date or has_group or is_calc_query

    def extract_query_params(self, query: str) -> Dict[str, Any]:
        """
        Extract query parameters from user query using regex
        """
        params = {}
        query_lower = query.lower()
        
        # 检查计算类型
        logger.info(f"Checking calculation type in query: {query_lower}")
        
        # 检查各种计算类型的关键词
        calculation_types = {
            'min': False,
            'max': False,
            'sum': False,
            'mean': False,
            'count': False,
            'median': False,
            'std': False,
            'variance': False,
            'percentile': False,
            'range': False,
            'growth': False
        }
        
        # 检查每种计算类型的关键词
        for calc_type in calculation_types:
            if calc_type in self.DATA_QUERY_KEYWORDS:
                # 检查关键词是否在查询中
                calculation_types[calc_type] = any(word in query_lower for word in self.DATA_QUERY_KEYWORDS[calc_type])
                # 特别检查 "sum of" 这样的模式
                if not calculation_types[calc_type] and calc_type == 'sum' and query_lower.startswith('sum of'):
                    calculation_types[calc_type] = True
                elif not calculation_types[calc_type] and calc_type == 'mean' and (query_lower.startswith('average of') or query_lower.startswith('mean of')):
                    calculation_types[calc_type] = True
                elif not calculation_types[calc_type] and calc_type == 'min' and query_lower.startswith('min of'):
                    calculation_types[calc_type] = True
                elif not calculation_types[calc_type] and calc_type == 'max' and query_lower.startswith('max of'):
                    calculation_types[calc_type] = True
                elif not calculation_types[calc_type] and calc_type == 'count' and query_lower.startswith('count of'):
                    calculation_types[calc_type] = True
        
        logger.info(f"Found calculation types: {calculation_types}")
        
        # 按优先级设置计算类型
        if calculation_types['min']:
            logger.info("Found calculation type: min")
            params['calculation'] = 'min'
        elif calculation_types['max']:
            logger.info("Found calculation type: max")
            params['calculation'] = 'max'
        elif calculation_types['sum']:
            logger.info("Found calculation type: sum")
            params['calculation'] = 'sum'
        elif calculation_types['mean']:
            logger.info("Found calculation type: mean")
            params['calculation'] = 'mean'
        elif calculation_types['count']:
            logger.info("Found calculation type: count")
            params['calculation'] = 'count'
        elif calculation_types['median']:
            logger.info("Found calculation type: median")
            params['calculation'] = 'median'
        elif calculation_types['std']:
            logger.info("Found calculation type: std")
            params['calculation'] = 'std'
        elif calculation_types['variance']:
            logger.info("Found calculation type: variance")
            params['calculation'] = 'variance'
        elif calculation_types['percentile']:
            logger.info("Found calculation type: percentile")
            params['calculation'] = 'percentile'
        elif calculation_types['range']:
            logger.info("Found calculation type: range")
            params['calculation'] = 'range'
        elif calculation_types['growth']:
            logger.info("Found calculation type: growth")
            params['calculation'] = 'growth'
        else:
            logger.info("No calculation type found in query")
        
        # 设置默认列为 amount
        params['column'] = 'amount'
        
        # 提取日期范围
        date_pattern = r'(\d{4}-\d{2}-\d{2})'
        dates = re.findall(date_pattern, query)
        if len(dates) >= 2:
            params['start_date'] = dates[0]
            params['end_date'] = dates[1]
            logger.info(f"Found date range: {dates[0]} to {dates[1]}")
        elif len(dates) == 1:
            params['start_date'] = dates[0]
            params['end_date'] = dates[0]
            logger.info(f"Found single date: {dates[0]}")
        
        # 检查是否包含 "gambling" 关键词
        if 'gambling' in query_lower:
            logger.info("Query contains 'gambling', filtering by label=gambling")
            params['label'] = 'gambling'
        
        # Extract transaction type with support for Chinese
        type_pattern = r'(?:type|transaction type|类型|交易类型)\s+(?:is|=|:|为|是)\s+(\w+)'
        type_match = re.search(type_pattern, query, re.IGNORECASE)
        if type_match:
            params['type'] = type_match.group(1).upper()
        
        # Extract label with support for Chinese
        label_pattern = r'(?:label|category|标签|类别)\s+(?:is|=|:|为|是)\s+(\w+)'
        label_match = re.search(label_pattern, query, re.IGNORECASE)
        if label_match:
            params['label'] = label_match.group(1).lower()
        
        # Extract account UUID with support for Chinese
        account_pattern = r'(?:account|账户|账号)\s+(?:is|=|:|为|是)\s+([a-zA-Z0-9-]+)'
        account_match = re.search(account_pattern, query, re.IGNORECASE)
        if account_match:
            params['account_uuid'] = account_match.group(1)
        
        # Extract merchant UUID with support for Chinese
        merchant_pattern = r'(?:merchant|商户|商家)\s+(?:is|=|:|为|是)\s+([a-zA-Z0-9-_]+)'
        merchant_match = re.search(merchant_pattern, query, re.IGNORECASE)
        if merchant_match:
            params['merchant_uuid'] = merchant_match.group(1)
        
        # 增强对分组查询的识别
        group_patterns = [
            r'group\s+by\s+(\w+)',
            r'grouped\s+by\s+(\w+)',
            r'按照\s*(\w+)',
            r'根据\s*(\w+)',
            r'分组\s*(?:按|by|根据)?\s*(\w+)'
        ]
        
        for pattern in group_patterns:
            group_match = re.search(pattern, query_lower)
            if group_match:
                group_column = group_match.group(1).lower()
                # 处理常见的分组列名
                if group_column in ['type', 'transaction type', 'transaction_type', '类型', '交易类型']:
                    params['group_by'] = 'type'
                elif group_column in ['label', 'category', '标签', '类别']:
                    params['group_by'] = 'label'
                elif group_column in ['date', 'day', 'time', '日期', '时间', '天']:
                    params['group_by'] = 'dateObj'
                elif group_column in ['account', 'account_uuid', '账户', '账号']:
                    params['group_by'] = 'account_dot_uuid'
                elif group_column in ['merchant', 'merchant_uuid', '商户', '商家']:
                    params['group_by'] = 'merchant_dot_uuid'
                else:
                    params['group_by'] = group_column
                
                logger.info(f"Found group by: {params['group_by']}")
                break
        
        # 特殊处理 "Sum of transactions group by type" 这样的查询
        if 'group_by' not in params and 'type' in query_lower and ('group' in query_lower or 'grouped' in query_lower):
            logger.info("Query contains 'type' and 'group', assuming group by type")
            params['group_by'] = 'type'
        
        # 如果查询中包含 "group by type" 或类似表达，但没有指定计算类型，默认为 sum
        if 'group_by' in params and 'calculation' not in params:
            logger.info("Query contains group by but no calculation type, defaulting to sum")
            params['calculation'] = 'sum'
        
        logger.info(f"Extracted parameters: {params}")
        return params
    
    def query_data(self, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Query data based on parameters
        """
        try:
            logger.info(f"Querying data with parameters: {params}")
            
            if self.df is None:
                logger.warning("DataFrame is None, attempting to reload data")
                self.load_data()
            
            if self.df is None or self.df.empty:
                raise ValueError("No data available for querying")
            
            # 创建过滤条件
            mask = pd.Series(True, index=self.df.index)
            
            # 日期过滤
            if 'start_date' in params:
                # 将日期字符串转换为完整的ISO格式
                start_date = f"{params['start_date']}T12:00:00.000Z"
                mask &= (self.df['dateObj'] >= start_date)
                logger.info(f"Filtering by start date: {start_date}")
            if 'end_date' in params:
                # 将日期字符串转换为完整的ISO格式
                end_date = f"{params['end_date']}T12:00:00.000Z"
                mask &= (self.df['dateObj'] <= end_date)
                logger.info(f"Filtering by end date: {end_date}")
            
            # 交易类型过滤
            if 'type' in params:
                mask &= (self.df['type'] == params['type'])
                logger.info(f"Filtering by type: {params['type']}")
            
            # 标签过滤
            if 'label' in params:
                mask &= (self.df['label'] == params['label'])
                logger.info(f"Filtering by label: {params['label']}")
            
            # 账户过滤
            if 'account_uuid' in params:
                mask &= (self.df['account_dot_uuid'] == params['account_uuid'])
                logger.info(f"Filtering by account UUID: {params['account_uuid']}")
            
            # 商户过滤
            if 'merchant_uuid' in params:
                mask &= (self.df['merchant_dot_uuid'] == params['merchant_uuid'])
                logger.info(f"Filtering by merchant UUID: {params['merchant_uuid']}")
            
            filtered_df = self.df[mask].copy()
            logger.info(f"Query returned {len(filtered_df)} rows")
            logger.info(f"Filtered data sample:\n{filtered_df.head()}")
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error querying data: {str(e)}", exc_info=True)
            raise
    
    def save_temp_data(self, df: pd.DataFrame, filename: str = "temp_data.csv") -> str:
        """
        Save filtered data to a temporary CSV file
        
        Parameters:
        - df: DataFrame to save
        - filename: Name of the temporary file
        
        Returns:
        - Path to the saved file
        """
        temp_path = os.path.join(os.path.dirname(self.csv_path), filename)
        df.to_csv(temp_path, index=False)
        logger.info(f"Temporary data saved to {temp_path}")
        return temp_path
    
    def delete_temp_data(self, filepath: str) -> None:
        """
        Delete temporary data file
        
        Parameters:
        - filepath: Path to the temporary file
        """
        try:
            os.remove(filepath)
            logger.info(f"Temporary file {filepath} deleted")
        except Exception as e:
            logger.error(f"Error deleting temporary file: {str(e)}") 