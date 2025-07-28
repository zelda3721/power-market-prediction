"""
数据预处理器模块
提供特征工程和数据预处理功能
"""
import logging
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_columns = []
        self.target_columns = ['day_ahead_price', 'real_time_price']
        self.categorical_columns = []
        self.numerical_columns = []
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建时间特征
        
        Args:
            df: 输入数据框
            
        Returns:
            添加了时间特征的数据框
        """
        df = df.copy()
        
        # 确保日期列是datetime类型
        date_col = None
        for col in ['date', 'target_date', 'forecast_target_date']:
            if col in df.columns:
                date_col = col
                df[date_col] = pd.to_datetime(df[date_col])
                break
                
        if date_col is None:
            logger.warning("未找到日期列")
            return df
            
        # 基础时间特征
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['dayofweek'] = df[date_col].dt.dayofweek
        df['quarter'] = df[date_col].dt.quarter
        df['dayofyear'] = df[date_col].dt.dayofyear
        df['weekofyear'] = df[date_col].dt.isocalendar().week
        
        # 时间点特征
        if 'time_point' in df.columns:
            df['hour'] = df['time_point'].str.split(':').str[0].astype(int)
            df['minute'] = df['time_point'].str.split(':').str[1].astype(int)
            
            # 计算时间间隔（如果不存在）
            if 'time_interval' not in df.columns:
                df['time_interval'] = df['hour'] * 4 + df['minute'] // 15 + 1
        
        # 周期性编码（使用sin/cos避免不连续性）
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        if 'hour' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
            df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
        
        # 工作日/周末标识
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['is_monday'] = (df['dayofweek'] == 0).astype(int)
        df['is_friday'] = (df['dayofweek'] == 4).astype(int)
        
        # 时段特征
        if 'hour' in df.columns:
            df['is_peak_hour'] = df['hour'].isin([9, 10, 11, 18, 19, 20]).astype(int)
            df['is_valley_hour'] = df['hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)
            df['is_flat_hour'] = (~df['hour'].isin([9, 10, 11, 18, 19, 20, 0, 1, 2, 3, 4, 5])).astype(int)
        
        logger.info(f"创建了 {len(df.columns) - len(df.columns)} 个时间特征")
        
        return df
        
    def create_lag_features(self, 
                           df: pd.DataFrame, 
                           columns: List[str], 
                           lag_periods: List[int] = None) -> pd.DataFrame:
        """
        创建滞后特征
        
        Args:
            df: 输入数据框
            columns: 需要创建滞后特征的列
            lag_periods: 滞后期数列表
            
        Returns:
            添加了滞后特征的数据框
        """
        if lag_periods is None:
            lag_periods = [1, 4, 96, 672]  # 默认：1期、1小时、1天、1周
            
        df = df.copy()
        
        # 确保数据按时间排序
        sort_cols = []
        for col in ['date', 'target_date', 'forecast_target_date']:
            if col in df.columns:
                sort_cols.append(col)
                break
        if 'time_interval' in df.columns:
            sort_cols.append('time_interval')
        
        if sort_cols:
            df = df.sort_values(sort_cols)
        
        # 创建滞后特征
        lag_features_created = 0
        for col in columns:
            if col in df.columns:
                for lag in lag_periods:
                    lag_col_name = f'{col}_lag_{lag}'
                    df[lag_col_name] = df[col].shift(lag)
                    lag_features_created += 1
                    
        logger.info(f"创建了 {lag_features_created} 个滞后特征")
        
        return df
        
    def create_rolling_features(self, 
                               df: pd.DataFrame,
                               columns: List[str],
                               windows: List[int] = None,
                               functions: List[str] = None) -> pd.DataFrame:
        """
        创建滚动统计特征
        
        Args:
            df: 输入数据框
            columns: 需要计算滚动统计的列
            windows: 窗口大小列表
            functions: 统计函数列表
            
        Returns:
            添加了滚动统计特征的数据框
        """
        if windows is None:
            windows = [4, 96, 672]  # 默认：1小时、1天、1周
            
        if functions is None:
            functions = ['mean', 'std', 'min', 'max']
            
        df = df.copy()
        
        # 确保数据按时间排序
        sort_cols = []
        for col in ['date', 'target_date', 'forecast_target_date']:
            if col in df.columns:
                sort_cols.append(col)
                break
        if 'time_interval' in df.columns:
            sort_cols.append('time_interval')
        
        if sort_cols:
            df = df.sort_values(sort_cols)
        
        # 创建滚动特征
        rolling_features_created = 0
        for col in columns:
            if col in df.columns:
                for window in windows:
                    for func in functions:
                        feature_name = f'{col}_roll_{func}_{window}'
                        if func == 'mean':
                            df[feature_name] = df[col].rolling(window=window, min_periods=1).mean()
                        elif func == 'std':
                            df[feature_name] = df[col].rolling(window=window, min_periods=1).std()
                        elif func == 'min':
                            df[feature_name] = df[col].rolling(window=window, min_periods=1).min()
                        elif func == 'max':
                            df[feature_name] = df[col].rolling(window=window, min_periods=1).max()
                        rolling_features_created += 1
                        
        logger.info(f"创建了 {rolling_features_created} 个滚动统计特征")
        
        return df
        
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建交互特征
        
        Args:
            df: 输入数据框
            
        Returns:
            添加了交互特征的数据框
        """
        df = df.copy()
        interaction_features_created = 0
        
        # 供需平衡特征
        if all(col in df.columns for col in ['load_actual', 'total_generation']):
            df['supply_demand_ratio'] = df['total_generation'] / (df['load_actual'] + 1e-6)
            df['supply_demand_diff'] = df['total_generation'] - df['load_actual']
            df['supply_demand_gap_pct'] = (df['supply_demand_diff'] / (df['load_actual'] + 1e-6)) * 100
            interaction_features_created += 3
            
        # 新能源特征
        if all(col in df.columns for col in ['wind_actual', 'solar_actual']):
            df['renewable_total'] = df['wind_actual'].fillna(0) + df['solar_actual'].fillna(0)
            interaction_features_created += 1
            
            if 'total_generation' in df.columns:
                df['renewable_ratio'] = df['renewable_total'] / (df['total_generation'] + 1e-6)
                df['renewable_ratio_pct'] = df['renewable_ratio'] * 100
                interaction_features_created += 2
                
            if 'load_actual' in df.columns:
                df['renewable_load_ratio'] = df['renewable_total'] / (df['load_actual'] + 1e-6)
                interaction_features_created += 1
                
        # 常规电源特征
        conventional_cols = ['coal_output', 'hydro_output', 'nuclear_output']
        existing_conv_cols = [col for col in conventional_cols if col in df.columns]
        if existing_conv_cols:
            df['conventional_total'] = df[existing_conv_cols].fillna(0).sum(axis=1)
            interaction_features_created += 1
            
            if 'total_generation' in df.columns:
                df['conventional_ratio'] = df['conventional_total'] / (df['total_generation'] + 1e-6)
                interaction_features_created += 1
                
        # 联络线影响
        if all(col in df.columns for col in ['load_actual', 'inter_line_power']):
            df['net_load'] = df['load_actual'] - df['inter_line_power'].fillna(0)
            df['inter_line_ratio'] = df['inter_line_power'].fillna(0) / (df['load_actual'] + 1e-6)
            interaction_features_created += 2
            
        # 检修影响
        if 'maintenance_capacity' in df.columns and 'total_generation' in df.columns:
            df['maintenance_ratio'] = df['maintenance_capacity'].fillna(0) / (df['total_generation'] + 1e-6)
            interaction_features_created += 1
            
        # 预测与实际的偏差（如果有）
        forecast_actual_pairs = [
            ('load_forecast', 'load_actual'),
            ('wind_forecast', 'wind_actual'),
            ('solar_forecast', 'solar_actual')
        ]
        
        for forecast_col, actual_col in forecast_actual_pairs:
            if all(col in df.columns for col in [forecast_col, actual_col]):
                df[f'{forecast_col}_error'] = df[actual_col] - df[forecast_col]
                df[f'{forecast_col}_error_pct'] = (df[f'{forecast_col}_error'] / (df[forecast_col] + 1e-6)) * 100
                df[f'{forecast_col}_error_abs'] = np.abs(df[f'{forecast_col}_error'])
                interaction_features_created += 3
                
        logger.info(f"创建了 {interaction_features_created} 个交互特征")
        
        return df
        
    def handle_missing_values(self, 
                            df: pd.DataFrame, 
                            method: str = 'interpolate',
                            columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            df: 输入数据框
            method: 处理方法 ('interpolate', 'forward_fill', 'mean', 'median', 'drop')
            columns: 需要处理的列，如果为None则处理所有数值列
            
        Returns:
            处理后的数据框
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
        missing_before = df[columns].isnull().sum().sum()
        
        if method == 'interpolate':
            # 时间序列插值
            df[columns] = df[columns].interpolate(method='time', limit_direction='both')
            # 对于仍有缺失的，使用线性插值
            df[columns] = df[columns].interpolate(method='linear', limit_direction='both')
        elif method == 'forward_fill':
            df[columns] = df[columns].fillna(method='ffill').fillna(method='bfill')
        elif method == 'mean':
            df[columns] = df[columns].fillna(df[columns].mean())
        elif method == 'median':
            df[columns] = df[columns].fillna(df[columns].median())
        elif method == 'drop':
            df = df.dropna(subset=columns)
            
        missing_after = df[columns].isnull().sum().sum()
        logger.info(f"缺失值处理: {missing_before} -> {missing_after}")
        
        return df
        
    def scale_features(self, 
                      df: pd.DataFrame, 
                      feature_columns: List[str],
                      method: str = 'standard',
                      fit: bool = True) -> pd.DataFrame:
        """
        特征缩放
        
        Args:
            df: 输入数据框
            feature_columns: 需要缩放的特征列
            method: 缩放方法 ('standard', 'minmax', 'robust')
            fit: 是否拟合缩放器（训练时为True，预测时为False）
            
        Returns:
            缩放后的数据框
        """
        df = df.copy()
        
        for col in feature_columns:
            if col in df.columns:
                if fit or col not in self.scalers:
                    # 创建或更新缩放器
                    if method == 'standard':
                        self.scalers[col] = StandardScaler()
                    elif method == 'minmax':
                        self.scalers[col] = MinMaxScaler()
                    elif method == 'robust':
                        self.scalers[col] = RobustScaler()
                    else:
                        raise ValueError(f"不支持的缩放方法: {method}")
                        
                    if fit:
                        df[col] = self.scalers[col].fit_transform(df[[col]])
                else:
                    # 使用已有的缩放器
                    df[col] = self.scalers[col].transform(df[[col]])
                    
        return df
        
    def remove_outliers(self, 
                       df: pd.DataFrame, 
                       columns: List[str],
                       method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        移除异常值
        
        Args:
            df: 输入数据框
            columns: 需要检查异常值的列
            method: 方法 ('iqr', 'zscore')
            threshold: 阈值
            
        Returns:
            移除异常值后的数据框
        """
        df = df.copy()
        outliers_before = len(df)
        
        if method == 'iqr':
            for col in columns:
                if col in df.columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        elif method == 'zscore':
            for col in columns:
                if col in df.columns:
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    df = df[z_scores < threshold]
                    
        outliers_removed = outliers_before - len(df)
        logger.info(f"移除了 {outliers_removed} 个异常值")
        
        return df
        
    def prepare_dataset(self, 
                       df: pd.DataFrame,
                       target_columns: List[str],
                       feature_columns: Optional[List[str]] = None,
                       test_size: float = 0.2,
                       validation_size: float = 0.1) -> Dict[str, Union[pd.DataFrame, List[str]]]:
        """
        准备完整的数据集
        
        Args:
            df: 输入数据框
            target_columns: 目标列
            feature_columns: 特征列（如果为None则自动选择）
            test_size: 测试集比例
            validation_size: 验证集比例
            
        Returns:
            包含训练集、验证集、测试集和特征列表的字典
        """
        # 创建所有特征
        df = self.create_time_features(df)
        df = self.create_lag_features(df, target_columns + ['load_actual', 'wind_actual', 'solar_actual'])
        df = self.create_rolling_features(df, target_columns + ['load_actual'])
        df = self.create_interaction_features(df)
        df = self.handle_missing_values(df)
        
        # 删除目标列中包含NaN的行
        df = df.dropna(subset=target_columns)
        
        # 选择特征
        if feature_columns is None:
            # 自动选择所有数值列作为特征（排除目标列和ID列）
            exclude_cols = target_columns + ['id', 'date', 'target_date', 'forecast_target_date', 'time_point']
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in numeric_cols if col not in exclude_cols]
            
        # 确保所有特征列都存在
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        # 特征缩放
        df = self.scale_features(df, feature_columns)
        
        # 分割数据集（时序数据，不能随机分割）
        n_samples = len(df)
        test_start = int(n_samples * (1 - test_size))
        val_start = int(n_samples * (1 - test_size - validation_size))
        
        train_df = df.iloc[:val_start]
        val_df = df.iloc[val_start:test_start]
        test_df = df.iloc[test_start:]
        
        # 保存特征列信息
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        
        result = {
            'train': train_df,
            'validation': val_df,
            'test': test_df,
            'feature_columns': feature_columns,
            'target_columns': target_columns,
            'train_X': train_df[feature_columns],
            'train_y': train_df[target_columns],
            'val_X': val_df[feature_columns],
            'val_y': val_df[target_columns],
            'test_X': test_df[feature_columns],
            'test_y': test_df[target_columns]
        }
        
        logger.info(f"数据集准备完成:")
        logger.info(f"  训练集: {len(train_df)} 样本")
        logger.info(f"  验证集: {len(val_df)} 样本")
        logger.info(f"  测试集: {len(test_df)} 样本")
        logger.info(f"  特征数: {len(feature_columns)}")
        
        return result