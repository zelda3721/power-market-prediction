"""
数据管道模块
整合数据获取、预处理和特征工程的完整流程
"""
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import joblib
import os

from ..data.database import DatabaseManager
from ..data.dao import PowerMarketDAO
from .preprocessor import DataPreprocessor
from .feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class ElectricityPricePipeline:
    """电价预测数据管道"""
    
    def __init__(self, db_manager: DatabaseManager, config: Optional[Dict] = None):
        """
        初始化数据管道
        
        Args:
            db_manager: 数据库管理器
            config: 配置字典
        """
        self.dao = PowerMarketDAO(db_manager)
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.config = config or {}
        self.logger = logger
        
        # 从配置中读取参数
        self.lag_periods = self.config.get('lag_periods', [1, 4, 96, 672])
        self.rolling_windows = self.config.get('rolling_windows', [4, 96, 672])
        self.use_holiday_features = self.config.get('use_holiday_features', True)
        
    def fetch_training_data(self, 
                           start_date: Union[str, date], 
                           end_date: Union[str, date]) -> pd.DataFrame:
        """
        获取训练数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            原始训练数据
        """
        self.logger.info(f"获取训练数据: {start_date} 到 {end_date}")
        
        # 获取综合特征数据
        df = self.dao.get_comprehensive_features(start_date, end_date)
        
        if df.empty:
            raise ValueError(f"未找到 {start_date} 到 {end_date} 的数据")
            
        self.logger.info(f"获取到 {len(df)} 条原始数据")
        return df
        
    def fetch_prediction_data(self, target_date: Union[str, date]) -> pd.DataFrame:
        """
        获取预测所需数据
        
        Args:
            target_date: 预测目标日期
            
        Returns:
            预测特征数据
        """
        self.logger.info(f"获取预测数据: {target_date}")
        
        # 获取预测特征
        pred_features = self.dao.get_prediction_features(target_date)
        
        # 获取历史数据用于计算滞后特征
        history_days = max(self.lag_periods) // 96 + 7  # 确保有足够的历史数据
        history_start = pd.to_datetime(target_date) - timedelta(days=history_days)
        history_end = pd.to_datetime(target_date) - timedelta(days=1)
        
        history_data = self.dao.get_comprehensive_features(
            history_start.date(), 
            history_end.date()
        )
        
        return pred_features, history_data
        
    def process_training_data(self, 
                             df: pd.DataFrame,
                             test_size: float = 0.2,
                             validation_size: float = 0.1) -> Dict[str, pd.DataFrame]:
        """
        处理训练数据
        
        Args:
            df: 原始数据
            test_size: 测试集比例
            validation_size: 验证集比例
            
        Returns:
            处理后的数据集字典
        """
        self.logger.info("开始处理训练数据...")
        
        # 基础预处理
        df = self._basic_preprocessing(df)
        
        # 特征工程
        df = self._feature_engineering(df)
        
        # 准备数据集
        dataset = self.preprocessor.prepare_dataset(
            df,
            target_columns=['day_ahead_price', 'real_time_price'],
            test_size=test_size,
            validation_size=validation_size
        )
        
        # 保存预处理器和特征列信息
        self._save_preprocessing_info()
        
        return dataset
        
    def process_prediction_data(self, 
                               pred_features: pd.DataFrame,
                               history_data: pd.DataFrame) -> pd.DataFrame:
        """
        处理预测数据
        
        Args:
            pred_features: 预测特征数据
            history_data: 历史数据
            
        Returns:
            处理后的预测特征
        """
        self.logger.info("开始处理预测数据...")
        
        # 合并历史数据和预测数据
        all_data = pd.concat([history_data, pred_features], ignore_index=True)
        
        # 基础预处理
        all_data = self._basic_preprocessing(all_data)
        
        # 特征工程
        all_data = self._feature_engineering(all_data, is_training=False)
        
        # 提取预测日期的数据
        target_date = pred_features['target_date'].iloc[0] if 'target_date' in pred_features.columns else pred_features['date'].iloc[0]
        prediction_data = all_data[all_data['date'] == pd.to_datetime(target_date)]
        
        # 确保包含所有训练时的特征
        if hasattr(self.preprocessor, 'feature_columns'):
            missing_cols = set(self.preprocessor.feature_columns) - set(prediction_data.columns)
            for col in missing_cols:
                prediction_data[col] = 0
                self.logger.warning(f"预测数据缺少特征 {col}，使用默认值0")
                
            # 只保留训练时使用的特征
            prediction_data = prediction_data[self.preprocessor.feature_columns]
            
        return prediction_data
        
    def _basic_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """基础预处理"""
        # 确保日期格式
        for date_col in ['date', 'target_date', 'forecast_target_date']:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col])
                
        # 按时间排序
        if 'date' in df.columns and 'time_interval' in df.columns:
            df = df.sort_values(['date', 'time_interval'])
            
        # 处理明显的异常值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # 将负值替换为0（对于出力等不应为负的变量）
            if col in ['load_actual', 'wind_actual', 'solar_actual', 'coal_output', 
                      'hydro_output', 'nuclear_output', 'total_generation']:
                df[col] = df[col].clip(lower=0)
                
        return df
        
    def _feature_engineering(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """特征工程"""
        # 创建时间特征
        df = self.preprocessor.create_time_features(df)
        
        # 创建节假日特征
        if self.use_holiday_features:
            df = self.feature_engineer.create_holiday_features(df)
            
        # 创建滞后特征
        lag_columns = ['day_ahead_price', 'real_time_price', 'load_actual', 
                      'wind_actual', 'solar_actual', 'load_forecast']
        existing_lag_cols = [col for col in lag_columns if col in df.columns]
        df = self.preprocessor.create_lag_features(df, existing_lag_cols, self.lag_periods)
        
        # 创建滚动特征
        rolling_columns = ['day_ahead_price', 'real_time_price', 'load_actual']
        existing_rolling_cols = [col for col in rolling_columns if col in df.columns]
        df = self.preprocessor.create_rolling_features(df, existing_rolling_cols, self.rolling_windows)
        
        # 创建交互特征
        df = self.preprocessor.create_interaction_features(df)
        
        # 创建高级特征
        df = self.feature_engineer.create_price_features(df)
        df = self.feature_engineer.create_load_features(df)
        df = self.feature_engineer.create_renewable_features(df)
        
        # 处理缺失值
        df = self.preprocessor.handle_missing_values(df, method='interpolate')
        
        return df
        
    def _save_preprocessing_info(self):
        """保存预处理信息"""
        preprocessing_info = {
            'feature_columns': self.preprocessor.feature_columns,
            'target_columns': self.preprocessor.target_columns,
            'scalers': self.preprocessor.scalers,
            'config': self.config
        }
        
        # 创建保存目录
        save_dir = 'models/preprocessing'
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存信息
        save_path = os.path.join(save_dir, 'preprocessing_info.pkl')
        joblib.dump(preprocessing_info, save_path)
        self.logger.info(f"预处理信息已保存至 {save_path}")
        
    def load_preprocessing_info(self):
        """加载预处理信息"""
        load_path = 'models/preprocessing/preprocessing_info.pkl'
        if os.path.exists(load_path):
            preprocessing_info = joblib.load(load_path)
            self.preprocessor.feature_columns = preprocessing_info['feature_columns']
            self.preprocessor.target_columns = preprocessing_info['target_columns']
            self.preprocessor.scalers = preprocessing_info['scalers']
            self.logger.info("预处理信息加载成功")
        else:
            self.logger.warning(f"未找到预处理信息文件: {load_path}")
            
    def create_feature_importance_report(self, 
                                       feature_importance: Dict[str, float],
                                       top_n: int = 30) -> pd.DataFrame:
        """
        创建特征重要性报告
        
        Args:
            feature_importance: 特征重要性字典
            top_n: 显示前N个重要特征
            
        Returns:
            特征重要性报告
        """
        # 转换为DataFrame
        importance_df = pd.DataFrame(
            list(feature_importance.items()),
            columns=['feature', 'importance']
        )
        
        # 排序
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # 添加累计重要性
        importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
        importance_df['cumulative_importance_pct'] = importance_df['cumulative_importance'] * 100
        
        # 特征分类
        importance_df['feature_type'] = importance_df['feature'].apply(self._classify_feature)
        
        # 只保留前N个
        if top_n:
            importance_df = importance_df.head(top_n)
            
        return importance_df
        
    def _classify_feature(self, feature_name: str) -> str:
        """分类特征类型"""
        if 'lag_' in feature_name:
            return '滞后特征'
        elif 'roll_' in feature_name:
            return '滚动特征'
        elif any(x in feature_name for x in ['hour', 'day', 'month', 'week', 'sin', 'cos']):
            return '时间特征'
        elif any(x in feature_name for x in ['ratio', 'diff', 'pct']):
            return '交互特征'
        elif any(x in feature_name for x in ['wind', 'solar', 'renewable']):
            return '新能源特征'
        elif any(x in feature_name for x in ['load', 'demand']):
            return '负荷特征'
        elif any(x in feature_name for x in ['price', 'da_', 'rt_']):
            return '价格特征'
        else:
            return '其他特征'