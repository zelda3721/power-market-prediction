"""
特征工程模块
提供高级特征创建功能
"""
import logging
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from datetime import datetime
import chinese_calendar as cn_cal

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """特征工程师"""
    
    def __init__(self):
        self.logger = logger
        
    def create_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建节假日特征
        
        Args:
            df: 输入数据框
            
        Returns:
            添加了节假日特征的数据框
        """
        df = df.copy()
        
        # 找到日期列
        date_col = None
        for col in ['date', 'target_date', 'forecast_target_date']:
            if col in df.columns:
                date_col = col
                break
                
        if date_col is None:
            self.logger.warning("未找到日期列，无法创建节假日特征")
            return df
            
        # 确保日期格式
        df[date_col] = pd.to_datetime(df[date_col])
        
        # 中国节假日
        df['is_holiday'] = df[date_col].apply(lambda x: cn_cal.is_holiday(x))
        df['is_workday'] = df[date_col].apply(lambda x: cn_cal.is_workday(x))
        
        # 节假日类型
        df['holiday_name'] = df[date_col].apply(self._get_holiday_name)
        
        # 距离最近节假日的天数
        df['days_to_holiday'] = df[date_col].apply(self._days_to_nearest_holiday)
        df['days_from_holiday'] = df[date_col].apply(self._days_from_nearest_holiday)
        
        # 是否是节假日前后
        df['is_pre_holiday'] = (df['days_to_holiday'] == 1).astype(int)
        df['is_post_holiday'] = (df['days_from_holiday'] == 1).astype(int)
        
        # 长假标识（春节、国庆等）
        major_holidays = ['Spring Festival', 'National Day']
        df['is_major_holiday'] = df['holiday_name'].isin(major_holidays).astype(int)
        
        self.logger.info("创建了节假日特征")
        
        return df
        
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建价格相关特征
        
        Args:
            df: 输入数据框
            
        Returns:
            添加了价格特征的数据框
        """
        df = df.copy()
        
        # 价格差值特征
        if all(col in df.columns for col in ['day_ahead_price', 'real_time_price']):
            df['price_spread'] = df['real_time_price'] - df['day_ahead_price']
            df['price_spread_pct'] = (df['price_spread'] / (df['day_ahead_price'] + 1e-6)) * 100
            df['price_spread_abs'] = np.abs(df['price_spread'])
            
            # 价格比率
            df['rt_da_ratio'] = df['real_time_price'] / (df['day_ahead_price'] + 1e-6)
            
        # 价格变化率
        for price_col in ['day_ahead_price', 'real_time_price']:
            if price_col in df.columns:
                # 环比变化
                df[f'{price_col}_change'] = df[price_col].diff()
                df[f'{price_col}_change_pct'] = df[price_col].pct_change() * 100
                
                # 同比变化（与上一天同时段比较）
                df[f'{price_col}_yoy_change'] = df[price_col].diff(96)
                df[f'{price_col}_yoy_change_pct'] = df[price_col].pct_change(96) * 100
                
        # 价格位置特征（在历史分位数中的位置）
        for price_col in ['day_ahead_price', 'real_time_price']:
            if price_col in df.columns:
                # 过去30天的分位数
                df[f'{price_col}_quantile_30d'] = df[price_col].rolling(window=96*30, min_periods=96).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1]
                )
                
        # 价格峰谷特征
        if 'time_interval' in df.columns:
            # 标记峰谷时段
            peak_hours = [37, 38, 39, 40, 41, 42, 43, 44, 73, 74, 75, 76, 77, 78, 79, 80]  # 9-11点, 18-20点
            valley_hours = list(range(1, 25))  # 0-6点
            df['is_peak_period'] = df['time_interval'].isin(peak_hours).astype(int)
            df['is_valley_period'] = df['time_interval'].isin(valley_hours).astype(int)
            
        self.logger.info("创建了价格特征")
        
        return df
        
    def create_load_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建负荷相关特征
        
        Args:
            df: 输入数据框
            
        Returns:
            添加了负荷特征的数据框
        """
        df = df.copy()
        
        # 负荷预测准确性特征
        if all(col in df.columns for col in ['load_forecast', 'load_actual']):
            df['load_forecast_error'] = df['load_actual'] - df['load_forecast']
            df['load_forecast_error_pct'] = (df['load_forecast_error'] / (df['load_forecast'] + 1e-6)) * 100
            df['load_forecast_error_abs'] = np.abs(df['load_forecast_error'])
            
        # 负荷变化特征
        for load_col in ['load_actual', 'load_forecast']:
            if load_col in df.columns:
                # 小时内变化
                df[f'{load_col}_hourly_change'] = df[load_col].diff(4)
                df[f'{load_col}_hourly_change_pct'] = df[load_col].pct_change(4) * 100
                
                # 日内模式
                df[f'{load_col}_daily_mean'] = df.groupby(df['date'])[load_col].transform('mean')
                df[f'{load_col}_daily_std'] = df.groupby(df['date'])[load_col].transform('std')
                df[f'{load_col}_daily_ratio'] = df[load_col] / (df[f'{load_col}_daily_mean'] + 1e-6)
                
        # 负荷水平分类
        if 'load_actual' in df.columns:
            # 基于历史分位数
            df['load_level'] = pd.qcut(df['load_actual'], q=5, labels=['很低', '低', '中', '高', '很高'])
            df['load_level_code'] = pd.qcut(df['load_actual'], q=5, labels=False)
            
        self.logger.info("创建了负荷特征")
        
        return df
        
    def create_renewable_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建新能源相关特征
        
        Args:
            df: 输入数据框
            
        Returns:
            添加了新能源特征的数据框
        """
        df = df.copy()
        
        # 新能源波动性特征
        for renewable in ['wind', 'solar']:
            actual_col = f'{renewable}_actual'
            forecast_col = f'{renewable}_forecast'
            
            if actual_col in df.columns:
                # 波动率（标准差）
                df[f'{renewable}_volatility_4h'] = df[actual_col].rolling(window=16, min_periods=1).std()
                df[f'{renewable}_volatility_1d'] = df[actual_col].rolling(window=96, min_periods=1).std()
                
                # 变异系数
                mean_val = df[actual_col].rolling(window=96, min_periods=1).mean()
                df[f'{renewable}_cv_1d'] = df[f'{renewable}_volatility_1d'] / (mean_val + 1e-6)
                
                # 爬坡率
                df[f'{renewable}_ramp_rate'] = df[actual_col].diff()
                df[f'{renewable}_ramp_rate_pct'] = df[actual_col].pct_change() * 100
                
            # 预测误差特征
            if all(col in df.columns for col in [actual_col, forecast_col]):
                df[f'{renewable}_forecast_bias'] = df[actual_col] - df[forecast_col]
                df[f'{renewable}_forecast_rmse_1d'] = df[f'{renewable}_forecast_bias'].rolling(
                    window=96, min_periods=1
                ).apply(lambda x: np.sqrt(np.mean(x**2)))
                
        # 新能源互补性特征
        if all(col in df.columns for col in ['wind_actual', 'solar_actual']):
            # 风光互补指数
            df['wind_solar_complement'] = -df['wind_actual'].rolling(window=96).corr(df['solar_actual'])
            
            # 新能源总出力的稳定性
            renewable_total = df['wind_actual'] + df['solar_actual']
            df['renewable_total_cv'] = (
                renewable_total.rolling(window=96).std() / 
                (renewable_total.rolling(window=96).mean() + 1e-6)
            )
            
        # 新能源出力模式
        if 'hour' in df.columns:
            # 光伏的日出日落特征
            if 'solar_actual' in df.columns:
                df['solar_daylight_ratio'] = df.apply(
                    lambda row: row['solar_actual'] / (df[df['hour'] == row['hour']]['solar_actual'].mean() + 1e-6),
                    axis=1
                )
                
        self.logger.info("创建了新能源特征")
        
        return df
        
    def create_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建市场相关特征
        
        Args:
            df: 输入数据框
            
        Returns:
            添加了市场特征的数据框
        """
        df = df.copy()
        
        # 市场紧张度指标
        if all(col in df.columns for col in ['load_actual', 'total_generation']):
            df['market_tightness'] = df['load_actual'] / (df['total_generation'] + 1e-6)
            df['reserve_margin'] = (df['total_generation'] - df['load_actual']) / (df['load_actual'] + 1e-6)
            
        # 机组检修影响
        if 'maintenance_capacity' in df.columns:
            df['maintenance_impact'] = df['maintenance_capacity'].rolling(window=96, min_periods=1).mean()
            
        # 跨省交易特征
        if 'inter_line_power' in df.columns:
            # 联络线利用率变化
            df['inter_line_change'] = df['inter_line_power'].diff()
            df['inter_line_volatility'] = df['inter_line_power'].rolling(window=96, min_periods=1).std()
            
        self.logger.info("创建了市场特征")
        
        return df
        
    def _get_holiday_name(self, date: pd.Timestamp) -> str:
        """获取节假日名称"""
        try:
            holiday_name, _ = cn_cal.get_holiday_detail(date)
            return holiday_name if holiday_name else 'None'
        except:
            return 'None'
            
    def _days_to_nearest_holiday(self, date: pd.Timestamp) -> int:
        """计算到最近节假日的天数"""
        for i in range(1, 15):
            future_date = date + pd.Timedelta(days=i)
            if cn_cal.is_holiday(future_date):
                return i
        return 15
        
    def _days_from_nearest_holiday(self, date: pd.Timestamp) -> int:
        """计算从最近节假日过去的天数"""
        for i in range(1, 15):
            past_date = date - pd.Timedelta(days=i)
            if cn_cal.is_holiday(past_date):
                return i
        return 15
        
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建所有特征
        
        Args:
            df: 输入数据框
            
        Returns:
            包含所有特征的数据框
        """
        df = self.create_holiday_features(df)
        df = self.create_price_features(df)
        df = self.create_load_features(df)
        df = self.create_renewable_features(df)
        df = self.create_market_features(df)
        
        self.logger.info(f"特征工程完成，总特征数: {len(df.columns)}")
        
        return df