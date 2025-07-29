#!/usr/bin/env python
"""
使用示例脚本
展示如何使用数据访问层获取数据并进行基础分析
"""
import os
import sys
from datetime import date, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

from src.data.database import DatabaseManager
from src.data.dao import PowerMarketDAO
from src.preprocessing.preprocessor import DataPreprocessor


def main():
    """主函数"""
    # 1. 初始化数据库连接
    print("1. 初始化数据库连接...")
    db_manager = DatabaseManager()
    db_manager.init_engine()
    
    # 2. 创建DAO实例
    dao = PowerMarketDAO(db_manager)
    
    # 3. 获取历史电价数据
    print("\n2. 获取历史电价数据...")
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=30)
    
    price_data = dao.get_price_data(start_date, end_date)
    print(f"获取到 {len(price_data)} 条电价数据")
    print(f"日前电价范围: {price_data['day_ahead_price'].min():.2f} - {price_data['day_ahead_price'].max():.2f}")
    print(f"实时电价范围: {price_data['real_time_price'].min():.2f} - {price_data['real_time_price'].max():.2f}")
    
    # 4. 获取某一天的综合特征
    print("\n3. 获取综合特征数据...")
    target_date = end_date
    comprehensive_data = dao.get_comprehensive_features(target_date, target_date)
    
    if not comprehensive_data.empty:
        print(f"特征列数: {len(comprehensive_data.columns)}")
        print(f"特征列表: {list(comprehensive_data.columns)[:10]}...")  # 显示前10个特征
        
        # 显示数据统计
        print("\n数据统计:")
        stats_cols = ['day_ahead_price', 'real_time_price', 'load_actual', 'wind_actual', 'solar_actual']
        for col in stats_cols:
            if col in comprehensive_data.columns:
                print(f"{col}: 均值={comprehensive_data[col].mean():.2f}, "
                      f"标准差={comprehensive_data[col].std():.2f}")
    
    # 5. 数据质量检查
    print("\n4. 数据质量检查...")
    quality_report = dao.check_data_quality(
        'load_forecast_data',
        'load_forecast_mw',
        start_date,
        end_date
    )
    
    if not quality_report.empty:
        print(f"发现 {len(quality_report)} 天数据存在问题")
        print(quality_report[['date', 'time_points_count', 'null_values', 'negative_values']].head())
    
    # 6. 获取预测特征
    print("\n5. 获取预测特征...")
    prediction_date = end_date + timedelta(days=1)
    pred_features = dao.get_prediction_features(prediction_date)
    
    if not pred_features.empty:
        print(f"预测日期: {prediction_date}")
        print(f"时间点数: {len(pred_features)}")
        print(f"负荷预测范围: {pred_features['load_forecast'].min():.2f} - {pred_features['load_forecast'].max():.2f}")
    
    # 7. 获取历史实际数据（用于构建滞后特征）
    print("\n6. 获取历史实际数据...")
    history_data = dao.get_historical_actuals(start_date, end_date)
    if not history_data.empty:
        print(f"历史数据行数: {len(history_data)}")
        print(f"包含的列: {list(history_data.columns)[:10]}...")
    
    # 8. 简单的数据可视化
    if not price_data.empty:
        print("\n6. 生成数据可视化...")
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 电价时序图
        sample_date = price_data['target_date'].iloc[-96:]
        axes[0, 0].plot(price_data['time_interval'].iloc[-96:], 
                       price_data['day_ahead_price'].iloc[-96:], 
                       label='日前电价', marker='o', markersize=3)
        axes[0, 0].plot(price_data['time_interval'].iloc[-96:], 
                       price_data['real_time_price'].iloc[-96:], 
                       label='实时电价', marker='s', markersize=3)
        axes[0, 0].set_title(f'{sample_date.iloc[0]} 电价曲线')
        axes[0, 0].set_xlabel('时间点')
        axes[0, 0].set_ylabel('电价 (元/MWh)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 电价分布图
        axes[0, 1].hist(price_data['day_ahead_price'], bins=50, alpha=0.5, label='日前电价')
        axes[0, 1].hist(price_data['real_time_price'], bins=50, alpha=0.5, label='实时电价')
        axes[0, 1].set_title('电价分布')
        axes[0, 1].set_xlabel('电价 (元/MWh)')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].legend()
        
        # 日前vs实时电价散点图
        axes[1, 0].scatter(price_data['day_ahead_price'], 
                          price_data['real_time_price'], 
                          alpha=0.5, s=10)
        axes[1, 0].plot([price_data['day_ahead_price'].min(), price_data['day_ahead_price'].max()],
                       [price_data['day_ahead_price'].min(), price_data['day_ahead_price'].max()],
                       'r--', label='y=x')
        axes[1, 0].set_title('日前电价 vs 实时电价')
        axes[1, 0].set_xlabel('日前电价 (元/MWh)')
        axes[1, 0].set_ylabel('实时电价 (元/MWh)')
        axes[1, 0].legend()
        
        # 电价差值分布
        price_diff = price_data['real_time_price'] - price_data['day_ahead_price']
        axes[1, 1].hist(price_diff, bins=50, edgecolor='black')
        axes[1, 1].axvline(x=0, color='r', linestyle='--', label='零差值')
        axes[1, 1].set_title('实时电价与日前电价差值分布')
        axes[1, 1].set_xlabel('价差 (元/MWh)')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('price_analysis.png', dpi=300, bbox_inches='tight')
        print("图表已保存为 price_analysis.png")
    
    # 8. 数据预处理示例
    print("\n7. 数据预处理示例...")
    preprocessor = DataPreprocessor()
    
    if not comprehensive_data.empty:
        # 创建时间特征
        processed_data = preprocessor.create_time_features(comprehensive_data)
        print(f"添加时间特征后的列数: {len(processed_data.columns)}")
        
        # 创建滞后特征
        lag_columns = ['day_ahead_price', 'real_time_price']
        processed_data = preprocessor.create_lag_features(processed_data, lag_columns, [1, 96])
        print(f"添加滞后特征后的列数: {len(processed_data.columns)}")
    
    # 关闭数据库连接
    db_manager.close()
    print("\n完成!")


if __name__ == "__main__":
    main()