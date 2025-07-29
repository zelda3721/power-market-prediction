#!/usr/bin/env python
"""
电价预测脚本
使用训练好的模型进行预测
"""
import os
import sys
import argparse
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.database import DatabaseManager
from src.preprocessing.pipeline import ElectricityPricePipeline
from src.models.traditional import XGBoostPriceModel, LightGBMPriceModel, EnsemblePriceModel
from config import get_config
from config.logging_config import setup_logging, get_logger

# 设置日志
setup_logging()
logger = get_logger(__name__)


class PricePredictor:
    """电价预测器"""
    
    def __init__(self, model_path: str = None, config_path: str = None):
        """
        初始化预测器
        
        Args:
            model_path: 模型文件路径
            config_path: 配置文件路径
        """
        self.config = get_config(config_path)
        self.db_manager = None
        self.pipeline = None
        self.model = None
        self.model_path = model_path
        
    def setup(self):
        """设置数据库和数据管道"""
        logger.info("初始化数据库连接...")
        self.db_manager = DatabaseManager()
        self.db_manager.init_engine()
        
        # 创建数据管道
        pipeline_config = self.config.get_data_config()
        self.pipeline = ElectricityPricePipeline(self.db_manager, pipeline_config)
        
        # 加载预处理信息
        self.pipeline.load_preprocessing_info()
        
    def load_model(self):
        """加载模型"""
        if not self.model_path:
            raise ValueError("未指定模型路径")
            
        logger.info(f"加载模型: {self.model_path}")
        
        # 加载模型数据
        model_data = joblib.load(self.model_path)
        
        # 确定模型类型
        model_obj = model_data.get('model')
        if hasattr(model_obj, 'model_name'):
            model_name = model_obj.model_name
        else:
            # 从文件名推断
            if 'xgboost' in self.model_path.lower():
                model_name = 'xgboost'
            elif 'lightgbm' in self.model_path.lower():
                model_name = 'lightgbm'
            else:
                model_name = 'ensemble'
                
        # 创建模型实例
        if 'xgboost' in model_name.lower():
            self.model = XGBoostPriceModel()
        elif 'lightgbm' in model_name.lower():
            self.model = LightGBMPriceModel()
        else:
            self.model = EnsemblePriceModel()
            
        # 加载模型
        self.model.load_model(self.model_path)
        logger.info(f"模型加载成功: {self.model}")
        
    def predict_single_day(self, target_date: date, return_uncertainty: bool = False):
        """
        预测单日电价
        
        Args:
            target_date: 预测日期
            return_uncertainty: 是否返回不确定性区间
            
        Returns:
            预测结果DataFrame
        """
        logger.info(f"预测日期: {target_date}")
        
        # 获取预测特征
        pred_features, history_data = self.pipeline.fetch_prediction_data(target_date)
        
        if pred_features.empty:
            logger.warning(f"未找到 {target_date} 的预测数据")
            return pd.DataFrame()
            
        # 处理预测数据
        X_pred = self.pipeline.process_prediction_data(pred_features, history_data)
        
        if X_pred.empty:
            logger.warning("预测特征处理失败")
            return pd.DataFrame()
            
        # 确保特征顺序与训练时一致
        if hasattr(self.model, 'feature_names') and self.model.feature_names:
            missing_features = set(self.model.feature_names) - set(X_pred.columns)
            if missing_features:
                logger.warning(f"缺少特征: {missing_features}")
                # 填充缺失特征
                for feat in missing_features:
                    X_pred[feat] = 0
                    
            # 按照训练时的顺序排列
            X_pred = X_pred[self.model.feature_names]
            
        # 预测
        if return_uncertainty:
            y_pred, y_lower, y_upper = self.model.predict_with_uncertainty(X_pred)
        else:
            y_pred = self.model.predict(X_pred)
            y_lower = y_upper = None
            
        # 构建结果DataFrame
        result = self._build_result_dataframe(
            target_date, pred_features, y_pred, y_lower, y_upper
        )
        
        return result
        
    def predict_multiple_days(self, start_date: date, end_date: date):
        """
        预测多日电价
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            预测结果DataFrame
        """
        results = []
        current_date = start_date
        
        while current_date <= end_date:
            try:
                daily_result = self.predict_single_day(current_date)
                if not daily_result.empty:
                    results.append(daily_result)
            except Exception as e:
                logger.error(f"预测 {current_date} 失败: {e}")
                
            current_date += timedelta(days=1)
            
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()
            
    def _build_result_dataframe(self, 
                               target_date: date,
                               pred_features: pd.DataFrame,
                               y_pred: np.ndarray,
                               y_lower: np.ndarray = None,
                               y_upper: np.ndarray = None) -> pd.DataFrame:
        """构建结果DataFrame"""
        result = pd.DataFrame()
        
        # 基础信息
        result['date'] = target_date
        result['time_point'] = pred_features['time_point'].values[:len(y_pred)]
        result['time_interval'] = pred_features['time_interval'].values[:len(y_pred)]
        
        # 预测结果
        if hasattr(self.model, 'target_names') and len(self.model.target_names) > 1:
            # 多目标
            for i, target_name in enumerate(self.model.target_names):
                result[f'{target_name}_pred'] = y_pred[:, i]
                if y_lower is not None:
                    result[f'{target_name}_lower'] = y_lower[:, i]
                if y_upper is not None:
                    result[f'{target_name}_upper'] = y_upper[:, i]
        else:
            # 单目标
            result['price_pred'] = y_pred
            if y_lower is not None:
                result['price_lower'] = y_lower
            if y_upper is not None:
                result['price_upper'] = y_upper
                
        # 添加输入特征（可选）
        feature_cols = ['load_forecast', 'wind_forecast', 'solar_forecast', 
                       'inter_line_forecast', 'renewable_total']
        for col in feature_cols:
            if col in pred_features.columns:
                result[col] = pred_features[col].values[:len(y_pred)]
                
        return result
        
    def save_predictions(self, predictions: pd.DataFrame, output_path: str):
        """
        保存预测结果
        
        Args:
            predictions: 预测结果
            output_path: 输出路径
        """
        # 确定文件格式
        if output_path.endswith('.csv'):
            predictions.to_csv(output_path, index=False)
        elif output_path.endswith('.xlsx'):
            predictions.to_excel(output_path, index=False)
        elif output_path.endswith('.json'):
            predictions.to_json(output_path, orient='records', date_format='iso')
        else:
            # 默认保存为CSV
            predictions.to_csv(output_path + '.csv', index=False)
            
        logger.info(f"预测结果已保存至: {output_path}")
        
    def generate_prediction_report(self, predictions: pd.DataFrame) -> dict:
        """
        生成预测报告
        
        Args:
            predictions: 预测结果
            
        Returns:
            报告字典
        """
        report = {
            'prediction_summary': {
                'start_date': predictions['date'].min().strftime('%Y-%m-%d'),
                'end_date': predictions['date'].max().strftime('%Y-%m-%d'),
                'days': predictions['date'].nunique(),
                'total_points': len(predictions)
            }
        }
        
        # 价格统计
        price_cols = [col for col in predictions.columns if col.endswith('_pred')]
        for col in price_cols:
            target_name = col.replace('_pred', '')
            report[f'{target_name}_statistics'] = {
                'mean': float(predictions[col].mean()),
                'std': float(predictions[col].std()),
                'min': float(predictions[col].min()),
                'max': float(predictions[col].max()),
                'q25': float(predictions[col].quantile(0.25)),
                'q50': float(predictions[col].quantile(0.50)),
                'q75': float(predictions[col].quantile(0.75))
            }
            
            # 峰谷分析
            daily_stats = predictions.groupby('date')[col].agg(['max', 'min', 'mean'])
            report[f'{target_name}_daily'] = {
                'avg_peak': float(daily_stats['max'].mean()),
                'avg_valley': float(daily_stats['min'].mean()),
                'avg_spread': float((daily_stats['max'] - daily_stats['min']).mean())
            }
            
        return report


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='电价预测')
    parser.add_argument('--model', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--date', type=str, required=True,
                       help='预测日期 (YYYY-MM-DD) 或日期范围 (YYYY-MM-DD:YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件路径')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--uncertainty', action='store_true',
                       help='是否输出不确定性区间')
    parser.add_argument('--report', action='store_true',
                       help='是否生成预测报告')
    
    args = parser.parse_args()
    
    # 解析日期
    if ':' in args.date:
        # 日期范围
        start_str, end_str = args.date.split(':')
        start_date = datetime.strptime(start_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_str, '%Y-%m-%d').date()
        single_day = False
    else:
        # 单日
        start_date = end_date = datetime.strptime(args.date, '%Y-%m-%d').date()
        single_day = True
        
    # 创建预测器
    predictor = PricePredictor(model_path=args.model, config_path=args.config)
    
    try:
        # 设置
        predictor.setup()
        predictor.load_model()
        
        # 预测
        if single_day:
            predictions = predictor.predict_single_day(
                start_date, 
                return_uncertainty=args.uncertainty
            )
        else:
            predictions = predictor.predict_multiple_days(start_date, end_date)
            
        if predictions.empty:
            logger.warning("未生成任何预测结果")
            return
            
        # 显示预测结果
        logger.info(f"\n预测结果预览:\n{predictions.head(10)}")
        
        # 保存结果
        if args.output:
            predictor.save_predictions(predictions, args.output)
        else:
            # 默认文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"predictions_{timestamp}.csv"
            predictor.save_predictions(predictions, output_path)
            
        # 生成报告
        if args.report:
            report = predictor.generate_prediction_report(predictions)
            report_path = args.output.replace('.csv', '_report.json') if args.output else f"prediction_report_{timestamp}.json"
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
                
            logger.info(f"预测报告已保存至: {report_path}")
            
            # 打印报告摘要
            logger.info("\n预测报告摘要:")
            for key, value in report['prediction_summary'].items():
                logger.info(f"  {key}: {value}")
                
    except Exception as e:
        logger.error(f"预测失败: {e}", exc_info=True)
        raise
    finally:
        if predictor.db_manager:
            predictor.db_manager.close()


if __name__ == "__main__":
    main()