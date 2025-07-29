#!/usr/bin/env python
"""
模型训练脚本
用于训练电价预测模型
"""
import os
import sys
import argparse
import yaml
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import warnings
warnings.filterwarnings('ignore')

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


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化训练器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = get_config(config_path)
        self.db_manager = None
        self.pipeline = None
        self.model = None
        
    def setup(self):
        """设置数据库和数据管道"""
        logger.info("初始化数据库连接...")
        self.db_manager = DatabaseManager()
        self.db_manager.init_engine()
        
        # 创建数据管道
        pipeline_config = self.config.get_data_config()
        self.pipeline = ElectricityPricePipeline(self.db_manager, pipeline_config)
        
    def prepare_data(self, start_date: date, end_date: date):
        """
        准备训练数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            训练数据集
        """
        logger.info(f"准备训练数据: {start_date} 到 {end_date}")
        
        # 获取训练数据
        X, y, feature_names = self.pipeline.create_training_dataset(
            start_date=start_date,
            end_date=end_date
        )
        
        logger.info(f"数据形状 - X: {X.shape}, y: {y.shape}")
        logger.info(f"特征数量: {len(feature_names)}")
        logger.info(f"目标列: {list(y.columns)}")
        
        # 数据质量检查
        self._check_data_quality(X, y)
        
        return X, y, feature_names
        
    def train_model(self, 
                   model_type: str,
                   X_train: pd.DataFrame,
                   y_train: pd.DataFrame,
                   X_val: pd.DataFrame,
                   y_val: pd.DataFrame,
                   optimize_hyperparams: bool = False):
        """
        训练模型
        
        Args:
            model_type: 模型类型
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            optimize_hyperparams: 是否优化超参数
            
        Returns:
            训练好的模型
        """
        logger.info(f"训练模型: {model_type}")
        
        # 获取模型配置
        if model_type in ['xgboost', 'lightgbm']:
            model_config = self.config.get_model_config('traditional').get(model_type, {})
        else:
            model_config = self.config.get_model_config('traditional').get('ensemble', {})
            
        # 创建模型
        if model_type == 'xgboost':
            model = XGBoostPriceModel(**model_config.get('params', {}))
        elif model_type == 'lightgbm':
            model = LightGBMPriceModel(**model_config.get('params', {}))
        elif model_type == 'ensemble':
            ensemble_config = {
                'ensemble_method': model_config.get('method', 'stacking'),
                'meta_model_type': model_config.get('meta_model', 'linear')
            }
            model = EnsemblePriceModel(**ensemble_config)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
            
        # 训练模型
        model.fit(
            X_train, y_train,
            X_val, y_val,
            optimize_hyperparams=optimize_hyperparams
        )
        
        return model
        
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.DataFrame):
        """
        评估模型
        
        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试目标
            
        Returns:
            评估结果
        """
        logger.info("评估模型性能...")
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算指标
        metrics = model.evaluate(X_test, y_test)
        
        # 详细评估
        if isinstance(y_test, pd.DataFrame) and y_test.shape[1] > 1:
            # 多目标
            for i, col in enumerate(y_test.columns):
                logger.info(f"\n{col} 评估结果:")
                logger.info(f"  MAE: {metrics.get(f'{col}_mae', 'N/A'):.4f}")
                logger.info(f"  RMSE: {metrics.get(f'{col}_rmse', 'N/A'):.4f}")
                logger.info(f"  MAPE: {metrics.get(f'{col}_mape', 'N/A'):.2f}%")
                logger.info(f"  R2: {metrics.get(f'{col}_r2', 'N/A'):.4f}")
        else:
            logger.info(f"MAE: {metrics.get('mae', 'N/A'):.4f}")
            logger.info(f"RMSE: {metrics.get('rmse', 'N/A'):.4f}")
            logger.info(f"MAPE: {metrics.get('mape', 'N/A'):.2f}%")
            logger.info(f"R2: {metrics.get('r2', 'N/A'):.4f}")
            
        logger.info(f"方向准确率: {metrics.get('direction_accuracy', 'N/A'):.2%}")
        
        return metrics
        
    def save_model(self, model, model_type: str, metrics: dict):
        """
        保存模型
        
        Args:
            model: 训练好的模型
            model_type: 模型类型
            metrics: 评估指标
        """
        # 创建保存目录
        save_dir = Path(f"models/traditional/{model_type}")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_type}_model_{timestamp}.pkl"
        filepath = save_dir / filename
        
        # 保存模型
        model.save_model(str(filepath), include_data=False)
        logger.info(f"模型已保存至: {filepath}")
        
        # 保存最新版本的链接
        latest_path = save_dir / f"{model_type}_model_latest.pkl"
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(filename)
        
        # 保存评估结果
        metrics_path = save_dir / f"{model_type}_metrics_{timestamp}.yaml"
        with open(metrics_path, 'w') as f:
            yaml.dump(metrics, f)
            
    def _check_data_quality(self, X: pd.DataFrame, y: pd.DataFrame):
        """数据质量检查"""
        # 检查缺失值
        X_missing = X.isnull().sum()
        y_missing = y.isnull().sum()
        
        if X_missing.any():
            logger.warning(f"特征中存在缺失值:\n{X_missing[X_missing > 0]}")
            
        if y_missing.any():
            logger.warning(f"目标中存在缺失值:\n{y_missing[y_missing > 0]}")
            
        # 检查无穷值
        X_inf = np.isinf(X).sum()
        y_inf = np.isinf(y).sum()
        
        if X_inf.any():
            logger.warning(f"特征中存在无穷值:\n{X_inf[X_inf > 0]}")
            
        if y_inf.any():
            logger.warning(f"目标中存在无穷值:\n{y_inf[y_inf > 0]}")
            
    def run_mlflow_experiment(self, 
                            model_type: str,
                            X_train: pd.DataFrame,
                            y_train: pd.DataFrame,
                            X_val: pd.DataFrame,
                            y_val: pd.DataFrame,
                            X_test: pd.DataFrame,
                            y_test: pd.DataFrame):
        """
        使用MLflow跟踪实验
        """
        mlflow.set_experiment(f"electricity_price_{model_type}")
        
        with mlflow.start_run():
            # 记录参数
            model_config = self.config.get_model_config('traditional').get(model_type, {})
            mlflow.log_params(model_config.get('params', {}))
            
            # 训练模型
            model = self.train_model(
                model_type=model_type,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                optimize_hyperparams=False
            )
            
            # 评估模型
            val_metrics = self.evaluate_model(model, X_val, y_val)
            test_metrics = self.evaluate_model(model, X_test, y_test)
            
            # 记录指标
            for key, value in val_metrics.items():
                mlflow.log_metric(f"val_{key}", value)
                
            for key, value in test_metrics.items():
                mlflow.log_metric(f"test_{key}", value)
                
            # 记录模型
            if model_type == 'xgboost':
                mlflow.xgboost.log_model(model.models[list(model.models.keys())[0]], "model")
            else:
                mlflow.sklearn.log_model(model, "model")
                
            # 记录特征重要性
            if hasattr(model, 'get_feature_importance'):
                importance_df = model.get_feature_importance()
                importance_path = "feature_importance.csv"
                importance_df.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path)
                os.remove(importance_path)
                
            return model, test_metrics


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练电价预测模型')
    parser.add_argument('--model', type=str, default='xgboost', 
                       choices=['xgboost', 'lightgbm', 'ensemble'],
                       help='模型类型')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--start-date', type=str, required=True,
                       help='训练数据开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True,
                       help='训练数据结束日期 (YYYY-MM-DD)')
    parser.add_argument('--optimize-hyperparams', action='store_true',
                       help='是否优化超参数')
    parser.add_argument('--use-mlflow', action='store_true',
                       help='是否使用MLflow跟踪实验')
    parser.add_argument('--test-split', type=float, default=0.2,
                       help='测试集比例')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='验证集比例')
    
    args = parser.parse_args()
    
    # 解析日期
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    
    # 创建训练器
    trainer = ModelTrainer(args.config)
    trainer.setup()
    
    try:
        # 准备数据
        X, y, feature_names = trainer.prepare_data(start_date, end_date)
        
        # 分割数据集
        n_samples = len(X)
        test_idx = int(n_samples * (1 - args.test_split))
        val_idx = int(n_samples * (1 - args.test_split - args.val_split))
        
        X_train = X.iloc[:val_idx]
        y_train = y.iloc[:val_idx]
        X_val = X.iloc[val_idx:test_idx]
        y_val = y.iloc[val_idx:test_idx]
        X_test = X.iloc[test_idx:]
        y_test = y.iloc[test_idx:]
        
        logger.info(f"数据集划分 - 训练: {len(X_train)}, 验证: {len(X_val)}, 测试: {len(X_test)}")
        
        if args.use_mlflow:
            # 使用MLflow
            model, metrics = trainer.run_mlflow_experiment(
                model_type=args.model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test
            )
        else:
            # 普通训练
            model = trainer.train_model(
                model_type=args.model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                optimize_hyperparams=args.optimize_hyperparams
            )
            
            # 评估
            metrics = trainer.evaluate_model(model, X_test, y_test)
            
        # 保存模型
        trainer.save_model(model, args.model, metrics)
        
        logger.info("训练完成!")
        
    except Exception as e:
        logger.error(f"训练失败: {e}", exc_info=True)
        raise
    finally:
        if trainer.db_manager:
            trainer.db_manager.close()


if __name__ == "__main__":
    main()