"""
XGBoost电价预测模型
"""
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna
from optuna.integration import XGBoostPruningCallback

from ..base import BaseModel

logger = logging.getLogger(__name__)


class XGBoostPriceModel(BaseModel):
    """XGBoost电价预测模型"""
    
    def __init__(self, 
                 model_name: str = "XGBoost_Price_Model",
                 model_version: str = "1.0",
                 random_state: int = 42,
                 **kwargs):
        """
        初始化XGBoost模型
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            random_state: 随机种子
            **kwargs: XGBoost参数
        """
        super().__init__(model_name, model_version, **kwargs)
        self.random_state = random_state
        self.models = {}  # 存储多个目标的模型
        self.best_params = {}  # 存储最优参数
        
        # 默认参数
        self.default_params = {
            'n_estimators': 1000,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_jobs': -1,
            'random_state': self.random_state,
            'tree_method': 'hist',  # 使用直方图算法加速
            'enable_categorical': True
        }
        
        # 更新参数
        self.model_params.update(self.default_params)
        self.model_params.update(kwargs)
        
    def build_model(self, params: Optional[Dict] = None) -> xgb.XGBRegressor:
        """
        构建模型
        
        Args:
            params: 模型参数
            
        Returns:
            XGBoost模型
        """
        if params is None:
            params = self.model_params.copy()
            
        return xgb.XGBRegressor(**params)
        
    def fit(self, 
            X_train: pd.DataFrame, 
            y_train: pd.DataFrame,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.DataFrame] = None,
            optimize_hyperparams: bool = False,
            n_trials: int = 100,
            **kwargs) -> 'XGBoostPriceModel':
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            optimize_hyperparams: 是否优化超参数
            n_trials: Optuna试验次数
            **kwargs: 其他参数
            
        Returns:
            self
        """
        self.logger.info(f"开始训练XGBoost模型...")
        
        # 保存特征和目标信息
        self.feature_names = list(X_train.columns)
        if isinstance(y_train, pd.DataFrame):
            self.target_names = list(y_train.columns)
        else:
            self.target_names = ['target']
            
        # 转换为numpy数组
        X_train_np = X_train.values
        y_train_np = y_train.values if isinstance(y_train, pd.DataFrame) else y_train.values.reshape(-1, 1)
        
        if X_val is not None and y_val is not None:
            X_val_np = X_val.values
            y_val_np = y_val.values if isinstance(y_val, pd.DataFrame) else y_val.values.reshape(-1, 1)
            eval_set = [(X_val_np, y_val_np)]
        else:
            # 如果没有提供验证集，使用一部分训练集
            split_idx = int(len(X_train_np) * 0.8)
            X_val_np = X_train_np[split_idx:]
            y_val_np = y_train_np[split_idx:]
            eval_set = [(X_val_np, y_val_np)]
            
        # 为每个目标训练一个模型
        for i, target_name in enumerate(self.target_names):
            self.logger.info(f"训练目标: {target_name}")
            
            y_train_target = y_train_np[:, i] if y_train_np.ndim > 1 else y_train_np.ravel()
            y_val_target = y_val_np[:, i] if y_val_np.ndim > 1 else y_val_np.ravel()
            
            if optimize_hyperparams:
                # 超参数优化
                self.logger.info(f"开始超参数优化...")
                best_params = self._optimize_hyperparams(
                    X_train_np, y_train_target,
                    X_val_np, y_val_target,
                    n_trials=n_trials
                )
                self.best_params[target_name] = best_params
                model_params = best_params
            else:
                model_params = self.model_params.copy()
                
            # 训练模型
            model = self.build_model(model_params)
            
            # 设置早停
            model.fit(
                X_train_np, y_train_target,
                eval_set=[(X_val_np, y_val_target)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            self.models[target_name] = model
            
            # 记录最佳迭代次数
            self.logger.info(f"{target_name} - 最佳迭代次数: {model.best_iteration}")
            
        self.is_fitted = True
        
        # 计算验证集性能
        val_metrics = self.evaluate(X_val, y_val)
        self.logger.info(f"验证集性能: {val_metrics}")
        
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测
        
        Args:
            X: 特征数据
            
        Returns:
            预测结果
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
            
        X_np = X.values
        predictions = []
        
        for target_name in self.target_names:
            model = self.models[target_name]
            pred = model.predict(X_np)
            predictions.append(pred)
            
        # 合并预测结果
        if len(predictions) == 1:
            return predictions[0]
        else:
            return np.column_stack(predictions)
            
    def predict_with_uncertainty(self, 
                               X: pd.DataFrame,
                               confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        带不确定性的预测（使用多个树的预测）
        
        Args:
            X: 特征数据
            confidence_level: 置信水平
            
        Returns:
            (预测值, 下界, 上界)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
            
        X_np = X.values
        all_predictions = []
        all_lower = []
        all_upper = []
        
        for target_name in self.target_names:
            model = self.models[target_name]
            
            # 获取所有树的预测
            booster = model.get_booster()
            # 设置输出所有树的预测
            iteration_range = (0, booster.best_iteration + 1)
            
            # 获取多个预测（通过不同的迭代）
            predictions_list = []
            step = max(1, booster.best_iteration // 100)  # 采样100个点
            
            for i in range(0, booster.best_iteration + 1, step):
                pred = model.predict(X_np, iteration_range=(0, i+1))
                predictions_list.append(pred)
                
            predictions_array = np.array(predictions_list)
            
            # 计算统计量
            mean_pred = np.mean(predictions_array, axis=0)
            std_pred = np.std(predictions_array, axis=0)
            
            # 计算置信区间
            z_score = 1.96 if confidence_level == 0.95 else 2.58  # 95% or 99%
            lower_bound = mean_pred - z_score * std_pred
            upper_bound = mean_pred + z_score * std_pred
            
            all_predictions.append(mean_pred)
            all_lower.append(lower_bound)
            all_upper.append(upper_bound)
            
        # 合并结果
        if len(all_predictions) == 1:
            return all_predictions[0], all_lower[0], all_upper[0]
        else:
            return (np.column_stack(all_predictions),
                   np.column_stack(all_lower),
                   np.column_stack(all_upper))
                   
    def _optimize_hyperparams(self,
                            X_train: np.ndarray,
                            y_train: np.ndarray,
                            X_val: np.ndarray,
                            y_val: np.ndarray,
                            n_trials: int = 100) -> Dict[str, Any]:
        """
        使用Optuna优化超参数
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            n_trials: 试验次数
            
        Returns:
            最优参数
        """
        def objective(trial):
            params = {
                'n_estimators': 1000,  # 固定，使用早停
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0),
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'n_jobs': -1,
                'random_state': self.random_state,
                'tree_method': 'hist',
                'enable_categorical': True
            }
            
            # 创建模型
            model = xgb.XGBRegressor(**params)
            
            # 添加剪枝回调
            pruning_callback = XGBoostPruningCallback(trial, 'validation_0-rmse')
            
            # 训练
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False,
                callbacks=[pruning_callback]
            )
            
            # 预测
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            
            return rmse
            
        # 创建study
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
        )
        
        # 优化
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
        
        # 获取最优参数
        best_params = self.default_params.copy()
        best_params.update(study.best_params)
        
        self.logger.info(f"最优参数: {study.best_params}")
        self.logger.info(f"最优RMSE: {study.best_value:.4f}")
        
        return best_params
        
    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性
        
        Returns:
            特征重要性DataFrame
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
            
        # 汇总所有目标的特征重要性
        importance_dict = {}
        
        for target_name, model in self.models.items():
            importance = model.feature_importances_
            
            for i, feature in enumerate(self.feature_names):
                key = f"{feature}_{target_name}"
                importance_dict[key] = importance[i]
                
        # 转换为DataFrame
        importance_df = pd.DataFrame(
            list(importance_dict.items()),
            columns=['feature', 'importance']
        )
        
        # 计算每个特征的平均重要性
        importance_df['feature_base'] = importance_df['feature'].str.rsplit('_', n=1).str[0]
        avg_importance = importance_df.groupby('feature_base')['importance'].mean().reset_index()
        avg_importance.columns = ['feature', 'avg_importance']
        
        # 排序
        avg_importance = avg_importance.sort_values('avg_importance', ascending=False)
        
        return avg_importance
        
    def cross_validate(self, 
                      X: pd.DataFrame, 
                      y: pd.DataFrame,
                      cv_splits: int = 5,
                      gap: int = 96) -> Dict[str, List[float]]:
        """
        时序交叉验证
        
        Args:
            X: 特征数据
            y: 目标数据
            cv_splits: 交叉验证折数
            gap: 训练集和验证集之间的间隔
            
        Returns:
            交叉验证结果
        """
        tscv = TimeSeriesSplit(n_splits=cv_splits, gap=gap)
        cv_results = {
            'mae': [],
            'rmse': [],
            'mape': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            self.logger.info(f"交叉验证折 {fold + 1}/{cv_splits}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 训练模型
            model = XGBoostPriceModel(
                model_name=f"{self.model_name}_cv{fold}",
                **self.model_params
            )
            model.fit(X_train, y_train, X_val, y_val)
            
            # 评估
            metrics = model.evaluate(X_val, y_val)
            cv_results['mae'].append(metrics.get('mae', metrics.get('day_ahead_price_mae', 0)))
            cv_results['rmse'].append(metrics.get('rmse', metrics.get('day_ahead_price_rmse', 0)))
            cv_results['mape'].append(metrics.get('mape', metrics.get('day_ahead_price_mape', 0)))
            
        # 计算平均值和标准差
        for metric in cv_results:
            values = cv_results[metric]
            self.logger.info(f"{metric}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")
            
        return cv_results