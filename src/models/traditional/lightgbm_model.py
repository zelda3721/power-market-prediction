"""
LightGBM电价预测模型
"""
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna
from optuna.integration import LightGBMPruningCallback
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from ..base import BaseModel

logger = logging.getLogger(__name__)


class LightGBMPriceModel(BaseModel):
    """LightGBM电价预测模型"""
    
    def __init__(self, 
                 model_name: str = "LightGBM_Price_Model",
                 model_version: str = "1.0",
                 random_state: int = 42,
                 **kwargs):
        """
        初始化LightGBM模型
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            random_state: 随机种子
            **kwargs: LightGBM参数
        """
        super().__init__(model_name, model_version, **kwargs)
        self.random_state = random_state
        self.models = {}  # 存储多个目标的模型
        self.best_params = {}  # 存储最优参数
        self.categorical_features = []  # 类别特征列表
        
        # 默认参数
        self.default_params = {
            'n_estimators': 1000,
            'num_leaves': 31,
            'max_depth': -1,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'min_split_gain': 0.0,
            'lambda_l1': 0,
            'lambda_l2': 1,
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'n_jobs': -1,
            'random_state': self.random_state,
            'verbosity': -1,
            'force_col_wise': True,  # 针对大量特征优化
            'min_data_in_bin': 3,
            'max_bin': 255
        }
        
        # 更新参数
        self.model_params.update(self.default_params)
        self.model_params.update(kwargs)
        
    def identify_categorical_features(self, X: pd.DataFrame) -> List[str]:
        """
        识别类别特征
        
        Args:
            X: 特征数据
            
        Returns:
            类别特征列表
        """
        categorical_features = []
        
        for col in X.columns:
            # 检查是否是类别相关的特征
            if any(keyword in col.lower() for keyword in ['hour', 'dayofweek', 'month', 'quarter', 'is_']):
                # 但不包括已经进行sin/cos编码的特征
                if not any(suffix in col.lower() for suffix in ['_sin', '_cos']):
                    categorical_features.append(col)
                    
        self.logger.info(f"识别到 {len(categorical_features)} 个类别特征: {categorical_features[:5]}...")
        return categorical_features
        
    def build_model(self, params: Optional[Dict] = None) -> lgb.LGBMRegressor:
        """
        构建模型
        
        Args:
            params: 模型参数
            
        Returns:
            LightGBM模型
        """
        if params is None:
            params = self.model_params.copy()
            
        return lgb.LGBMRegressor(**params)
        
    def fit(self, 
            X_train: pd.DataFrame, 
            y_train: pd.DataFrame,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.DataFrame] = None,
            optimize_hyperparams: bool = False,
            n_trials: int = 100,
            use_categorical: bool = True,
            **kwargs) -> 'LightGBMPriceModel':
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            optimize_hyperparams: 是否优化超参数
            n_trials: Optuna试验次数
            use_categorical: 是否使用类别特征
            **kwargs: 其他参数
            
        Returns:
            self
        """
        self.logger.info(f"开始训练LightGBM模型...")
        
        # 保存特征和目标信息
        self.feature_names = list(X_train.columns)
        if isinstance(y_train, pd.DataFrame):
            self.target_names = list(y_train.columns)
        else:
            self.target_names = ['target']
            
        # 识别类别特征
        if use_categorical:
            self.categorical_features = self.identify_categorical_features(X_train)
            categorical_indices = [X_train.columns.get_loc(c) for c in self.categorical_features if c in X_train.columns]
        else:
            categorical_indices = []
            
        # 准备数据
        X_train_values = X_train.values
        y_train_np = y_train.values if isinstance(y_train, pd.DataFrame) else y_train.values.reshape(-1, 1)
        
        if X_val is not None and y_val is not None:
            X_val_values = X_val.values
            y_val_np = y_val.values if isinstance(y_val, pd.DataFrame) else y_val.values.reshape(-1, 1)
        else:
            # 如果没有提供验证集，使用一部分训练集
            split_idx = int(len(X_train_values) * 0.8)
            X_val_values = X_train_values[split_idx:]
            y_val_np = y_train_np[split_idx:]
            X_train_values = X_train_values[:split_idx]
            y_train_np = y_train_np[:split_idx]
            
        # 为每个目标训练一个模型
        for i, target_name in enumerate(self.target_names):
            self.logger.info(f"训练目标: {target_name}")
            
            y_train_target = y_train_np[:, i] if y_train_np.ndim > 1 else y_train_np.ravel()
            y_val_target = y_val_np[:, i] if y_val_np.ndim > 1 else y_val_np.ravel()
            
            if optimize_hyperparams:
                # 超参数优化
                self.logger.info(f"开始超参数优化...")
                best_params = self._optimize_hyperparams(
                    X_train_values, y_train_target,
                    X_val_values, y_val_target,
                    categorical_indices,
                    n_trials=n_trials
                )
                self.best_params[target_name] = best_params
                model_params = best_params
            else:
                model_params = self.model_params.copy()
                
            # 训练模型
            model = self.build_model(model_params)
            
            # 设置类别特征和早停
            fit_params = {
                'eval_set': [(X_val_values, y_val_target)],
                'eval_metric': 'rmse',
                'callbacks': [
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            }
            
            if categorical_indices:
                fit_params['categorical_feature'] = categorical_indices
                
            model.fit(
                X_train_values, y_train_target,
                **fit_params
            )
            
            self.models[target_name] = model
            
            # 记录最佳迭代次数
            if hasattr(model, 'best_iteration_'):
                self.logger.info(f"{target_name} - 最佳迭代次数: {model.best_iteration_}")
                
        self.is_fitted = True
        
        # 计算验证集性能
        if X_val is not None and y_val is not None:
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
            
        X_values = X.values
        predictions = []
        
        for target_name in self.target_names:
            model = self.models[target_name]
            pred = model.predict(X_values)
            predictions.append(pred)
            
        # 合并预测结果
        if len(predictions) == 1:
            return predictions[0]
        else:
            return np.column_stack(predictions)
            
    def predict_with_uncertainty(self, 
                               X: pd.DataFrame,
                               confidence_level: float = 0.95,
                               n_iterations: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        带不确定性的预测（使用dropout）
        
        Args:
            X: 特征数据
            confidence_level: 置信水平
            n_iterations: 预测迭代次数
            
        Returns:
            (预测值, 下界, 上界)
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
            
        X_values = X.values
        all_predictions = []
        all_lower = []
        all_upper = []
        
        for target_name in self.target_names:
            model = self.models[target_name]
            
            # 多次预测（通过feature_fraction模拟dropout）
            predictions_list = []
            
            for _ in range(n_iterations):
                # 随机选择特征子集
                n_features = X_values.shape[1]
                n_selected = int(n_features * 0.8)  # 使用80%的特征
                selected_features = np.random.choice(n_features, n_selected, replace=False)
                
                # 创建特征掩码
                X_masked = X_values.copy()
                mask = np.ones(n_features, dtype=bool)
                mask[selected_features] = False
                X_masked[:, mask] = 0
                
                pred = model.predict(X_masked)
                predictions_list.append(pred)
                
            predictions_array = np.array(predictions_list)
            
            # 计算统计量
            mean_pred = np.mean(predictions_array, axis=0)
            std_pred = np.std(predictions_array, axis=0)
            
            # 计算置信区间
            z_score = 1.96 if confidence_level == 0.95 else 2.58
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
                            categorical_indices: List[int],
                            n_trials: int = 100) -> Dict[str, Any]:
        """
        使用Optuna优化超参数
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            categorical_indices: 类别特征索引
            n_trials: 试验次数
            
        Returns:
            最优参数
        """
        def objective(trial):
            params = {
                'n_estimators': 1000,
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'lambda_l1': trial.suggest_float('lambda_l1', 0, 10),
                'lambda_l2': trial.suggest_float('lambda_l2', 0, 10),
                'min_split_gain': trial.suggest_float('min_split_gain', 0, 1),
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'n_jobs': -1,
                'random_state': self.random_state,
                'verbosity': -1,
                'force_col_wise': True
            }
            
            # 创建模型
            model = lgb.LGBMRegressor(**params)
            
            # 训练
            pruning_callback = LightGBMPruningCallback(trial, 'valid_0')
            
            fit_params = {
                'eval_set': [(X_val, y_val)],
                'eval_metric': 'rmse',
                'callbacks': [
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0),
                    pruning_callback
                ]
            }
            
            if categorical_indices:
                fit_params['categorical_feature'] = categorical_indices
                
            model.fit(X_train, y_train, **fit_params)
            
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
            # LightGBM提供两种重要性：split和gain
            importance_split = model.feature_importances_  # 默认是split
            
            # 获取gain重要性
            if hasattr(model, 'booster_'):
                importance_gain = model.booster_.feature_importance(importance_type='gain')
            else:
                importance_gain = importance_split
                
            for i, feature in enumerate(self.feature_names):
                key_split = f"{feature}_{target_name}_split"
                key_gain = f"{feature}_{target_name}_gain"
                importance_dict[key_split] = importance_split[i]
                importance_dict[key_gain] = importance_gain[i]
                
        # 转换为DataFrame
        importance_df = pd.DataFrame(
            list(importance_dict.items()),
            columns=['feature', 'importance']
        )
        
        # 分别计算split和gain的平均重要性
        importance_df['feature_base'] = importance_df['feature'].str.rsplit('_', n=2).str[0]
        importance_df['importance_type'] = importance_df['feature'].str.rsplit('_', n=1).str[1]
        
        # 按类型分组计算平均值
        avg_importance = importance_df.groupby(['feature_base', 'importance_type'])['importance'].mean().reset_index()
        avg_importance_pivot = avg_importance.pivot(index='feature_base', columns='importance_type', values='importance').reset_index()
        avg_importance_pivot.columns = ['feature', 'importance_gain', 'importance_split']
        
        # 计算综合重要性（split和gain的加权平均）
        avg_importance_pivot['importance_combined'] = (
            0.5 * avg_importance_pivot['importance_split'] + 
            0.5 * avg_importance_pivot['importance_gain']
        )
        
        # 排序
        avg_importance_pivot = avg_importance_pivot.sort_values('importance_combined', ascending=False)
        
        return avg_importance_pivot
        
    def plot_feature_importance(self, top_n: int = 20, importance_type: str = 'combined'):
        """
        绘制特征重要性图
        
        Args:
            top_n: 显示前N个重要特征
            importance_type: 重要性类型 ('split', 'gain', 'combined')
        """
        import matplotlib.pyplot as plt
        
        importance_df = self.get_feature_importance()
        
        # 选择重要性类型
        importance_col = f'importance_{importance_type}'
        if importance_col not in importance_df.columns:
            importance_col = 'importance_combined'
            
        # 选择前N个
        top_features = importance_df.nlargest(top_n, importance_col)
        
        # 绘图
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features[importance_col])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel(f'Feature Importance ({importance_type})')
        plt.title(f'Top {top_n} Feature Importance - LightGBM')
        plt.tight_layout()
        plt.show()