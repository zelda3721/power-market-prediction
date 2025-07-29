"""
集成模型 - 结合多个模型的预测
"""
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import optuna

from ..base import BaseModel
from .xgboost_model import XGBoostPriceModel
from .lightgbm_model import LightGBMPriceModel

logger = logging.getLogger(__name__)


class EnsemblePriceModel(BaseModel):
    """集成电价预测模型"""
    
    def __init__(self, 
                 model_name: str = "Ensemble_Price_Model",
                 model_version: str = "1.0",
                 ensemble_method: str = "stacking",
                 base_models: Optional[List[str]] = None,
                 random_state: int = 42,
                 **kwargs):
        """
        初始化集成模型
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            ensemble_method: 集成方法 ('stacking', 'blending', 'voting', 'weighted')
            base_models: 基础模型列表
            random_state: 随机种子
            **kwargs: 其他参数
        """
        super().__init__(model_name, model_version, **kwargs)
        self.ensemble_method = ensemble_method
        self.random_state = random_state
        
        # 默认使用XGBoost和LightGBM
        if base_models is None:
            base_models = ['xgboost', 'lightgbm']
        self.base_model_names = base_models
        
        # 存储基础模型和元模型
        self.base_models = {}
        self.meta_models = {}  # 为每个目标存储一个元模型
        self.weights = {}  # 存储模型权重
        
        # Stacking特征
        self.use_original_features = kwargs.get('use_original_features', True)
        self.meta_model_type = kwargs.get('meta_model_type', 'linear')
        
    def _create_base_model(self, model_name: str) -> BaseModel:
        """
        创建基础模型实例
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型实例
        """
        if model_name == 'xgboost':
            return XGBoostPriceModel(
                model_name=f"XGB_base_{self.model_name}",
                random_state=self.random_state
            )
        elif model_name == 'lightgbm':
            return LightGBMPriceModel(
                model_name=f"LGB_base_{self.model_name}",
                random_state=self.random_state
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_name}")
            
    def build_model(self) -> None:
        """构建所有基础模型"""
        for model_name in self.base_model_names:
            self.base_models[model_name] = self._create_base_model(model_name)
            
    def fit(self, 
            X_train: pd.DataFrame, 
            y_train: pd.DataFrame,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.DataFrame] = None,
            optimize_hyperparams: bool = False,
            optimize_weights: bool = True,
            **kwargs) -> 'EnsemblePriceModel':
        """
        训练集成模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            optimize_hyperparams: 是否优化基础模型超参数
            optimize_weights: 是否优化集成权重
            **kwargs: 其他参数
            
        Returns:
            self
        """
        self.logger.info(f"开始训练集成模型 (方法: {self.ensemble_method})...")
        
        # 保存特征和目标信息
        self.feature_names = list(X_train.columns)
        if isinstance(y_train, pd.DataFrame):
            self.target_names = list(y_train.columns)
        else:
            self.target_names = ['target']
            
        # 创建基础模型
        self.build_model()
        
        if self.ensemble_method == 'stacking':
            self._fit_stacking(X_train, y_train, X_val, y_val, optimize_hyperparams)
        elif self.ensemble_method == 'blending':
            self._fit_blending(X_train, y_train, X_val, y_val, optimize_hyperparams)
        elif self.ensemble_method == 'voting':
            self._fit_voting(X_train, y_train, X_val, y_val, optimize_hyperparams)
        elif self.ensemble_method == 'weighted':
            self._fit_weighted(X_train, y_train, X_val, y_val, optimize_hyperparams, optimize_weights)
        else:
            raise ValueError(f"不支持的集成方法: {self.ensemble_method}")
            
        self.is_fitted = True
        
        # 评估性能
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            self.logger.info(f"验证集性能: {val_metrics}")
            
        return self
        
    def _fit_stacking(self, 
                     X_train: pd.DataFrame, 
                     y_train: pd.DataFrame,
                     X_val: Optional[pd.DataFrame],
                     y_val: Optional[pd.DataFrame],
                     optimize_hyperparams: bool):
        """Stacking集成训练"""
        self.logger.info("使用Stacking方法训练...")
        
        # 分割训练集用于训练基础模型和元模型
        if X_val is None:
            split_idx = int(len(X_train) * 0.8)
            X_base = X_train.iloc[:split_idx]
            y_base = y_train.iloc[:split_idx]
            X_meta = X_train.iloc[split_idx:]
            y_meta = y_train.iloc[split_idx:]
        else:
            X_base = X_train
            y_base = y_train
            X_meta = X_val
            y_meta = y_val
            
        # 训练基础模型并获取预测
        base_predictions = {}
        for name, model in self.base_models.items():
            self.logger.info(f"训练基础模型: {name}")
            model.fit(X_base, y_base, optimize_hyperparams=optimize_hyperparams)
            
            # 获取元训练集的预测
            pred = model.predict(X_meta)
            base_predictions[name] = pred
            
        # 构建元特征
        meta_features = self._create_meta_features(base_predictions, X_meta)
        
        # 训练元模型
        if isinstance(y_meta, pd.DataFrame):
            for i, target_name in enumerate(self.target_names):
                self.logger.info(f"训练元模型: {target_name}")
                y_target = y_meta.iloc[:, i].values
                
                # 选择元模型类型
                if self.meta_model_type == 'linear':
                    meta_model = Ridge(alpha=1.0, random_state=self.random_state)
                elif self.meta_model_type == 'xgboost':
                    meta_model = xgb.XGBRegressor(
                        n_estimators=100,
                        max_depth=3,
                        learning_rate=0.1,
                        random_state=self.random_state
                    )
                else:
                    meta_model = LinearRegression()
                    
                meta_model.fit(meta_features, y_target)
                self.meta_models[target_name] = meta_model
        else:
            # 单目标
            if self.meta_model_type == 'linear':
                meta_model = Ridge(alpha=1.0, random_state=self.random_state)
            else:
                meta_model = LinearRegression()
                
            meta_model.fit(meta_features, y_meta.values.ravel())
            self.meta_models['target'] = meta_model
            
    def _fit_blending(self, 
                     X_train: pd.DataFrame, 
                     y_train: pd.DataFrame,
                     X_val: Optional[pd.DataFrame],
                     y_val: Optional[pd.DataFrame],
                     optimize_hyperparams: bool):
        """Blending集成训练（简化版Stacking）"""
        self.logger.info("使用Blending方法训练...")
        
        # Blending只使用验证集预测作为元特征
        if X_val is None:
            split_idx = int(len(X_train) * 0.8)
            X_base = X_train.iloc[:split_idx]
            y_base = y_train.iloc[:split_idx]
            X_blend = X_train.iloc[split_idx:]
            y_blend = y_train.iloc[split_idx:]
        else:
            X_base = X_train
            y_base = y_train
            X_blend = X_val
            y_blend = y_val
            
        # 训练基础模型
        base_predictions = {}
        for name, model in self.base_models.items():
            self.logger.info(f"训练基础模型: {name}")
            model.fit(X_base, y_base, optimize_hyperparams=optimize_hyperparams)
            
            # 获取blending集的预测
            pred = model.predict(X_blend)
            base_predictions[name] = pred
            
        # 使用简单平均或加权平均
        predictions = np.stack(list(base_predictions.values()), axis=0)
        
        # 优化权重
        if isinstance(y_blend, pd.DataFrame):
            for i, target_name in enumerate(self.target_names):
                y_target = y_blend.iloc[:, i].values
                weights = self._optimize_blend_weights(predictions[:, :, i] if predictions.ndim > 2 else predictions, y_target)
                self.weights[target_name] = weights
        else:
            weights = self._optimize_blend_weights(predictions, y_blend.values.ravel())
            self.weights['target'] = weights
            
    def _fit_voting(self, 
                   X_train: pd.DataFrame, 
                   y_train: pd.DataFrame,
                   X_val: Optional[pd.DataFrame],
                   y_val: Optional[pd.DataFrame],
                   optimize_hyperparams: bool):
        """Voting集成训练（简单平均）"""
        self.logger.info("使用Voting方法训练...")
        
        # 训练所有基础模型
        for name, model in self.base_models.items():
            self.logger.info(f"训练基础模型: {name}")
            model.fit(X_train, y_train, X_val, y_val, optimize_hyperparams=optimize_hyperparams)
            
        # Voting使用相等权重
        equal_weight = 1.0 / len(self.base_models)
        for target_name in self.target_names:
            self.weights[target_name] = [equal_weight] * len(self.base_models)
            
    def _fit_weighted(self, 
                     X_train: pd.DataFrame, 
                     y_train: pd.DataFrame,
                     X_val: Optional[pd.DataFrame],
                     y_val: Optional[pd.DataFrame],
                     optimize_hyperparams: bool,
                     optimize_weights: bool):
        """加权平均集成训练"""
        self.logger.info("使用加权平均方法训练...")
        
        # 如果没有验证集，创建一个
        if X_val is None:
            split_idx = int(len(X_train) * 0.8)
            X_train_split = X_train.iloc[:split_idx]
            y_train_split = y_train.iloc[:split_idx]
            X_val = X_train.iloc[split_idx:]
            y_val = y_train.iloc[split_idx:]
        else:
            X_train_split = X_train
            y_train_split = y_train
            
        # 训练所有基础模型
        val_predictions = {}
        for name, model in self.base_models.items():
            self.logger.info(f"训练基础模型: {name}")
            model.fit(X_train_split, y_train_split, X_val, y_val, optimize_hyperparams=optimize_hyperparams)
            
            # 获取验证集预测
            pred = model.predict(X_val)
            val_predictions[name] = pred
            
        # 优化权重
        if optimize_weights:
            predictions = np.stack(list(val_predictions.values()), axis=0)
            
            if isinstance(y_val, pd.DataFrame):
                for i, target_name in enumerate(self.target_names):
                    y_target = y_val.iloc[:, i].values
                    if predictions.ndim > 2:
                        weights = self._optimize_weights_optuna(predictions[:, :, i], y_target)
                    else:
                        weights = self._optimize_weights_optuna(predictions, y_target)
                    self.weights[target_name] = weights
            else:
                weights = self._optimize_weights_optuna(predictions, y_val.values.ravel())
                self.weights['target'] = weights
        else:
            # 使用性能反比权重
            self._calculate_performance_weights(X_val, y_val)
            
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
            
        # 获取基础模型预测
        base_predictions = {}
        for name, model in self.base_models.items():
            pred = model.predict(X)
            base_predictions[name] = pred
            
        if self.ensemble_method == 'stacking':
            # 构建元特征
            meta_features = self._create_meta_features(base_predictions, X)
            
            # 使用元模型预测
            predictions = []
            for target_name in self.target_names:
                meta_model = self.meta_models[target_name]
                pred = meta_model.predict(meta_features)
                predictions.append(pred)
                
            if len(predictions) == 1:
                return predictions[0]
            else:
                return np.column_stack(predictions)
                
        else:
            # 加权平均
            predictions = []
            for i, target_name in enumerate(self.target_names):
                weights = np.array(self.weights[target_name])
                
                if len(base_predictions) == 1:
                    # 单模型
                    pred = list(base_predictions.values())[0]
                else:
                    # 多模型加权
                    model_preds = []
                    for name in self.base_model_names:
                        model_pred = base_predictions[name]
                        if model_pred.ndim > 1:
                            model_preds.append(model_pred[:, i])
                        else:
                            model_preds.append(model_pred)
                            
                    model_preds = np.stack(model_preds, axis=0)
                    pred = np.sum(model_preds * weights.reshape(-1, 1), axis=0)
                    
                predictions.append(pred)
                
            if len(predictions) == 1:
                return predictions[0]
            else:
                return np.column_stack(predictions)
                
    def _create_meta_features(self, base_predictions: Dict[str, np.ndarray], X: pd.DataFrame) -> np.ndarray:
        """创建元特征"""
        features = []
        
        # 基础模型预测
        for name in self.base_model_names:
            pred = base_predictions[name]
            if pred.ndim == 1:
                features.append(pred.reshape(-1, 1))
            else:
                features.append(pred)
                
        # 是否包含原始特征
        if self.use_original_features:
            features.append(X.values)
            
        return np.hstack(features)
        
    def _optimize_blend_weights(self, predictions: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """优化blending权重"""
        n_models = predictions.shape[0]
        
        # 使用最小二乘法
        def objective(weights):
            weights = weights / weights.sum()  # 归一化
            pred = np.sum(predictions * weights.reshape(-1, 1), axis=0)
            return mean_squared_error(y_true, pred)
            
        # 初始权重
        initial_weights = np.ones(n_models) / n_models
        
        # 优化
        from scipy.optimize import minimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=[(0, 1)] * n_models,
            constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1}
        )
        
        return result.x
        
    def _optimize_weights_optuna(self, predictions: np.ndarray, y_true: np.ndarray, n_trials: int = 50) -> np.ndarray:
        """使用Optuna优化权重"""
        n_models = predictions.shape[0]
        
        def objective(trial):
            # 生成权重
            weights = []
            for i in range(n_models - 1):
                w = trial.suggest_float(f'w{i}', 0, 1)
                weights.append(w)
                
            # 最后一个权重由其他权重决定
            weights.append(1 - sum(weights))
            
            # 如果权重无效，返回大的损失
            if weights[-1] < 0 or weights[-1] > 1:
                return float('inf')
                
            weights = np.array(weights)
            pred = np.sum(predictions * weights.reshape(-1, 1), axis=0)
            return mean_squared_error(y_true, pred)
            
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # 提取最优权重
        best_weights = []
        for i in range(n_models - 1):
            best_weights.append(study.best_params[f'w{i}'])
        best_weights.append(1 - sum(best_weights))
        
        return np.array(best_weights)
        
    def _calculate_performance_weights(self, X_val: pd.DataFrame, y_val: pd.DataFrame):
        """基于性能计算权重"""
        performances = []
        
        for name, model in self.base_models.items():
            pred = model.predict(X_val)
            
            if isinstance(y_val, pd.DataFrame):
                # 多目标
                rmse_list = []
                for i in range(y_val.shape[1]):
                    if pred.ndim > 1:
                        rmse = np.sqrt(mean_squared_error(y_val.iloc[:, i], pred[:, i]))
                    else:
                        rmse = np.sqrt(mean_squared_error(y_val.iloc[:, i], pred))
                    rmse_list.append(rmse)
                performances.append(np.mean(rmse_list))
            else:
                rmse = np.sqrt(mean_squared_error(y_val, pred))
                performances.append(rmse)
                
        # 性能反比权重
        performances = np.array(performances)
        weights = 1 / performances
        weights = weights / weights.sum()
        
        # 为每个目标设置相同的权重
        for target_name in self.target_names:
            self.weights[target_name] = weights
            
    def get_model_weights(self) -> pd.DataFrame:
        """获取模型权重"""
        if not self.weights:
            raise ValueError("模型权重尚未计算")
            
        weight_data = []
        for target_name, weights in self.weights.items():
            for i, model_name in enumerate(self.base_model_names):
                weight_data.append({
                    'target': target_name,
                    'model': model_name,
                    'weight': weights[i]
                })
                
        return pd.DataFrame(weight_data)