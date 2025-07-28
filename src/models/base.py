"""
基础模型类
所有预测模型的基类
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """基础模型抽象类"""
    
    def __init__(self, model_name: str, model_version: str = "1.0", **kwargs):
        """
        初始化基础模型
        
        Args:
            model_name: 模型名称
            model_version: 模型版本
            **kwargs: 其他参数
        """
        self.model_name = model_name
        self.model_version = model_version
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.target_names = None
        self.model_params = kwargs
        self.training_history = []
        self.logger = logger
        
        # 模型元数据
        self.metadata = {
            'model_name': model_name,
            'model_version': model_version,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'model_params': kwargs,
            'metrics': {}
        }
        
    @abstractmethod
    def build_model(self) -> Any:
        """构建模型"""
        pass
        
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame, 
            X_val: Optional[pd.DataFrame] = None, 
            y_val: Optional[pd.DataFrame] = None, **kwargs) -> 'BaseModel':
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            **kwargs: 其他训练参数
            
        Returns:
            self
        """
        pass
        
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测
        
        Args:
            X: 特征数据
            
        Returns:
            预测结果
        """
        pass
        
    def fit_predict(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                   X_test: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        训练并预测
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_test: 测试特征
            **kwargs: 其他参数
            
        Returns:
            预测结果
        """
        self.fit(X_train, y_train, **kwargs)
        return self.predict(X_test)
        
    def save_model(self, filepath: str, include_data: bool = False) -> None:
        """
        保存模型
        
        Args:
            filepath: 保存路径
            include_data: 是否包含训练数据
        """
        self.logger.info(f"保存模型到 {filepath}")
        
        # 创建目录
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 更新元数据
        self.metadata['updated_at'] = datetime.now().isoformat()
        self.metadata['is_fitted'] = self.is_fitted
        self.metadata['feature_names'] = self.feature_names
        self.metadata['target_names'] = self.target_names
        
        # 保存内容
        save_dict = {
            'model': self.model,
            'metadata': self.metadata,
            'model_params': self.model_params,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'target_names': self.target_names
        }
        
        if include_data:
            save_dict['training_history'] = self.training_history
            
        # 保存
        joblib.dump(save_dict, filepath)
        
        # 同时保存元数据为JSON（便于查看）
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"模型保存成功")
        
    def load_model(self, filepath: str) -> None:
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
        """
        self.logger.info(f"从 {filepath} 加载模型")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
            
        # 加载
        save_dict = joblib.load(filepath)
        
        # 恢复属性
        self.model = save_dict['model']
        self.metadata = save_dict['metadata']
        self.model_params = save_dict['model_params']
        self.is_fitted = save_dict['is_fitted']
        self.feature_names = save_dict['feature_names']
        self.target_names = save_dict['target_names']
        
        if 'training_history' in save_dict:
            self.training_history = save_dict['training_history']
            
        self.logger.info(f"模型加载成功")
        
    def evaluate(self, X: pd.DataFrame, y_true: pd.DataFrame) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            X: 特征数据
            y_true: 真实值
            
        Returns:
            评估指标字典
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        y_pred = self.predict(X)
        
        # 如果是多输出，分别计算每个输出的指标
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            metrics = {}
            for i, col in enumerate(self.target_names or range(y_true.shape[1])):
                metrics[f'{col}_mae'] = mean_absolute_error(y_true.iloc[:, i], y_pred[:, i])
                metrics[f'{col}_rmse'] = np.sqrt(mean_squared_error(y_true.iloc[:, i], y_pred[:, i]))
                metrics[f'{col}_mape'] = self._calculate_mape(y_true.iloc[:, i], y_pred[:, i])
                metrics[f'{col}_r2'] = r2_score(y_true.iloc[:, i], y_pred[:, i])
        else:
            # 单输出
            metrics = {
                'mae': mean_absolute_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mape': self._calculate_mape(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }
            
        # 计算方向准确率
        if len(y_true) > 1:
            metrics['direction_accuracy'] = self._calculate_direction_accuracy(y_true, y_pred)
            
        # 保存到元数据
        self.metadata['metrics'] = metrics
        
        return metrics
        
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        获取特征重要性
        
        Returns:
            特征重要性DataFrame
        """
        if not hasattr(self.model, 'feature_importances_'):
            self.logger.warning("当前模型不支持特征重要性")
            return None
            
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
        
        return importance_df
        
    def predict_with_uncertainty(self, X: pd.DataFrame, 
                               confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        带不确定性的预测
        
        Args:
            X: 特征数据
            confidence_level: 置信水平
            
        Returns:
            (预测值, 下界, 上界)
        """
        # 默认实现：使用历史误差估计不确定性
        y_pred = self.predict(X)
        
        # 这里需要子类实现具体的不确定性估计方法
        # 默认返回预测值和正负10%的区间
        lower_bound = y_pred * 0.9
        upper_bound = y_pred * 1.1
        
        return y_pred, lower_bound, upper_bound
        
    def _calculate_mape(self, y_true: Union[pd.Series, np.ndarray], 
                       y_pred: Union[pd.Series, np.ndarray]) -> float:
        """计算MAPE"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
    def _calculate_direction_accuracy(self, y_true: Union[pd.Series, np.ndarray], 
                                    y_pred: Union[pd.Series, np.ndarray]) -> float:
        """计算方向准确率"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if len(y_true.shape) > 1:
            # 多输出情况，计算平均方向准确率
            accuracies = []
            for i in range(y_true.shape[1]):
                true_direction = np.diff(y_true[:, i]) > 0
                pred_direction = np.diff(y_pred[:, i]) > 0
                accuracy = np.mean(true_direction == pred_direction)
                accuracies.append(accuracy)
            return np.mean(accuracies)
        else:
            # 单输出
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            return np.mean(true_direction == pred_direction)
            
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.model_name}', version='{self.model_version}', fitted={self.is_fitted})"