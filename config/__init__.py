"""
配置加载模块
"""
import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Config:
    """配置类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置
        
        Args:
            config_path: 配置文件路径
        """
        if config_path is None:
            # 默认配置文件路径
            config_path = os.path.join(
                os.path.dirname(__file__), 
                'config.yaml'
            )
            
        self.config_path = config_path
        self._config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            logger.warning(f"配置文件不存在: {self.config_path}")
            return {}
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.info(f"配置文件加载成功: {self.config_path}")
                return config or {}
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            return {}
            
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项
        
        Args:
            key: 配置键，支持点号分隔的嵌套键 (例如: 'models.xgboost.params')
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
        
    def get_data_config(self) -> Dict[str, Any]:
        """获取数据配置"""
        return self.get('data', {})
        
    def get_model_config(self, model_type: str = 'traditional') -> Dict[str, Any]:
        """
        获取模型配置
        
        Args:
            model_type: 模型类型 ('traditional' 或 'deep_learning')
        """
        return self.get(f'models.{model_type}', {})
        
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return self.get('training', {})
        
    def get_prediction_config(self) -> Dict[str, Any]:
        """获取预测配置"""
        return self.get('prediction', {})
        
    def get_api_config(self) -> Dict[str, Any]:
        """获取API配置"""
        return self.get('api', {})
        
    def update(self, updates: Dict[str, Any]) -> None:
        """
        更新配置
        
        Args:
            updates: 要更新的配置项
        """
        self._config.update(updates)
        
    def save(self, path: Optional[str] = None) -> None:
        """
        保存配置
        
        Args:
            path: 保存路径
        """
        save_path = path or self.config_path
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"配置已保存至: {save_path}")
        
    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self.get(key)
        
    def __contains__(self, key: str) -> bool:
        """支持 in 操作"""
        return self.get(key) is not None
        
    def __repr__(self) -> str:
        return f"Config(path='{self.config_path}')"


# 全局配置实例
_config_instance = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    获取全局配置实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置实例
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config(config_path)
        
    return _config_instance


def load_config(config_path: str) -> Config:
    """
    加载新的配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        新的配置实例
    """
    return Config(config_path)