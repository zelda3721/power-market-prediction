# src/__init__.py
"""电力市场预测系统"""
__version__ = "1.0.0"
__author__ = "Your Name"

# src/data/__init__.py
"""数据访问层"""
from .database import DatabaseManager, db_manager
from .dao import PowerMarketDAO
from .queries import *

__all__ = ['DatabaseManager', 'db_manager', 'PowerMarketDAO']

# src/preprocessing/__init__.py
"""数据预处理模块"""
from .preprocessor import DataPreprocessor
from .feature_engineering import FeatureEngineer
from .pipeline import ElectricityPricePipeline

__all__ = ['DataPreprocessor', 'FeatureEngineer', 'ElectricityPricePipeline']

# src/models/__init__.py
"""模型模块"""
from .base import BaseModel

__all__ = ['BaseModel']

# src/models/traditional/__init__.py
"""传统机器学习模型"""

# src/models/deep_learning/__init__.py
"""深度学习模型"""

# src/training/__init__.py
"""训练模块"""

# src/prediction/__init__.py
"""预测服务模块"""

# src/evaluation/__init__.py
"""评估模块"""

# src/utils/__init__.py
"""工具模块"""