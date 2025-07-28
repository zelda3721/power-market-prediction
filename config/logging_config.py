"""
日志配置模块
"""
import os
import logging
import logging.handlers
from datetime import datetime
import json


class JSONFormatter(logging.Formatter):
    """JSON格式化器"""
    
    def format(self, record):
        log_obj = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
        }
        
        if hasattr(record, 'user_id'):
            log_obj['user_id'] = record.user_id
            
        if hasattr(record, 'request_id'):
            log_obj['request_id'] = record.request_id
            
        if record.exc_info:
            log_obj['exc_info'] = self.formatException(record.exc_info)
            
        return json.dumps(log_obj, ensure_ascii=False)


def setup_logging(
    log_level: str = 'INFO',
    log_dir: str = 'logs',
    log_format: str = 'text',
    app_name: str = 'power_prediction'
):
    """
    设置日志配置
    
    Args:
        log_level: 日志级别
        log_dir: 日志目录
        log_format: 日志格式 ('text' 或 'json')
        app_name: 应用名称
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 获取根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除现有的处理器
    root_logger.handlers.clear()
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # 文件处理器（按日期轮转）
    log_file = os.path.join(log_dir, f'{app_name}.log')
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_file,
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, log_level.upper()))
    
    # 错误日志单独记录
    error_file = os.path.join(log_dir, f'{app_name}_error.log')
    error_handler = logging.handlers.TimedRotatingFileHandler(
        filename=error_file,
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    
    # 设置格式
    if log_format == 'json':
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)
    
    # 添加处理器
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    
    # 设置第三方库的日志级别
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    # 记录启动信息
    root_logger.info(f"日志系统初始化完成 - 级别: {log_level}, 格式: {log_format}")


def get_logger(name: str) -> logging.Logger:
    """
    获取日志器
    
    Args:
        name: 日志器名称
        
    Returns:
        日志器实例
    """
    return logging.getLogger(name)


# 便捷函数
def log_function_call(func):
    """函数调用日志装饰器"""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"调用函数 {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"函数 {func.__name__} 执行成功")
            return result
        except Exception as e:
            logger.error(f"函数 {func.__name__} 执行失败: {e}", exc_info=True)
            raise
    return wrapper


def log_execution_time(func):
    """执行时间日志装饰器"""
    import time
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.info(f"函数 {func.__name__} 执行时间: {execution_time:.2f}秒")
        return result
    return wrapper


# 初始化默认配置
if __name__ != "__main__":
    # 从环境变量读取配置
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    log_format = os.getenv('LOG_FORMAT', 'text')
    log_dir = os.getenv('LOG_DIR', 'logs')
    
    setup_logging(
        log_level=log_level,
        log_format=log_format,
        log_dir=log_dir
    )