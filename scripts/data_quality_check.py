#!/usr/bin/env python
"""
数据质量检查脚本
检查各个数据表的数据质量问题
"""
import os
import sys
import argparse
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.database import DatabaseManager
from src.data.dao import PowerMarketDAO
from config.logging_config import setup_logging, get_logger

# 设置日志
setup_logging()
logger = get_logger(__name__)


class DataQualityChecker:
    """数据质量检查器"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.dao = PowerMarketDAO(db_manager)
        self.issues = []
        
    def check_all_tables(self, start_date: date, end_date: date) -> Dict[str, pd.DataFrame]:
        """
        检查所有表的数据质量
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            各表的质量报告
        """
        logger.info(f"开始数据质量检查: {start_date} 到 {end_date}")
        
        reports = {}
        
        # 检查各个表
        tables_to_check = [
            ('processed_data', 'day_ahead_price'),
            ('processed_data', 'real_time_price'),
            ('load_forecast_data', 'load_forecast_mw'),
            ('renewable_forecast_data', 'renewable_forecast_value'),
            ('coal_forecast_data', 'coal_forecast_value'),
            ('hydropower_forecast_data', 'hydropower_forecast_value'),
            ('nuclear_forecast_data', 'nuclear_forecast_value'),
            ('inter_line_forecast_data', 'inter_line_forecast_value'),
            ('maintenance_plan_data', 'maintenance_capacity'),
            ('non_market_generation_forecast_data', 'non_market_forecast_value'),
            ('power_generation_forecast_data', 'generation_forecast_value')
        ]
        
        for table_name, value_column in tables_to_check:
            logger.info(f"检查表: {table_name}")
            try:
                report = self.dao.check_data_quality(
                    table_name, value_column, start_date, end_date
                )
                reports[table_name] = report
                
                # 分析问题
                if not report.empty:
                    self._analyze_issues(table_name, report)
                    
            except Exception as e:
                logger.error(f"检查 {table_name} 失败: {e}")
                
        return reports
        
    def check_data_completeness(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        检查数据完整性
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            完整性报告
        """
        logger.info("检查数据完整性...")
        
        # 获取综合数据
        df = self.dao.get_comprehensive_features(start_date, end_date)
        
        if df.empty:
            logger.warning("未找到数据")
            return pd.DataFrame()
            
        # 检查每天是否有96个点
        completeness = df.groupby('date').agg({
            'time_interval': 'count',
            'day_ahead_price': lambda x: x.notna().sum(),
            'real_time_price': lambda x: x.notna().sum(),
            'load_actual': lambda x: x.notna().sum(),
            'wind_actual': lambda x: x.notna().sum(),
            'solar_actual': lambda x: x.notna().sum()
        })
        
        completeness.columns = ['点数', '日前电价', '实时电价', '负荷', '风电', '光伏']
        completeness['完整率'] = completeness['点数'] / 96 * 100
        
        # 标记不完整的日期
        incomplete_dates = completeness[completeness['点数'] < 96]
        if not incomplete_dates.empty:
            self.issues.append({
                'type': '数据不完整',
                'description': f"发现 {len(incomplete_dates)} 天数据不完整",
                'dates': incomplete_dates.index.tolist()
            })
            
        return completeness
        
    def check_data_consistency(self, start_date: date, end_date: date) -> Dict[str, pd.DataFrame]:
        """
        检查数据一致性
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            一致性检查结果
        """
        logger.info("检查数据一致性...")
        
        consistency_issues = {}
        
        # 获取数据
        df = self.dao.get_comprehensive_features(start_date, end_date)
        
        if df.empty:
            return consistency_issues
            
        # 1. 检查供需平衡
        if all(col in df.columns for col in ['load_actual', 'total_generation', 'inter_line_power']):
            df['balance_check'] = df['load_actual'] - df['total_generation'] - df['inter_line_power'].fillna(0)
            imbalance = df[abs(df['balance_check']) > 100]  # 误差大于100MW
            
            if not imbalance.empty:
                consistency_issues['供需不平衡'] = imbalance[['date', 'time_point', 'load_actual', 
                                                          'total_generation', 'inter_line_power', 'balance_check']]
                                                          
        # 2. 检查预测与实际的偏差
        forecast_actual_pairs = [
            ('load_forecast', 'load_actual', 0.2),  # 20%偏差阈值
            ('wind_forecast', 'wind_actual', 0.5),   # 50%偏差阈值
            ('solar_forecast', 'solar_actual', 0.5)  # 50%偏差阈值
        ]
        
        for forecast_col, actual_col, threshold in forecast_actual_pairs:
            if all(col in df.columns for col in [forecast_col, actual_col]):
                df[f'{forecast_col}_error_rate'] = abs(df[actual_col] - df[forecast_col]) / (df[forecast_col] + 1e-6)
                large_errors = df[df[f'{forecast_col}_error_rate'] > threshold]
                
                if not large_errors.empty:
                    consistency_issues[f'{forecast_col}_大偏差'] = large_errors[
                        ['date', 'time_point', forecast_col, actual_col, f'{forecast_col}_error_rate']
                    ]
                    
        # 3. 检查价格异常
        if 'day_ahead_price' in df.columns:
            # 负电价
            negative_prices = df[df['day_ahead_price'] < 0]
            if not negative_prices.empty:
                consistency_issues['负电价'] = negative_prices[['date', 'time_point', 'day_ahead_price']]
                
            # 极高电价（大于2000元/MWh）
            high_prices = df[df['day_ahead_price'] > 2000]
            if not high_prices.empty:
                consistency_issues['极高电价'] = high_prices[['date', 'time_point', 'day_ahead_price']]
                
        return consistency_issues
        
    def generate_quality_report(self, 
                              start_date: date, 
                              end_date: date,
                              output_path: str = 'data_quality_report.html') -> None:
        """
        生成数据质量报告
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            output_path: 输出路径
        """
        logger.info("生成数据质量报告...")
        
        # 执行各项检查
        table_reports = self.check_all_tables(start_date, end_date)
        completeness_report = self.check_data_completeness(start_date, end_date)
        consistency_issues = self.check_data_consistency(start_date, end_date)
        
        # 生成HTML报告
        html_content = f"""
        <html>
        <head>
            <title>数据质量检查报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .issue {{ background-color: #ffdddd; }}
                .warning {{ background-color: #fff3cd; }}
                .good {{ background-color: #d4edda; }}
            </style>
        </head>
        <body>
            <h1>数据质量检查报告</h1>
            <p>检查时间范围: {start_date} 到 {end_date}</p>
            <p>报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>1. 数据完整性</h2>
            {self._df_to_html(completeness_report)}
            
            <h2>2. 各表数据质量问题</h2>
        """
        
        for table_name, report in table_reports.items():
            if not report.empty:
                html_content += f"<h3>{table_name}</h3>"
                html_content += self._df_to_html(report)
                
        html_content += "<h2>3. 数据一致性问题</h2>"
        for issue_type, issue_df in consistency_issues.items():
            if not issue_df.empty:
                html_content += f"<h3>{issue_type}</h3>"
                html_content += self._df_to_html(issue_df.head(50))  # 只显示前50条
                
        html_content += f"""
            <h2>4. 问题汇总</h2>
            <ul>
        """
        
        for issue in self.issues:
            html_content += f"<li><strong>{issue['type']}</strong>: {issue['description']}</li>"
            
        html_content += """
            </ul>
        </body>
        </html>
        """
        
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        logger.info(f"报告已保存至: {output_path}")
        
    def _analyze_issues(self, table_name: str, report: pd.DataFrame):
        """分析数据问题"""
        if report['null_values'].sum() > 0:
            null_dates = report[report['null_values'] > 0]['date'].tolist()
            self.issues.append({
                'type': '空值',
                'description': f"{table_name} 在 {len(null_dates)} 天存在空值",
                'dates': null_dates
            })
            
        if report['negative_values'].sum() > 0:
            negative_dates = report[report['negative_values'] > 0]['date'].tolist()
            self.issues.append({
                'type': '负值',
                'description': f"{table_name} 在 {len(negative_dates)} 天存在负值",
                'dates': negative_dates
            })
            
    def _df_to_html(self, df: pd.DataFrame) -> str:
        """DataFrame转HTML"""
        if df.empty:
            return "<p>无数据</p>"
        return df.to_html(classes='table', index=True)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='数据质量检查工具')
    parser.add_argument('--start-date', type=str, required=True, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='data_quality_report.html', help='输出文件路径')
    
    args = parser.parse_args()
    
    # 解析日期
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    
    # 初始化数据库
    logger.info("初始化数据库连接...")
    db_manager = DatabaseManager()
    db_manager.init_engine()
    
    # 创建检查器
    checker = DataQualityChecker(db_manager)
    
    # 生成报告
    checker.generate_quality_report(start_date, end_date, args.output)
    
    # 关闭数据库连接
    db_manager.close()
    
    logger.info("数据质量检查完成!")


if __name__ == "__main__":
    main()