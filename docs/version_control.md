# 数据版本控制说明

## 1. 版本控制原理

在辽宁电力市场预测系统中，预测数据会多次更新。版本控制基于以下字段：

- **forecast_generation_date** - 预测生成日期（关键字段）
- **forecast_target_date** - 预测目标日期
- **forecast_version** - 版本号（辅助字段）
- **is_latest_version** - 最新版本标记（可能不准确）

## 2. 版本控制逻辑

### 2.1 最新版本定义
对于同一个 `forecast_target_date` 和 `time_point`：
- **最新版本** = `forecast_generation_date` 最大的记录
- 不依赖 `is_latest_version` 字段（该字段可能未及时更新）

### 2.2 版本查询策略

#### 使用窗口函数（推荐）
```sql
WITH data_ranked AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (
            PARTITION BY forecast_target_date, time_point, data_type 
            ORDER BY forecast_generation_date DESC
        ) as rn
    FROM load_forecast_data
    WHERE forecast_target_date = '2025-01-15'
)
SELECT * FROM data_ranked WHERE rn = 1
```

#### 使用子查询
```sql
SELECT * FROM load_forecast_data l1
WHERE forecast_generation_date = (
    SELECT MAX(forecast_generation_date) 
    FROM load_forecast_data l2
    WHERE l2.forecast_target_date = l1.forecast_target_date
        AND l2.time_point = l1.time_point
        AND l2.data_type = l1.data_type
)
```

## 3. 实际应用场景

### 3.1 日前市场预测
- D-7：提前7天的初步预测
- D-3：提前3天的更新预测
- D-1：日前最终预测（最重要）

### 3.2 数据更新时序
```
时间线示例（预测2025-01-15的数据）：
- 2025-01-08 生成第一版预测 (forecast_generation_date = 2025-01-08)
- 2025-01-12 更新预测 (forecast_generation_date = 2025-01-12)
- 2025-01-14 最终预测 (forecast_generation_date = 2025-01-14) ← 这是最新版本
```

## 4. 查询优化建议

### 4.1 索引优化
确保以下字段有合适的索引：
```sql
INDEX idx_version_control (forecast_target_date, time_point, data_type, forecast_generation_date DESC)
```

### 4.2 性能考虑
- 对于大数据量，使用窗口函数比子查询更高效
- 可以考虑创建物化视图存储最新版本数据
- 定期清理旧版本数据（保留最近N个版本）

## 5. 注意事项

1. **数据一致性**：确保所有相关表都使用相同的版本
2. **时区问题**：注意 `forecast_generation_date` 的时区设置
3. **缺失数据**：某些时间点可能没有所有类型的数据
4. **版本回退**：特殊情况下可能需要使用非最新版本

## 6. 代码示例

### Python中获取最新版本
```python
def get_latest_forecast(dao, target_date, data_type):
    """获取最新版本的预测数据"""
    # DAO已经封装了基于forecast_generation_date的查询
    return dao.get_load_forecast_data(target_date, data_type)
```

### 获取特定版本
```python
def get_specific_version(dao, target_date, generation_date):
    """获取特定生成日期的版本"""
    query = """
    SELECT * FROM load_forecast_data
    WHERE forecast_target_date = %s
        AND forecast_generation_date = %s
    ORDER BY time_interval
    """
    return dao.db.execute_query(query, [target_date, generation_date])
```

## 7. 版本管理最佳实践

1. **版本标签**：使用 `forecast_version_label` 记录版本含义（如 "D-1", "紧急更新"）
2. **版本日志**：记录每次更新的原因和变更内容
3. **版本对比**：定期分析不同版本间的差异，评估预测准确性的提升
4. **自动化检查**：实现自动检查数据完整性和版本一致性的脚本