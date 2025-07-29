# 数据字典

## 1. 基础字段

### 时间相关
- `date` / `target_date` / `forecast_target_date` - 日期（YYYY-MM-DD）
- `time_point` - 时间点（HH:MM格式，如 "00:15", "00:30"）
- `time_interval` - 时间间隔序号（1-96，每15分钟一个点）
- `forecast_generation_date` - 预测生成日期
- `days_ahead` - 提前天数

### 版本控制（重要）
- `forecast_generation_date` - 预测生成日期（决定版本新旧的关键字段）
- `forecast_target_date` - 预测目标日期
- `forecast_version` - 预测版本号（递增编号）
- `forecast_version_label` - 版本标签（如 "D-1", "D-2"）
- `days_ahead` - 提前天数 (target_date - generation_date)
- `is_latest_version` - 是否为最新版本标记（不可靠，仅供参考）
- `is_superseded` - 是否已被新版本替代

**重要说明**：
- 最新版本的判定基于 `forecast_generation_date` 的最大值
- 不要依赖 `is_latest_version` 字段，它可能未及时更新
- 查询时应使用窗口函数或子查询获取最新版本

## 2. 电价数据

### 来源表：processed_data
- `day_ahead_price` - 日前电价（元/MWh）
- `real_time_price` - 实时电价（元/MWh）

## 3. 负荷数据

### 来源表：load_forecast_data
- `load_forecast` - 负荷预测值（MW）
- `load_actual` - 负荷实际值（MW）

## 4. 新能源数据

### 来源表：renewable_forecast_data
- `wind_forecast` - 风电出力预测（MW）
- `wind_actual` - 风电出力实际（MW）
- `solar_forecast` - 光伏出力预测（MW）
- `solar_actual` - 光伏出力实际（MW）
- `renewable_type` - 新能源类型（"风电" / "光伏"）

## 5. 常规电源数据

### 地方煤（coal_forecast_data）
- `coal_actual` / `coal_output` - 地方煤出力实际（MW）
- `coal_forecast` - 地方煤出力预测（MW）

### 水电（hydropower_forecast_data）
- `hydro_actual` / `hydro_output` - 水电出力实际（MW）
- `hydro_forecast` - 水电出力预测（MW）

### 核电（nuclear_forecast_data）
- `nuclear_actual` / `nuclear_output` - 核电出力实际（MW）
- `nuclear_forecast` - 核电出力预测（MW）

## 6. 其他数据

### 省间联络线（inter_line_forecast_data）
- `inter_line_forecast` - 联络线功率预测（MW，正值为送入）
- `inter_line_actual` / `inter_line_power` - 联络线功率实际（MW）
- `line_direction` - 线路方向（"送入" / "送出"）

### 检修计划（maintenance_plan_data）
- `maintenance_capacity` - 检修容量（MW）
- `equipment_type` - 设备类型
- `maintenance_type` - 检修类型

### 非市场化发电（non_market_generation_forecast_data）
- `non_market_actual` / `non_market_output` - 非市场化发电实际（MW）
- `non_market_forecast` - 非市场化发电预测（MW）

### 总发电（power_generation_forecast_data）
- `total_generation_actual` / `total_generation` - 总发电出力实际（MW）
- `total_generation_forecast` - 总发电出力预测（MW）

## 7. 衍生特征

### 供需平衡特征
- `supply_demand_ratio` - 供需比（总发电/负荷）
- `supply_demand_diff` - 供需差（总发电-负荷）
- `supply_demand_gap_pct` - 供需缺口百分比
- `net_load` - 净负荷（负荷-联络线）
- `market_tightness` - 市场紧张度
- `reserve_margin` - 备用率

### 新能源特征
- `renewable_total` - 新能源总出力（风电+光伏）
- `renewable_ratio` - 新能源占比
- `renewable_load_ratio` - 新能源负荷比
- `wind_volatility_*` - 风电波动率
- `solar_volatility_*` - 光伏波动率
- `wind_solar_complement` - 风光互补指数

### 预测误差特征
- `*_forecast_error` - 预测误差（实际-预测）
- `*_forecast_error_pct` - 预测误差百分比
- `*_forecast_error_abs` - 预测误差绝对值

### 时间特征
- `hour`, `minute` - 小时、分钟
- `dayofweek` - 星期几（0=周一，6=周日）
- `is_weekend` - 是否周末
- `is_peak_hour` - 是否峰时段
- `is_valley_hour` - 是否谷时段
- `month_sin`, `month_cos` - 月份周期编码
- `hour_sin`, `hour_cos` - 小时周期编码

### 节假日特征
- `is_holiday` - 是否节假日
- `is_workday` - 是否工作日
- `holiday_name` - 节假日名称
- `days_to_holiday` - 距离下个节假日天数
- `is_major_holiday` - 是否重大节假日

## 8. 单位说明

- **功率单位**：MW（兆瓦）
- **电价单位**：元/MWh（元/兆瓦时）
- **百分比**：%（0-100）
- **比率**：无单位（通常0-1之间）

## 9. 数据质量指标

- `data_quality_score` - 数据质量评分（0-1）
- `is_valid` - 数据是否有效（0/1）
- `validation_errors` - 验证错误信息（JSON）

## 10. 注意事项

1. **地方煤**：指自备电厂等小火电，不是所有火电
2. **联络线**：正值表示送入本省，负值表示送出
3. **非市场化发电**：包括优先发电等政策性电源
4. **时间间隔**：1-96对应00:00-23:45，每15分钟一个点
5. **数据完整性**：每天应有96个时间点的数据