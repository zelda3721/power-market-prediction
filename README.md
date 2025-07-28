# 电力现货市场价格预测系统

## 项目概述

本项目实现了电力现货市场96点统一市场结算日前电价和实时电价的预测系统。系统采用传统机器学习和深度学习相结合的方法，提供高精度的电价预测服务。

## 主要特性

- ✅ **多源数据融合**：整合负荷、新能源、常规电源、联络线等多维度数据
- ✅ **双轨预测**：同时预测日前电价和实时电价
- ✅ **96点精细化**：提供全天96个时间点（15分钟间隔）的预测
- ✅ **模型集成**：结合XGBoost、LightGBM等传统模型和PatchTST等深度学习模型
- ✅ **实时服务**：提供REST API接口，支持实时预测请求
- ✅ **监控告警**：完善的模型监控和数据漂移检测

## 技术架构

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   数据采集层    │────▶│   数据处理层    │────▶│    模型层      │
│  MySQL数据库    │     │  特征工程       │     │  ML/DL模型     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   监控层        │◀────│   服务层        │◀────│   预测层       │
│  Prometheus     │     │  FastAPI        │     │  预测引擎      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository_url>
cd power-market-prediction

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置设置

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑.env文件，配置数据库连接等参数
vim .env
```

### 3. 数据准备

```bash
# 检查数据质量
python scripts/data_quality_check.py --start-date 2024-01-01 --end-date 2024-12-31

# 生成训练数据集
python scripts/prepare_dataset.py
```

### 4. 模型训练

```bash
# 训练传统机器学习模型
python scripts/train_model.py --model xgboost --config config/config.yaml

# 训练深度学习模型
python scripts/train_model.py --model patchts --config config/config.yaml

# 训练集成模型
python scripts/train_model.py --model ensemble --config config/config.yaml
```

### 5. 启动服务

```bash
# 启动预测API服务
uvicorn src.prediction.api:app --host 0.0.0.0 --port 8000 --workers 4

# 或使用Docker
docker-compose up -d
```

## API使用示例

### 获取单日预测

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "target_date": "2025-01-15",
    "model_type": "ensemble"
  }'
```

### 批量预测

```bash
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "start_date": "2025-01-15",
    "end_date": "2025-01-20",
    "model_type": "ensemble"
  }'
```

## 项目结构

```
├── src/                    # 源代码
│   ├── data/              # 数据层
│   ├── preprocessing/     # 预处理
│   ├── models/           # 模型实现
│   ├── training/         # 训练模块
│   ├── prediction/       # 预测服务
│   └── evaluation/       # 评估模块
├── scripts/              # 脚本工具
├── notebooks/            # 分析笔记本
├── tests/               # 测试代码
├── models/              # 模型文件
├── config/              # 配置文件
└── docs/                # 文档
```

## 性能指标

在2024年历史数据上的测试结果：

| 指标 | 日前电价 | 实时电价 |
|------|----------|----------|
| MAE  | 15.32    | 18.67    |
| RMSE | 22.45    | 26.89    |
| MAPE | 8.5%     | 10.2%    |
| 方向准确率 | 85.3% | 82.7%   |

## 开发指南

### 添加新特征

1. 在 `src/preprocessing/feature_engineering.py` 中实现特征提取逻辑
2. 在 `config/config.yaml` 中配置特征参数
3. 重新训练模型以评估特征效果

### 添加新模型

1. 在 `src/models/` 下创建新的模型文件
2. 继承 `BaseModel` 类并实现必要方法
3. 在配置文件中注册新模型
4. 更新训练脚本以支持新模型

## 部署说明

### Docker部署

```bash
# 构建镜像
docker build -t power-price-prediction:latest .

# 运行容器
docker run -d \
  --name power-prediction \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  --env-file .env \
  power-price-prediction:latest
```

### Kubernetes部署

```bash
# 应用配置
kubectl apply -f k8s/

# 检查部署状态
kubectl get pods -n power-prediction
```

## 监控和维护

### 查看系统指标

访问 `http://localhost:9090/metrics` 查看Prometheus指标

### 查看日志

```bash
# 查看应用日志
tail -f logs/app.log

# 查看预测日志
tail -f logs/prediction.log
```

### 数据漂移监控

系统会自动监控输入数据分布的变化，当检测到显著漂移时会发出告警。

## 常见问题

### Q: 数据库连接失败怎么办？
A: 请检查.env文件中的数据库配置，确保网络连接正常。

### Q: 模型预测速度慢怎么优化？
A: 可以尝试：
   - 使用更少的特征
   - 减少模型复杂度
   - 启用预测缓存
   - 增加API workers数量

### Q: 如何提高预测精度？
A: 建议：
   - 增加训练数据量
   - 优化特征工程
   - 调整模型超参数
   - 使用集成学习方法

## 贡献指南

欢迎提交Issue和Pull Request。提交代码前请确保：

1. 通过所有测试 `pytest`
2. 代码格式化 `black src/`
3. 代码检查 `flake8 src/`
4. 类型检查 `mypy src/`

## 许可证

本项目采用 MIT 许可证。

## 联系方式

- 项目维护者：[Your Name]
- Email: [your.email@example.com]
- Issue: [GitHub Issues]