# Makefile for 电力市场预测系统

.PHONY: help install test clean lint format train predict check docker-build docker-up docker-down

# 默认目标
help:
	@echo "可用命令:"
	@echo "  make install     - 安装项目依赖"
	@echo "  make test        - 运行测试"
	@echo "  make lint        - 代码检查"
	@echo "  make format      - 代码格式化"
	@echo "  make train       - 训练模型"
	@echo "  make predict     - 运行预测"
	@echo "  make check       - 数据质量检查"
	@echo "  make clean       - 清理临时文件"
	@echo "  make docker-build - 构建Docker镜像"
	@echo "  make docker-up    - 启动Docker服务"
	@echo "  make docker-down  - 停止Docker服务"

# 安装依赖
install:
	pip install -r requirements.txt
	pip install -e .

# 运行测试
test:
	pytest tests/ -v --cov=src --cov-report=html

# 代码检查
lint:
	flake8 src/ --max-line-length=120 --ignore=E501,W503
	mypy src/ --ignore-missing-imports

# 代码格式化
format:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

# 训练模型
train:
	python scripts/train_model.py --model xgboost --config config/config.yaml

# 运行预测
predict:
	python scripts/predict.py --date 2025-01-15 --model ensemble

# 数据质量检查
check:
	python scripts/data_quality_check.py --start-date 2024-01-01 --end-date 2024-12-31

# 清理临时文件
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info
	rm -rf htmlcov/ .coverage
	rm -rf logs/*.log

# Docker相关
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f prediction-api

# 开发环境设置
dev-setup: install
	pre-commit install
	mkdir -p logs models data

# 生产环境部署
deploy:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 数据库迁移（如果需要）
migrate:
	python scripts/migrate_db.py

# 备份模型
backup-models:
	tar -czf models_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz models/

# 性能测试
benchmark:
	python scripts/benchmark.py