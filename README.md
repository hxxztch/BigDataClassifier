# Spark 大数据快速分类系统

基于 PySpark + Flask 的大数据分类预测平台，支持多种机器学习模型（随机森林、GBDT、XGBoost、朴素贝叶斯）。

## 环境要求

- Python 3.8+
- Java JDK 8 / 11 / 17 / 21（PySpark 依赖）
- Windows / Linux / macOS

## 快速开始

`ash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动后端
python backend/app.py

# 3. 打开浏览器访问
http://localhost:5000
`

## 目录说明

- ackend/ — Flask 后端、Spark 处理、模型训练
- rontend/ — 前端单页应用
- models/ — 训练好的模型文件
- ackend/datasets/ — 训练数据集
- data/ — 上传文件、缓存

## 使用流程

1. 系统管理 → 场景管理 → 添加场景
2. 系统管理 → 模型训练 → 选择数据集和场景训练
3. 分类预测 → 上传 CSV → 数据质检 → 开始分析

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| JAVA_HOME | JDK 路径 | 自动检测 |
| SPARK_MASTER_URL | Spark 运行模式 | local[*] |
