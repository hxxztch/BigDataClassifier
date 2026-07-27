
# Spark 大数据快速分类系统

基于 PySpark + Flask 的大数据分类预测平台，支持多种机器学习模型。

## 界面预览

| 首页 | 配置参数 |
|------|----------|
| ![首页](screenshots/homepage.png) | ![配置参数](screenshots/Configuration.png) |

| 数据质检 |
|----------|
| ![数据质检](screenshots/DataInspector.png) |

| 分类结果 |
|----------|
| ![结果1](screenshots/result_1.png) |
| ![结果2](screenshots/result_2.png) |

| 评估指标 |
|----------|
| ![评估指标](screenshots/Evaluation.png) |

| 场景管理 | 模型训练 |
|----------|----------|
| ![场景管理](screenshots/scene_management.png) | ![模型训练](screenshots/model_train.png) |

| 历史记录 |
|----------|
| ![历史记录](screenshots/record.png) |

## 搭建环境

- Python 3.10.2
- Java 21
- Windows

## 快速开始

`Bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动后端
python backend/app.py

# 3. 打开浏览器访问
http://localhost:5000
`

## 目录说明

| 目录 | 说明 |
|------|------|
| backend/ | Flask 后端、Spark 处理、模型训练 |
| frontend/ | 前端单页应用 |
| models/ | 训练好的模型文件 |
| backend/datasets/ | 训练数据集 |
| data/ | 上传文件、缓存 |
| screenshots/ | 界面截图 |

## 使用流程

1. **添加场景**：系统管理 → 场景管理 → 添加场景（填写 ID、名称、目标列、分类）
2. **训练模型**：系统管理 → 模型训练 → 选择数据集和场景 → 开始训练
3. **数据预测**：分类预测 → 上传 CSV → 选择场景和模型 → 数据质检 → 开始分析

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| JAVA_HOME | JDK 路径 | 自动检测 |
| SPARK_MASTER_URL | Spark 运行模式 | local[*] |