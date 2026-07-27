
# Spark 大数据快速分类系统

基于 PySpark + Flask 的大数据分类预测平台，支持多种机器学习模型。

## 界面预览
## 项目简介

Spark 大数据快速分类系统是一个基于 PySpark 分布式计算框架的企业级分类预测平台。系统以 Flask 为 Web 服务、Vue.js 为前端，集成随机森林、GBDT、XGBoost、朴素贝叶斯四种机器学习算法，提供从数据上传、质量检测、模型训练、版本管理到分类预测的完整闭环。支持场景化管理，可灵活配置不同业务场景的模型参数和特征列，适用于用户流失预测、交易欺诈检测、航班延误分析、设备故障诊断等多种大数据分类场景。

## 核心功能

- **智能分类预测**：上传 CSV 数据，自动择优或手动选择模型，一键完成分布式预测分析
- **数据质量检测**：自动校验数据格式与场景匹配度，生成列级缺失率、数据类型、值范围报告
- **多模型训练**：一键训练随机森林、GBDT、XGBoost、朴素贝叶斯四种模型，自动保存最优版本
- **场景化管理**：支持多业务场景的独立配置，包括目标列、特征列、分类标签等
- **版本切换**：模型训练后自动保存版本，可在不同版本间灵活切换
- **可视化结果**：饼图、混淆矩阵、特征重要性、条形图等多维度结果展示
- **历史回溯**：分页查询历史预测任务，支持按场景筛选和结果查看

## 功能模块

| 模块 | 说明 |
|------|------|
| 分类预测 | 上传数据、选择场景和模型、数据质检、查看分析结果 |
| 历史记录 | 翻页浏览历史任务，支持场景筛选 |
| 系统管理 | 场景管理与分类配置、模型训练与版本切换 |

## 技术栈

| 层级 | 技术 |
|------|------|
| 大数据引擎 | Apache Spark (PySpark) |
| 后端框架 | Flask + Gunicorn |
| 前端框架 | Vue.js 2 + Element UI |
| 可视化 | ECharts 5 |
| 机器学习 | Scikit-learn + XGBoost |
| 数据存储 | SQLite (任务记录) + YAML (场景配置) + Parquet (缓存) |
| 环境 | Python 3.10、Java JDK 21 |

## 项目结构

`
BigDataClassifier/
├── backend/                  # 后端服务
│   ├── app.py                # Flask 主应用
│   ├── admin_routes.py       # 系统管理、场景管理、训练 API
│   ├── spark_utils.py        # Spark 分类预测引擎
│   ├── train_sklearn.py      # sklearn 模型训练
│   ├── train_worker.py       # 训练子进程入口
│   ├── database.py           # SQLite 数据库操作
│   ├── scenes.yaml           # 场景配置文件
│   ├── datasets/             # 训练数据集
│   └── utils/                # 工具模块
│       ├── config.py         # 全局配置、Spark 初始化
│       ├── preprocessing.py  # 数据预处理与特征工程
│       ├── data_quality.py   # 数据质量检测
│       └── version_manager.py# 模型版本管理
├── frontend/
│   └── index.html            # 前端单页应用
├── models/                   # 训练好的模型文件
├── data/                     # 上传文件与缓存
│   ├── uploads/
│   └── cache/
├── jars/                     # Spark 扩展 JAR
├── screenshots/              # 系统截图
├── requirements.txt          # Python 依赖
└── README.md
`


### 首页
![首页](screenshots/homepage.png)

### 配置参数
![配置参数](screenshots/Configuration.png)

### 数据质检
![数据质检](screenshots/DataInspector.png)

### 分类结果
![结果1](screenshots/result_1.png)
![结果2](screenshots/result_2.png)

### 评估指标
![评估指标](screenshots/Evaluation.png)

### 场景管理
![场景管理](screenshots/scene_management.png)

### 模型训练
![模型训练](screenshots/model_train.png)

### 历史记录
![历史记录](screenshots/record.png)

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
