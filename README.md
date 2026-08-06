# Spark 大数据快速分类系统

基于 PySpark + Flask 的分布式分类预测平台，支持 Spark MLlib 全量分布式训练、Delta Lake 数据湖、MLflow 实验追踪。

## 技术架构

### 核心栈
| 层级 | 技术 | 说明 |
|------|------|------|
| 大数据引擎 | Apache Spark 3.3.0 (PySpark) | Standalone 集群 + local 双模式 |
| 后端框架 | Flask + Gunicorn | REST API + 异步任务 |
| 前端 | Vue.js 2 + Element UI | SPA 单页应用 |
| 可视化 | ECharts 5 | 饼图/混淆矩阵/特征重要性 |
| 机器学习 | Spark MLlib + XGBoost + sklearn | 双路径训练 |
| 数据湖 | Delta Lake 2.2.0 | ACID 事务 + Time Travel + Z-Order |
| 实验追踪 | MLflow 2.17 | 参数/指标/模型自动记录 |
| 数据存储 | SQLite + YAML + Delta | 任务记录 + 场景配置 + 数据缓存 |
| 环境 | Python 3.10, JDK 17 | 已验证稳定运行 |

### 大数据特性
- **Spark SQL 分析**: CTE + CASE WHEN 实现列统计，输出 Catalyst 执行计划
- **AQE 自适应执行**: 动态分区合并 + 倾斜 Join 优化
- **数据倾斜检测**: 分区级 max/avg ratio 分析 + 加盐打散
- **Delta Lake**: 数据缓存 ACID 事务 + DESCRIBE HISTORY 版本追溯 + OPTIMIZE Z-Order

### 机器学习特性
- **双路径训练**: Spark MLlib (RF/GBT/NB/XGBoost) 全量分布式 + sklearn 单机回退
- **四模型自动择优**: Auto 模式自动选择最优模型
- **Platt 校准 + 最优阈值**: sklearn 模型概率校准
- **版本管理**: 模型版本切换 + 元数据追踪

## 项目结构

```
BigDataClassifier/
├── backend/
│   ├── app.py                     # Flask 主应用 + API
│   ├── admin_routes.py            # 系统管理 (场景/训练/版本)
│   ├── spark_utils.py             # Spark 分类预测引擎
│   ├── train_sklearn.py           # 双路径训练 (MLlib + sklearn)
│   ├── train_worker.py            # 训练子进程入口
│   ├── test_training.py           # 自动化测试 (16 项)
│   ├── database.py                # SQLite 任务记录
│   ├── scenes.yaml                # 场景配置
│   ├── datasets/                  # 训练数据集
│   ├── utils/
│   │   ├── config.py              # Spark 配置 + AQE + Delta + MLflow
│   │   ├── preprocessing.py       # 特征工程流水线
│   │   ├── data_quality.py        # 数据质量检测 + SQL 倾斜分析
│   │   ├── version_manager.py     # 模型版本管理
│   │   ├── spark_sql_utils.py     # Spark SQL 分析 + 倾斜检测 + 加盐
│   │   ├── delta_utils.py         # Delta Lake 操作 (ACID/Time Travel)
│   │   └── mlflow_utils.py        # MLflow 实验追踪
│   └── mlruns/                    # MLflow 实验数据
├── frontend/
│   └── index.html                 # 前端单页应用
├── cluster/
│   ├── start-cluster.bat          # 启动 1 Master + 3 Workers
│   ├── stop-cluster.ps1           # 停止集群
│   ├── status-cluster.ps1         # 集群状态
│   └── run-app-cluster.ps1        # 集群模式启动 Flask
├── models/                        # 训练好的模型
├── data/                          # 上传文件 + 缓存
└── requirements.txt
```

## 环境搭建

### 依赖
- Python 3.10+
- JDK 17 (必需，JDK 21 不兼容)
- Spark 3.3.0 (standalone 版本在 E:\\bigdata\\spark-3.3.0-bin-hadoop3)

### 安装

```bash
pip install -r requirements.txt
```

关键依赖: pyspark, scikit-learn, xgboost, delta-spark, mlflow, flask, pyyaml

### 配置 JDK

系统需安装 JDK 17。config.py 会自动设置 JAVA_HOME，也可手动:

```powershell
$env:JAVA_HOME = "C:\Program Files\Java\jdk-17.0.20"
```

## 快速开始

### 1. 本地模式启动

```bash
cd backend
python app.py
# 访问 http://localhost:5000
```

### 2. 集群模式启动

```bash
# 启动 Spark 集群
cd cluster
.\start-cluster.bat
# 等待 20 秒，确认 http://localhost:9090 显示 3 Workers ALIVE

# 集群模式启动 Flask
$env:SPARK_MASTER_URL = "spark://127.0.0.1:7077"
cd ..\backend
python app.py
```

### 3. 自动化测试

```bash
cd backend
python test_training.py           # 本地模式 16 项测试
python test_training.py --cluster # 集群模式测试
```

## 模型训练

### 命令行训练

```bash
cd backend
python -c "from train_sklearn import train_sklearn; train_sklearn('datasets/trans_delay.csv', scene_id='trans_delay')"
```

训练流程:
1. **Phase 1: Spark MLlib** — RF/GBT/NB/XGBoost 全量分布式训练 (60 万行)
2. **Phase 2: sklearn** — XGBoost 单机训练 (3 万行采样) + Platt 校准

预测时 `_load_model` 优先加载 MLlib PipelineModel。

### 训练结果对比 (trans_delay 场景)

| 模型 | MLlib F1 | MLlib Acc | sklearn F1 | sklearn Acc |
|------|----------|-----------|------------|-------------|
| XGBoost | 93.59% | 94.16% | 87.60% | 89.86% |
| GBT | 92.37% | 93.19% | 84.54% | 88.20% |
| RF | 83.23% | 87.80% | 79.60% | 86.05% |
| NB | 80.26% | 86.52% | 79.43% | 84.42% |

## MLflow 实验追踪

每次训练自动记录模型参数、指标、标签到 MLflow。

### 查看实验记录

前端: 系统管理 → 场景管理 → 点击「MLflow」按钮

或直接访问 API: http://localhost:5000/api/mlflow/runs

### 启动 MLflow UI

```bash
cd E:\Study\Spark大数据快速分类\BigDataClassifier
mlflow ui --port 5001
# 访问 http://localhost:5001
```

## Delta Lake 数据湖

数据质量检测后自动以 Delta 格式缓存:
- ACID 事务写入
- DESCRIBE HISTORY 查看数据版本
- VACUUM 清理旧文件
- OPTIMIZE + ZORDER 聚簇优化

## 许可证

MIT
