# Spark Big Data Classifier

基于 PySpark 的分布式智能分类系统，支持电商、交通、工业三大领域 10 种业务场景的分类预测。

## 系统要求

| 组件 | 版本 |
|------|------|
| Python | 3.8 - 3.11 |
| JDK | 11 / 17 / 21（自动检测） |
| 内存 | 建议 8GB+ |
| GPU（可选） | NVIDIA + CUDA 11+ |

## 一键启动

```bash
python launch.py
```

自动检测 Java → 安装依赖 → 训练模型 → 启动服务 → 打开浏览器。

## 手动安装

```bash
# 1. 安装 Python 依赖
pip install -r requirements.txt

# 2. 训练模型
cd backend
python train_models.py

# 3. 启动服务
python app.py

# 4. 打开浏览器访问
http://localhost:5000
```

## 使用流程

1. 上传 CSV 数据文件
2. 选择业务场景（电商/交通/工业）
3. 选择算法（RF / GBDT / XGBoost / NaiveBayes）
4. 自动择优（Auto）：系统自动对比四种算法，选择 F1 最高的
5. 查看预测结果（混淆矩阵、特征重要性、分布图）

## 场景支持

| 领域 | 场景 | 目标 |
|------|------|------|
| 电商 | 用户流失预警 | Churn |
| 电商 | 交易欺诈检测 | Class |
| 电商 | 用户下单预测 | ordered |
| 交通 | 乘客满意度 | satisfaction |
| 交通 | 航班延误预测 | DEP_DEL15 |
| 交通 | 交通事故定责 | Severity |
| 工业 | 设备故障诊断 | Machine_failure |
| 工业 | 产品质量控制 | Quality |

## GPU 加速

系统自动检测 GPU：
- 有 GPU：XGBoost 训练/推理 + Rapids 流水线加速
- 无 GPU：静默降级到纯 CPU 执行
- 强制关闭 GPU：`set DISABLE_RAPIDS=true`

## 项目结构

```
BigDataClassifier/
  launch.py              # 一键启动脚本
  requirements.txt       # Python 依赖
  backend/
    app.py               # Flask API 服务
    train_models.py      # 模型训练
    spark_utils.py       # Spark 预测引擎
    database.py          # SQLite 任务管理
    utils/               # 工具包
      config.py          # 全局配置
      preprocessing.py   # 数据预处理
      logger.py          # 日志
    datasets/            # 示例数据
    scripts/             # 辅助工具
  frontend/
    index.html           # 前端页面
  models/                # 训练好的模型
  jars/                  # GPU 加速 jar
```

## 性能参考

| 场景 | RF | GBDT | XGBoost | NB |
|------|------|------|------|------|
| trans_delay | ~71% | ~87% | ~87% | ~57% |
| trans_satisfaction | ~94% | ~94% | ~94% | ~82% |

## 许可证

MIT License
