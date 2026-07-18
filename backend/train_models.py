import os
import glob
import time
import shutil
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, MinMaxScaler, StringIndexer
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import col, when
from pyspark import StorageLevel

# ================= 路径配置 =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(CURRENT_DIR, 'datasets')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
WAREHOUSE_DIR = os.path.join(PROJECT_ROOT, 'spark-warehouse')

DATASET_META = {
    'shop_shipping': 'Reached_on_Time_Y_N', 
    'shop_churn':    'Churn',
    'shop_fraud':    'Class',
    'shop_order':    'ordered',
    'trans_satisfaction': 'satisfaction',
    'trans_delay':        'DEP_DEL15',
    'trans_accident':     'Severity',
    'ind_failure':   'Machine_failure',
    'ind_quality':   'Quality' 
}

def get_spark_session():
    builder = SparkSession.builder \
        .appName("SmartTrainer_FullPipeline") \
        .master("local[*]") \
        .config("spark.driver.memory", "8g") \
        .config("spark.sql.warehouse.dir", WAREHOUSE_DIR) \
        .config("spark.ui.enabled", "false")
    
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def clean_column_names(df):
    new_cols = [c.strip().replace('.', '_').replace(' ', '_') for c in df.columns]
    return df.toDF(*new_cols)

def custom_preprocessing(df, task_name):
    if task_name == 'trans_accident':
        print("   🧹 [特征工程] 过滤极少量的 Severity 1 和 4...")
        df = df.filter(col("Severity").isin([2, 3]))
    elif task_name == 'shop_shipping':
        print("   🧪 [特征工程] 提取 High_Discount 和 Is_Heavy 特征...")
        df = df.withColumn("High_Discount", when(col("Discount_offered") > 10, 1).otherwise(0))
        df = df.withColumn("Is_Heavy", when(col("Weight_in_gms") > 4000, 1).otherwise(0))
    return df

def train_one_file(spark, file_path):
    filename = os.path.basename(file_path)
    task_name = os.path.splitext(filename)[0]
    
    if task_name.endswith('_clean'): task_name = task_name.replace('_clean', '')
    if task_name not in DATASET_META:
        print(f"⚠️ 跳过未配置场景文件: {filename}")
        return

    print(f"\n" + "="*60)
    print(f"🔄 正在处理场景: [{task_name}]")
    
    try:
        # 读取数据
        sample_df = spark.read.option("header", "true").option("inferSchema", "true").csv(file_path).limit(1000)
        schema = sample_df.schema
        df = spark.read.option("header", "true").schema(schema).csv(file_path)
        
        df = clean_column_names(df)
        df = df.fillna(0).fillna("Unknown")
        df = custom_preprocessing(df, task_name)
        
        target_col = DATASET_META.get(task_name)
        if target_col: target_col = target_col.replace('.', '_').replace(' ', '_')
        
        if not target_col or target_col not in df.columns:
            print(f"❌ 错误：找不到目标列 {target_col}")
            return

        print(f"   🎯 锁定标签列: {target_col}")
        
        # === 1. 构建特征预处理流水线 ===
        feature_cols = [c for c in df.columns if c != target_col and 
                        c.lower() not in ['id', 'user_id', 'order_id', 'Unnamed: 0', '_c0']]
        
        stages = []
        # 标签索引化
        stages.append(StringIndexer(inputCol=target_col, outputCol="label").setHandleInvalid("skip"))
        
        # 特征索引化 + 向量化
        assembler_inputs = []
        for col_name in feature_cols:
            dtype = [d for n, d in df.dtypes if n == col_name][0]
            if dtype == 'string':
                idx_name = f"{col_name}_idx"
                # 关键：保存StringIndexer模型，否则预测时无法处理字符串
                stages.append(StringIndexer(inputCol=col_name, outputCol=idx_name).setHandleInvalid("keep"))
                assembler_inputs.append(idx_name)
            else:
                assembler_inputs.append(col_name)
        
        stages.append(VectorAssembler(inputCols=assembler_inputs, outputCol="raw_features"))
        stages.append(MinMaxScaler(inputCol="raw_features", outputCol="features"))
        
        # 训练预处理模型
        preprocessing_pipeline = Pipeline(stages=stages)
        preprocessing_model = preprocessing_pipeline.fit(df)
        final_df = preprocessing_model.transform(df).select("features", "label")
        
        final_df.persist(StorageLevel.MEMORY_AND_DISK)
        
        num_classes = final_df.select("label").distinct().count()
        train_data, test_data = final_df.randomSplit([0.8, 0.2], seed=42)

        # 智能过采样
        label_counts = train_data.groupBy("label").count().collect()
        counts = {row['label']: row['count'] for row in label_counts}
        if len(counts) > 1:
            max_count = max(counts.values())
            min_count = min(counts.values())
            if max_count / min_count > 2:
                print(f"   ⚖️ 执行过采样 (比例 {max_count/min_count:.1f}:1)...")
                dfs = []
                for label, count_val in counts.items():
                    label_df = train_data.filter(col("label") == label)
                    if count_val < max_count:
                        limit = 50.0 if task_name == 'ind_quality' else 10.0
                        ratio = min(max_count / count_val, limit)
                        label_df = label_df.sample(withReplacement=True, fraction=ratio, seed=42)
                    dfs.append(label_df)
                train_data = dfs[0]
                for d in dfs[1:]: train_data = train_data.union(d)
                train_data = train_data.repartition(spark.sparkContext.defaultParallelism)

        # 定义分类器
        rf = RandomForestClassifier(labelCol="label", featuresCol="features", seed=42, maxBins=128)
        gbt = GBTClassifier(labelCol="label", featuresCol="features", seed=42, maxBins=128)
        nb = NaiveBayes(labelCol="label", featuresCol="features")

        classifiers = {'random_forest': rf, 'gbdt': gbt, 'naive_bayes': nb}
        if num_classes > 2: del classifiers['gbdt']

        # 使用 F1 评估
        evaluator = MulticlassClassificationEvaluator(metricName="f1")
        rf_grid = ParamGridBuilder().addGrid(rf.maxDepth, [10]).addGrid(rf.numTrees, [40]).build()
        
        if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)

        for algo_name, clf in classifiers.items():
            save_path = os.path.join(MODELS_DIR, f"{task_name}_{algo_name}.model")
            if os.path.exists(save_path): shutil.rmtree(save_path)

            print(f"   🚀 正在训练: {algo_name} ...")
            start = time.time()
            
            # 训练分类模型
            if algo_name == 'random_forest':
                cv = CrossValidator(estimator=clf, estimatorParamMaps=rf_grid, evaluator=evaluator, numFolds=3)
                best_clf_model = cv.fit(train_data).bestModel
            else:
                best_clf_model = clf.fit(train_data)

            # 评估
            preds = best_clf_model.transform(test_data)
            f1 = evaluator.evaluate(preds)
            acc = MulticlassClassificationEvaluator(metricName="accuracy").evaluate(preds)
            
            # === 核心：拼接并保存完整 PipelineModel ===
            # full_stages = [预处理的所有stage] + [分类模型]
            full_stages = preprocessing_model.stages + [best_clf_model]
            full_pipeline_model = PipelineModel(stages=full_stages)
            
            full_pipeline_model.write().overwrite().save(save_path)
            print(f"     ✅ {algo_name} F1: {f1*100:.2f}% (Acc: {acc*100:.2f}%) - 已保存完整流水线")

        final_df.unpersist()

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    spark = get_spark_session()
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    for f in csv_files:
        train_one_file(spark, f)
    spark.stop()