import os
import glob
import time
import shutil
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, MinMaxScaler, StringIndexer
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import col
from pyspark import StorageLevel

# ================= 路径配置 =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(CURRENT_DIR, 'datasets')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
WAREHOUSE_DIR = os.path.join(PROJECT_ROOT, 'spark-warehouse')

# === ！！！ 训练时完全移除 GPU 加速 ！！！ ===
RAPIDS_JAR_PATH = None 

# ============================================
# 数据集标签列配置
# ============================================
DATASET_META = {
    'shop_shipping': 'Reached_on_Time_Y_N', 
    'shop_churn':    'Churn',
    'shop_fraud':    'Class',
    'shop_order':    'ordered',
    'trans_satisfaction': 'satisfaction',
    'trans_delay':        'DepDel15',
    'trans_accident':     'Severity',
    'ind_failure':   'Machine_failure',
    'ind_quality':   'Quality',
    'ind_safety':    'Accident_Level'
}

def get_spark_session():
    # 针对 GB 级数据优化内存配置
    builder = SparkSession.builder \
        .appName("SmartTrainer_Pro") \
        .master("local[*]") \
        .config("spark.driver.memory", "16g") \
        .config("spark.executor.memory", "16g") \
        .config("spark.driver.maxResultSize", "4g") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.memory.offHeap.size", "4g") \
        .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
        .config("spark.sql.warehouse.dir", WAREHOUSE_DIR) \
        .config("spark.ui.enabled", "false") 
    
    print("✅ Spark Session 配置已优化 (High Memory for GB-scale data).")
    return builder.getOrCreate()

def clean_column_names(df):
    new_cols = [c.strip().replace('.', '_').replace(' ', '_') for c in df.columns]
    return df.toDF(*new_cols)

def train_one_file(spark, file_path):
    filename = os.path.basename(file_path)
    task_name = os.path.splitext(filename)[0]
    
    print(f"\n" + "="*60)
    print(f"🔄 正在处理场景: [{task_name}]")
    print(f"📄 数据源: {filename}")
    
    try:
        # === 优化 1: 快速 Schema 推断 ===
        # 对于 GB 级 CSV，inferSchema=True 会扫描全表，非常慢。
        # 这里只读前 1000 行推断结构，然后应用到大文件。
        print("   ⏳ 正在采样推断 Schema...")
        sample_df = spark.read.option("header", "true") \
                        .option("inferSchema", "true") \
                        .csv(file_path) \
                        .limit(1000)
        target_schema = sample_df.schema
        
        print("   📥 正在读取全量数据...")
        df = spark.read.option("header", "true") \
                  .schema(target_schema) \
                  .csv(file_path)

        df = clean_column_names(df)
        df = df.fillna(0).fillna("Unknown")
        
        # 确定目标列
        target_col = DATASET_META.get(task_name)
        if target_col:
            target_col = target_col.replace('.', '_').replace(' ', '_')
        
        if not target_col or target_col not in df.columns:
            cols_no_id = [c for c in df.columns if 'id' not in c.lower()]
            if cols_no_id:
                target_col = cols_no_id[-1]
            else:
                print("❌ 错误：无法确定有效列。")
                return

        print(f"   🎯 锁定标签列: {target_col}")
        
        # 特征选择
        feature_cols = []
        for c in df.columns:
            lower_c = c.lower()
            if (lower_c == 'id' or lower_c.endswith('_id') or 'user_id' in lower_c or 
                'session_id' in lower_c or 'transaction_id' in lower_c or 
                'order_id' in lower_c or 'customer_id' in lower_c):
                continue
            if c == target_col: continue
            if 'unnamed' in lower_c or '_c0' in lower_c: continue
            feature_cols.append(c)
        
        print(f"   - 特征数量: {len(feature_cols)}")

        # 构建预处理 Pipeline
        stages = []
        label_indexer = StringIndexer(inputCol=target_col, outputCol="label").setHandleInvalid("skip")
        stages.append(label_indexer)
        
        assembler_inputs = []
        for col_name in feature_cols:
            dtype = [d for n, d in df.dtypes if n == col_name][0]
            if dtype == 'string':
                idx_name = f"{col_name}_idx"
                stages.append(StringIndexer(inputCol=col_name, outputCol=idx_name).setHandleInvalid("keep"))
                assembler_inputs.append(idx_name)
            else:
                assembler_inputs.append(col_name)
        
        stages.append(VectorAssembler(inputCols=assembler_inputs, outputCol="raw_features"))
        stages.append(MinMaxScaler(inputCol="raw_features", outputCol="features"))

        pipeline = Pipeline(stages=stages)
        print("   🔨 执行特征工程 Pipeline...")
        model_pipeline = pipeline.fit(df)
        final_df = model_pipeline.transform(df).select("features", "label")
        
        # === 优化 2: 缓存数据 ===
        # 交叉验证会多次重复使用数据，必须缓存到内存/磁盘
        final_df.persist(StorageLevel.MEMORY_AND_DISK)
        
        num_classes = final_df.select("label").distinct().count()
        print(f"   📊 数据集类别数量: {num_classes}")

        train_data, test_data = final_df.randomSplit([0.8, 0.2], seed=42)

        # === 样本不平衡处理 (保持原有逻辑) ===
        if num_classes == 2:
            label_counts = train_data.groupBy("label").count().collect()
            if len(label_counts) == 2:
                c0 = next((c['count'] for c in label_counts if c.label == 0), 0)
                c1 = next((c['count'] for c in label_counts if c.label == 1), 0)

                if c0 > 0 and c1 > 0 and (c0/c1 > 3 or c1/c0 > 3):
                    minority_label = 0 if c0 < c1 else 1
                    majority_label = 1 if c0 < c1 else 0
                    
                    minority_df = train_data.filter(col("label") == minority_label)
                    majority_df = train_data.filter(col("label") == majority_label)
                    
                    ratio = majority_df.count() / minority_df.count()
                    oversampled = minority_df.sample(withReplacement=True, fraction=ratio, seed=42)
                    
                    train_data = majority_df.unionAll(oversampled).repartition(spark.sparkContext.defaultParallelism)
                    print(f"     ⚖️ 已执行过采样平衡数据 (比例 {ratio:.2f})")

        # === 优化 3: 引入 GridSearch 和 CrossValidator 以提升准确率 ===
        # 定义基础分类器
        rf = RandomForestClassifier(labelCol="label", featuresCol="features", seed=42)
        gbt = GBTClassifier(labelCol="label", featuresCol="features", seed=42)
        nb = NaiveBayes(labelCol="label", featuresCol="features")

        # 定义参数网格 (为 GB 级数据寻找更优参数)
        # 注意: 参数越多训练越慢，这里选择了性价比高的参数组合
        rf_paramGrid = ParamGridBuilder() \
            .addGrid(rf.numTrees, [50, 100]) \
            .addGrid(rf.maxDepth, [5, 10]) \
            .build()
            
        gbt_paramGrid = ParamGridBuilder() \
            .addGrid(gbt.maxIter, [20, 50]) \
            .addGrid(gbt.maxDepth, [5]) \
            .build()
            
        nb_paramGrid = ParamGridBuilder().build() # 贝叶斯通常不需要复杂调优

        classifiers_config = {
            'random_forest': (rf, rf_paramGrid),
            'gbdt': (gbt, gbt_paramGrid),
            'naive_bayes': (nb, nb_paramGrid)
        }

        if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)

        for algo_name, (clf, param_grid) in classifiers_config.items():
            model_save_name = f"{task_name}_{algo_name}.model"
            save_path = os.path.join(MODELS_DIR, model_save_name)
            
            # GBDT 限制检查
            if algo_name == 'gbdt' and num_classes > 2:
                 print(f"     ⚠️  跳过 GBDT: 数据为多分类({num_classes}类)，GBDT仅支持二分类。")
                 continue

            if os.path.exists(save_path):
                try: shutil.rmtree(save_path)
                except: pass

            print(f"   🚀 正在训练 (交叉验证): {algo_name} ...")
            
            try:
                start_time = time.time()
                
                # 使用交叉验证 (3折)
                evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
                cv = CrossValidator(estimator=clf,
                                    estimatorParamMaps=param_grid,
                                    evaluator=evaluator,
                                    numFolds=3) # 3折能较好平衡速度和准确率

                cv_model = cv.fit(train_data)
                best_model = cv_model.bestModel
                
                # 在测试集上验证最终准确率
                predictions = best_model.transform(test_data)
                acc = evaluator.evaluate(predictions)
                
                # 保存最佳模型
                best_model.write().overwrite().save(save_path)
                
                duration = time.time() - start_time
                print(f"     ✅ 成功! 最佳准确率: {acc*100:.2f}% (耗时: {duration:.1f}s)")
                
            except Exception as e:
                print(f"     ❌ 训练失败: {e}")
                import traceback
                traceback.print_exc()

        # 清理缓存
        final_df.unpersist()

    except Exception as e:
        print(f"❌ 文件处理错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"❌ 错误: {DATA_DIR} 为空。")
        exit()

    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"🔎 发现 {len(csv_files)} 个数据集，开始批量高精度训练...")
    
    spark = get_spark_session()
    
    for csv_file in csv_files:
        train_one_file(spark, csv_file)
    
    spark.stop()
    print("\n" + "="*50)
    print("🎉 所有模型训练优化完毕！请重启 app.py")