import os
import shutil
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel, GBTClassificationModel, NaiveBayesModel
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler, StringIndexer, MinMaxScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col, when, lit

from database import update_task_progress

class SparkClassifier:
    def __init__(self):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.current_dir)
        self.models_dir = os.path.join(self.project_root, 'models')
        self.warehouse_dir = os.path.join(self.project_root, 'spark-warehouse')
        
        # 内存优化配置
        builder = SparkSession.builder \
            .appName("SparkPredictor_FullLabels") \
            .master("local[*]") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.sql.warehouse.dir", self.warehouse_dir) \
            .config("spark.ui.enabled", "false") \
            .config("spark.rpc.message.maxSize", "256")
            
        self.spark = builder.getOrCreate()
        self.spark.sparkContext.setLogLevel("ERROR")

        # === 1. 数据集目标列映射 ===
        self.DATASET_META = {
            'shop_churn':    'Churn',
            'shop_fraud':    'Class',
            'shop_order':    'ordered',
            'trans_satisfaction': 'satisfaction',
            'trans_delay':        'DEP_DEL15',
            'trans_accident':     'Severity',
            'ind_failure':   'Machine_failure',
            'ind_quality':   'Quality' 
        }

        # === 2. 标签含义映射 (现在会完整显示) ===
        self.SCENE_LABEL_MAP = {
            'shop_churn':         {0: '😊 用户留存', 1: '😭 用户流失'},
            'shop_fraud':         {0: '🟢 正常交易', 1: '🔴 欺诈交易'},
            'shop_order':         {0: '👀 仅浏览',   1: '🛒 下单购买'},
            'trans_satisfaction': {0: '😐 中立/不满', 1: '😄 满意'},
            'trans_delay':        {0: '✈️ 准点起飞',   1: '⏳ 航班延误'},
            'trans_accident':     {
                0: '一般', 1: '严重', 
                2: '一般', 3: '严重'
            },
            'ind_failure':        {0: '⚙️ 设备完好', 1: '🔥 设备故障'},
            'ind_quality':        {0: '✨ 质量合格', 1: '🔧 存在缺陷'}
        }
        
        # 动态生成 ind_quality 特征 (0-589)
        ind_quality_drop_cols = {'157', '158', '220', '245', '292', '293', '358', '492', '517', '85', 'Time'}
        ind_quality_features = [str(i) for i in range(590) if str(i) not in ind_quality_drop_cols]

        # === 3. 场景特征列精准白名单 ===
        self.SCENE_REQUIRED_COLS = {
            'shop_churn': [
                'Churn', 'Tenure', 'PreferredLoginDevice', 'CityTier', 'WarehouseToHome', 'PreferredPaymentMode', 
                'Gender', 'HourSpendOnApp', 'NumberOfDeviceRegistered', 'PreferedOrderCat', 'SatisfactionScore', 
                'MaritalStatus', 'NumberOfAddress', 'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 
                'OrderCount', 'DaySinceLastOrder', 'CashbackAmount'
            ],
            'shop_fraud': [
                'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 
                'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 
                'V28', 'Amount', 'Class'
            ],
            'shop_order': [
                'basket_icon_click', 'basket_add_list', 'basket_add_detail', 'sort_by', 'image_picker', 
                'account_page_click', 'promo_banner_click', 'detail_wishlist_add', 'list_size_dropdown', 
                'closed_minibasket_click', 'checked_delivery_detail', 'checked_returns_detail', 'sign_in', 
                'saw_checkout', 'saw_sizecharts', 'saw_delivery', 'saw_account_upgrade', 'saw_homepage', 
                'device_mobile', 'device_computer', 'device_tablet', 'returning_user', 'loc_uk', 'ordered'
            ],
            'trans_satisfaction': [
                'Gender', 'Customer_Type', 'Age', 'Type_of_Travel', 'Class', 'Flight_Distance', 
                'Inflight_wifi_service', 'Departure/Arrival_time_convenient', 'Ease_of_Online_booking', 
                'Gate_location', 'Food_and_drink', 'Online_boarding', 'Seat_comfort', 'Inflight_entertainment', 
                'On-board_service', 'Leg_room_service', 'Baggage_handling', 'Checkin_service', 'Inflight_service', 
                'Cleanliness', 'Departure_Delay_in_Minutes', 'Arrival_Delay_in_Minutes', 'satisfaction'
            ],
            'trans_delay': [
                'DAY_OF_MONTH', 'DAY_OF_WEEK', 'OP_UNIQUE_CARRIER', 'OP_CARRIER', 'ORIGIN_AIRPORT_ID', 
                'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN', 'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID', 'DEST', 
                'DEP_TIME', 'DEP_TIME_BLK', 'DISTANCE', 'DEP_DEL15'
            ],
            'trans_accident': [
                'Source', 'Start_Lat', 'Start_Lng', 'End_Lat', 'End_Lng', 'Distance(mi)', 'County', 'State', 
                'Timezone', 'Airport_Code', 'Weather_Timestamp', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 
                'Pressure(in)', 'Visibility(mi)', 'Wind_Direction', 'Wind_Speed(mph)', 'Precipitation(in)', 
                'Weather_Condition', 'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 
                'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop', 
                'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight', 'Severity'
            ],
            'ind_failure': [
                'Type', 'Air_temperature_[K]', 'Process_temperature_[K]', 'Rotational_speed_[rpm]', 
                'Torque_[Nm]', 'Tool_wear_[min]', 'Machine_failure'
            ],
            'ind_quality': ind_quality_features + ['Quality']
        }

    def clean_column_names(self, df):
        new_cols = [c.strip().replace('.', '_').replace(' ', '_') for c in df.columns]
        return df.toDF(*new_cols)

    def custom_preprocessing(self, df, task_name):
        if task_name == 'trans_accident':
            df = df.filter(col("Severity").isin([2, 3]))
        elif task_name == 'ind_quality':
            if 'Time' in df.columns: df = df.drop('Time')
            drop_cols = ['157', '158', '220', '245', '292', '293', '358', '492', '517', '85']
            for c in drop_cols:
                 if c in df.columns: df = df.drop(c)
            if 'Pass/Fail' in df.columns:
                df = df.withColumn("Quality", when(col('Pass/Fail') == 1, 1).otherwise(0))
        return df
    
    def _validate_and_filter_columns(self, df, scene_type):
        required_cols = self.SCENE_REQUIRED_COLS.get(scene_type, [])
        if not required_cols:
            return df 
            
        current_cols = set(df.columns)
        required_set = set(required_cols)
        
        missing_cols = required_set - current_cols
        if missing_cols:
            missing_list = sorted(list(missing_cols))
            msg = f"{', '.join(missing_list[:5])}..." if len(missing_list) > 5 else ', '.join(missing_list)
            raise Exception(f"数据格式错误！缺少 {len(missing_list)} 个关键特征列: {msg}")
            
        target_col = self.DATASET_META.get(scene_type)
        if target_col: target_col = target_col.replace('.', '_').replace(' ', '_')
        
        allowed_cols = required_set.copy()
        if target_col: allowed_cols.add(target_col)
        
        if scene_type == 'ind_quality':
            allowed_cols.add('Pass/Fail')
            allowed_cols.add('Time')
        
        id_cols = {'id', 'ID', 'user_id', 'User_ID', 'customer_id', '_c0', 'unnamed:_0'}
        for c in df.columns:
            if c.lower() in id_cols:
                allowed_cols.add(c)
        
        final_cols = [c for c in df.columns if c in allowed_cols]
        print(f"   🛡️ [数据校验] 原始列数: {len(df.columns)}, 校验后保留: {len(final_cols)}")
        return df.select(*final_cols)

    def _get_feature_cols(self, df, target_col):
        feature_cols = []
        for c in df.columns:
            if c == target_col: continue
            lower_c = c.lower()
            if lower_c in ['id', 'user_id', 'order_id', '_c0', 'unnamed: 0']: continue
            if c == 'Time': continue
            if c in ['157', '158', '220', '245', '292', '293', '358', '492', '517', '85']: continue
            if c == 'Pass/Fail': continue 
            feature_cols.append(c)
        return feature_cols

    def _prepare_features_fallback(self, df, feature_cols):
        numeric_cols = [f.name for f in df.schema.fields if f.name in feature_cols and f.dataType.simpleString() != 'string']
        if not numeric_cols: return None
        assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features", handleInvalid="skip")
        return assembler.transform(df)

    def _calculate_confusion_matrix(self, predictions, target_col, scene_type):
        counts = predictions.groupBy("label", "prediction").count().collect()
        labels = predictions.select("label").distinct().rdd.flatMap(lambda x: x).collect()
        labels = sorted([int(l) for l in labels])
        
        label_map = self.SCENE_LABEL_MAP.get(scene_type, {})
        categories = []
        for l in labels:
            name = label_map.get(l, str(l))
            # === 修改：不再移除文字描述，保留完整标签 ===
            # name = str(name).replace("等级", "").split("(")[0] 
            categories.append(name)
            
        matrix_data = []
        count_dict = {}
        for row in counts:
            count_dict[(int(row['label']), int(row['prediction']))] = row['count']
            
        for i, actual_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                val = count_dict.get((actual_label, pred_label), 0)
                matrix_data.append([j, i, val])
        return categories, matrix_data

    def _extract_feature_importance(self, model, feature_cols):
        importances = []
        try:
            clf_model = model.stages[-1] if isinstance(model, PipelineModel) else model
            if hasattr(clf_model, 'featureImportances'):
                vals = clf_model.featureImportances.toArray()
                if len(vals) <= len(feature_cols): 
                    for i, score in enumerate(vals):
                        name = feature_cols[i] if i < len(feature_cols) else f"Feature_{i}"
                        importances.append({"name": name, "value": float(score)})
                    importances.sort(key=lambda x: x['value'], reverse=True)
                    importances = importances[:10]
        except Exception as e:
            print(f"   ⚠️ 无法提取特征重要性: {e}")
        return importances

    def _run_model(self, df, model_type, scene_type, task_id, has_label, target_col, feature_cols):
        model_name = f"{scene_type}_{model_type}.model"
        model_path = os.path.join(self.models_dir, model_name)
        
        if not os.path.exists(model_path):
            return 0.0, 0.0, None, [], f"模型文件不存在: {model_name}"
            
        try:
            update_task_progress(task_id, current_model=model_type, model_progress_status=f"正在评估 {model_type}...")
            print(f"   🤖 正在运行: {model_type} ...")
            
            model = None
            is_pipeline = False
            try:
                model = PipelineModel.load(model_path)
                is_pipeline = True
            except:
                if model_type == 'random_forest': model = RandomForestClassificationModel.load(model_path)
                elif model_type == 'gbdt': model = GBTClassificationModel.load(model_path)
                elif model_type == 'naive_bayes': model = NaiveBayesModel.load(model_path)
                else: return 0.0, 0.0, None, [], f"不支持的模型: {model_type}"

            if not is_pipeline and 'features' not in df.columns:
                df_prep = self._prepare_features_fallback(df, feature_cols)
                if df_prep is None: return 0.0, 0.0, None, [], "特征处理失败"
                df = df_prep

            predictions = model.transform(df)
            
            acc = 0.0
            f1 = 0.0
            if has_label:
                predictions = predictions.withColumn("prediction", col("prediction").cast("double"))
                if scene_type == 'trans_accident':
                    predictions = predictions.withColumn("label", 
                        when(col(target_col) == 2, 0.0)
                        .when(col(target_col) == 3, 1.0)
                        .otherwise(None)
                    ).filter(col("label").isNotNull())
                else:
                    predictions = predictions.withColumn("label", col(target_col).cast("double"))
                
                f1 = MulticlassClassificationEvaluator(metricName="f1").evaluate(predictions)
                acc = MulticlassClassificationEvaluator(metricName="accuracy").evaluate(predictions)
                print(f"     📈 {model_type} -> F1: {f1*100:.2f}%, Acc: {acc*100:.2f}%")
            
            feature_imp = self._extract_feature_importance(model, feature_cols)

            return f1, acc, predictions, feature_imp, None
            
        except Exception as e:
            print(f"     ❌ {model_type} 运行失败: {e}")
            return 0.0, 0.0, None, [], str(e)

    def predict(self, file_path, model_type, scene_type, task_id):
        print(f"⚡ 开始预测: {scene_type} | 模式: {model_type}")
        
        try:
            update_task_progress(task_id, current_model='None', model_progress_status="正在读取数据/预处理...")
            
            sample_df = self.spark.read.option("header", "true").option("inferSchema", "true").csv(file_path).limit(1000)
            target_schema = sample_df.schema
            df = self.spark.read.option("header", "true").schema(target_schema).csv(file_path)
            
            df = self.clean_column_names(df)
            df = self._validate_and_filter_columns(df, scene_type)
            df = df.fillna(0).fillna("Unknown")
            df = self.custom_preprocessing(df, scene_type)
            
            target_col = self.DATASET_META.get(scene_type)
            if target_col: target_col = target_col.replace('.', '_').replace(' ', '_')
            has_label = target_col in df.columns
            
            feature_cols = self._get_feature_cols(df, target_col)

            best_acc = -1.0
            best_predictions = None
            final_model_name = model_type
            final_feature_imp = [] 
            
            if model_type == 'auto':
                candidates = ['random_forest', 'gbdt', 'naive_bayes']
                results = [] 
                for m in candidates:
                    f1, acc, preds, imp, err = self._run_model(df, m, scene_type, task_id, has_label, target_col, feature_cols)
                    if preds is not None: results.append((f1, acc, m, preds, imp))
                
                if not results: return {"error": "所有模型运行失败。"}
                
                if has_label:
                    results.sort(key=lambda x: x[0], reverse=True) 
                    best_f1, final_acc, final_model_name, best_predictions, final_feature_imp = results[0]
                    print(f"   🏆 Auto 最佳: {final_model_name} (F1: {best_f1*100:.2f}%)")
                    best_acc = final_acc
                else:
                    rf_res = next((r for r in results if r[2] == 'random_forest'), results[0])
                    best_f1, final_acc, final_model_name, best_predictions, final_feature_imp = rf_res
                    best_acc = final_acc
            else:
                best_f1, final_acc, best_predictions, final_feature_imp, err = self._run_model(df, model_type, scene_type, task_id, has_label, target_col, feature_cols)
                if best_predictions is None: return {"error": err}
                best_acc = final_acc
                final_model_name = model_type

            confusion_matrix = []
            matrix_categories = []
            if has_label:
                try:
                    matrix_categories, confusion_matrix = self._calculate_confusion_matrix(best_predictions, target_col, scene_type)
                except Exception as e:
                    print(f"   ⚠️ 混淆矩阵计算失败: {e}")

            update_task_progress(task_id, current_model=final_model_name, model_progress_status="生成报表...")
            
            results_rows = best_predictions.select("prediction", target_col if has_label else lit(None)).limit(100).collect()
            dist_rows = best_predictions.groupBy("prediction").count().collect()
            
            current_label_map = self.SCENE_LABEL_MAP.get(scene_type, {})
            
            distribution = []
            for row in dist_rows:
                pred_val = int(row["prediction"])
                name = current_label_map.get(pred_val, str(pred_val))
                # === 修改：不再移除文字描述 ===
                # name = str(name).replace("等级", "").split("(")[0]
                distribution.append({"name": name, "value": row["count"]})

            chart_data = []
            for i, r in enumerate(results_rows):
                pred_val = int(r["prediction"])
                # === 修改：保留完整预测字符串 ===
                pred_str = current_label_map.get(pred_val, str(pred_val))
                
                act_str = "-"
                if has_label and r[1] is not None:
                    act_val = int(r[1])
                    # === 修改：保留完整真实字符串 ===
                    act_str = current_label_map.get(act_val, str(act_val))
                
                chart_data.append({"id": i+1, "actual": act_str, "predicted": pred_str})

            update_task_progress(task_id, current_model=final_model_name, model_progress_status="完成")
            
            return {
                "accuracy": best_acc,
                "chart_data": chart_data,
                "distribution": distribution,
                "final_model": final_model_name,
                "confusion_matrix": confusion_matrix,
                "all_labels_indices": matrix_categories,
                "feature_importance": final_feature_imp,
                "classification_report": []
            }

        except Exception as e:
            print(f"❌ 预测出错: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def stop(self):
        self.spark.stop()