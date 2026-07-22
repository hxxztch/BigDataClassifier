"""共享预处理函数 — 消除 spark_utils.py 与 train_models.py 之间的重复"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when
from .config import (
    SCENE_REQUIRED_COLS, DATASET_META, IND_QUALITY_DROP_COLS
)
from .logger import get_logger

logger = get_logger(__name__)


def clean_column_names(df: DataFrame) -> DataFrame:
    """清理列名：去除首尾空格，将 . 和空格替换为 _"""
    new_cols = [c.strip().replace('.', '_').replace(' ', '_') for c in df.columns]
    return df.toDF(*new_cols)


def custom_preprocessing(df: DataFrame, scene_type: str) -> DataFrame:
    """场景特定的预处理逻辑"""
    if scene_type == 'trans_accident':
        logger.info("过滤极少量 Severity 1 和 4...")
        df = df.filter(col("Severity").isin([2, 3]))

    elif scene_type == 'ind_quality':
        if 'Time' in df.columns:
            df = df.drop('Time')
        for c in IND_QUALITY_DROP_COLS - {'Time'}:
            if c in df.columns:
                df = df.drop(c)
        if 'Pass/Fail' in df.columns:
            df = df.withColumn("Quality", when(col('Pass/Fail') == 1, 1).otherwise(0))

    elif scene_type == 'shop_shipping':
        logger.info("提取 High_Discount 和 Is_Heavy 特征...")
        df = df.withColumn("High_Discount", when(col("Discount_offered") > 10, 1).otherwise(0))
        df = df.withColumn("Is_Heavy", when(col("Weight_in_gms") > 4000, 1).otherwise(0))

    return df


def validate_and_filter_columns(df: DataFrame, scene_type: str) -> DataFrame:
    """校验数据列是否满足场景要求，并过滤在白名单之外的无用列"""
    required_cols = SCENE_REQUIRED_COLS.get(scene_type, [])
    if not required_cols:
        return df

    current_cols = set(df.columns)
    required_set = set(required_cols)

    # 检查缺失列
    missing_cols = required_set - current_cols
    if missing_cols:
        missing_list = sorted(missing_cols)
        msg = f"{', '.join(missing_list[:5])}..." if len(missing_list) > 5 else ', '.join(missing_list)
        raise ValueError(f"数据格式错误！缺少 {len(missing_list)} 个关键特征列: {msg}")

    # 确定目标列名（可能已被 clean_column_names 改写过）
    target_col = DATASET_META.get(scene_type, '')
    if target_col:
        target_col = target_col.replace('.', '_').replace(' ', '_')

    allowed_cols = required_set.copy()
    if target_col:
        allowed_cols.add(target_col)

    # ind_quality 额外保留 Pass/Fail 和 Time 用于自定义预处理
    if scene_type == 'ind_quality':
        allowed_cols.add('Pass/Fail')
        allowed_cols.add('Time')

    # 保留常见 ID 列（不参与建模但用于索引）
    id_cols = {'id', 'ID', 'user_id', 'User_ID', 'customer_id', '_c0', 'unnamed:_0'}
    for c in df.columns:
        if c.lower() in id_cols:
            allowed_cols.add(c)

    final_cols = [c for c in df.columns if c in allowed_cols]
    logger.info(f"原始列数: {len(df.columns)}, 校验后保留: {len(final_cols)}")
    return df.select(*final_cols)


def get_feature_cols(df: DataFrame, target_col: str = None) -> list:
    """从 DataFrame 中提取特征列名（排除目标列、ID 列、特殊噪声列）"""
    skip_cols = {'id', 'user_id', 'order_id', '_c0', 'unnamed: 0', 'Time'}
    skip_lower = {s.lower() for s in skip_cols}

    feature_cols = []
    for c in df.columns:
        if target_col and c == target_col:
            continue
        if c.lower() in skip_lower:
            continue
        if c in IND_QUALITY_DROP_COLS:
            continue
        if c == 'Pass/Fail':
            continue
        feature_cols.append(c)
    return feature_cols


def prepare_features_fallback(df: DataFrame, feature_cols: list) -> DataFrame:
    """兜底特征向量化：当加载的模型不是完整 PipelineModel 时使用
       只选取数值型列，不处理字符串特征
    """
    from pyspark.ml.feature import VectorAssembler
    numeric_cols = [
        f.name for f in df.schema.fields
        if f.name in feature_cols and f.dataType.simpleString() != 'string'
    ]
    if not numeric_cols:
        return None
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features", handleInvalid="skip")
    return assembler.transform(df)


def calculate_confusion_matrix(predictions: DataFrame, target_col: str, scene_type: str):
    """计算混淆矩阵数据及类别标签

    Returns:
        (categories: list[str], matrix_data: list[list[int]])
        matrix_data 的格式为 [[pred_idx, true_idx, count], ...]
    """
    from .config import SCENE_LABEL_MAP

    counts = predictions.groupBy("label", "prediction").count().collect()
    labels = sorted([int(l) for l in predictions.select("label").distinct().rdd.flatMap(lambda x: x).collect()])

    label_map = SCENE_LABEL_MAP.get(scene_type, {})
    categories = [label_map.get(l, str(l)) for l in labels]

    count_dict = {}
    for row in counts:
        count_dict[(int(row['label']), int(row['prediction']))] = row['count']

    matrix_data = []
    for i, actual_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            val = count_dict.get((actual_label, pred_label), 0)
            matrix_data.append([j, i, val])

    return categories, matrix_data


def extract_feature_importance(model, feature_cols: list) -> list:
    """从树模型中提取 Top-10 特征重要性"""
    importances = []
    try:
        clf_model = model
        if hasattr(model, 'stages'):
            clf_model = model.stages[-1]

        if hasattr(clf_model, 'featureImportances'):
            vals = clf_model.featureImportances.toArray()
            limit = min(len(vals), len(feature_cols))
            for i in range(limit):
                importances.append({"name": feature_cols[i], "value": float(vals[i])})
        elif hasattr(clf_model, 'feature_importances_'):
            fi = clf_model.feature_importances_
            if isinstance(fi, dict):
                for key, val in fi.items():
                    try:
                        idx = int(str(key).lstrip('f'))
                        if 0 <= idx < len(feature_cols):
                            importances.append({"name": feature_cols[idx], "value": float(val)})
                    except (ValueError, TypeError):
                        pass
            else:
                for i in range(min(len(fi), len(feature_cols))):
                    importances.append({"name": feature_cols[i], "value": float(fi[i])})
        elif hasattr(clf_model, 'get_feature_importances'):
            fi = clf_model.get_feature_importances()
            if isinstance(fi, dict):
                for key, val in fi.items():
                    try:
                        idx = int(str(key).lstrip('f'))
                        if 0 <= idx < len(feature_cols):
                            importances.append({"name": feature_cols[idx], "value": float(val)})
                    except (ValueError, TypeError):
                        pass
        if importances:
            importances.sort(key=lambda x: x['value'], reverse=True)
            importances = importances[:10]
    except Exception as e:
        logger.warning(f"无法提取特征重要性: {e}")
    return importances
def build_feature_preprocessing_stages(df: DataFrame, feature_cols: list, target_col: str):
    """构建训练阶段的 Pipeline 预处理 stages

    返回:
        (stages: list, assembler_inputs: list)
    """
    from pyspark.ml.feature import VectorAssembler, MinMaxScaler, StringIndexer

    stages = []
    # 标签索引化
    stages.append(StringIndexer(inputCol=target_col, outputCol="label").setHandleInvalid("skip"))

    # 特征列：字符串列索引化 + 全部组合为向量
    assembler_inputs = []
    for col_name in feature_cols:
        dtype = next((t for n, t in df.dtypes if n == col_name), 'string')
        if dtype == 'string':
            idx_name = f"{col_name}_idx"
            stages.append(StringIndexer(inputCol=col_name, outputCol=idx_name).setHandleInvalid("keep"))
            assembler_inputs.append(idx_name)
        else:
            assembler_inputs.append(col_name)

    stages.append(VectorAssembler(inputCols=assembler_inputs, outputCol="raw_features"))
    stages.append(MinMaxScaler(inputCol="raw_features", outputCol="features"))

    return stages
