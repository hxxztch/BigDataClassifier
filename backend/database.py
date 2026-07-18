import sqlite3
import os
import json
from datetime import datetime

# 数据库存储路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(CURRENT_DIR, 'classification.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    # 增加 finish_time 字段
    c.execute('''
        CREATE TABLE IF NOT EXISTS prediction_task (
            task_id INTEGER PRIMARY KEY AUTOINCREMENT,
            scene_type TEXT,
            predict_time TIMESTAMP,
            finish_time TIMESTAMP,
            file_path TEXT,
            model_type TEXT,
            accuracy REAL DEFAULT 0.0,
            status TEXT DEFAULT 'processing',
            result_json TEXT,
            current_model TEXT,
            model_progress_status TEXT
        )
    ''')
    
    # 检查并添加 finish_time 列（针对旧数据库的兼容性迁移）
    c.execute("PRAGMA table_info(prediction_task)")
    columns = [info[1] for info in c.fetchall()]
    if 'finish_time' not in columns:
        print("⚠️ 检测到旧版数据库，正在执行迁移: 添加 finish_time 列...")
        c.execute('ALTER TABLE prediction_task ADD COLUMN finish_time TIMESTAMP')
        
    conn.commit()
    conn.close()
    print(f"✅ 数据库就绪: {DB_PATH}")

def create_task_entry(file_path, model_type, scene_type):
    conn = get_db_connection()
    c = conn.cursor()
    predict_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not scene_type: scene_type = "unknown"
    
    c.execute('''
        INSERT INTO prediction_task (scene_type, predict_time, file_path, model_type, status, accuracy, current_model, model_progress_status)
        VALUES (?, ?, ?, ?, 'processing', 0.0, 'None', '任务排队中...')
    ''', (scene_type, predict_time, file_path, model_type))
    
    task_id = c.lastrowid
    conn.commit()
    conn.close()
    return task_id

def update_task_result(task_id, accuracy, result_data, status='completed'):
    conn = get_db_connection()
    c = conn.cursor()
    
    result_json_str = json.dumps(result_data, ensure_ascii=False) if result_data else ""
    finish_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    c.execute('''
        UPDATE prediction_task
        SET accuracy = ?, status = ?, result_json = ?, finish_time = ?, current_model = 'None', model_progress_status = '完成'
        WHERE task_id = ?
    ''', (accuracy, status, result_json_str, finish_time, task_id))
    
    conn.commit()
    conn.close()

def update_task_progress(task_id, current_model=None, model_progress_status=None):
    conn = get_db_connection()
    c = conn.cursor()
    
    updates = []
    params = []
    
    if current_model is not None:
        updates.append("current_model = ?")
        params.append(current_model)
    if model_progress_status is not None:
        updates.append("model_progress_status = ?")
        params.append(model_progress_status)
        
    if updates:
        query = f"UPDATE prediction_task SET {', '.join(updates)} WHERE task_id = ?"
        params.append(task_id)
        c.execute(query, params)
        conn.commit()
        
    conn.close()

def get_task_status(task_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM prediction_task WHERE task_id = ?", (task_id,))
    row = c.fetchone()
    conn.close()
    
    if row:
        row_dict = dict(row)
        if row_dict.get('result_json'):
            try:
                row_dict['result'] = json.loads(row_dict['result_json'])
            except:
                row_dict['result'] = None
        # 删除大字段以减轻网络传输（除非特定需要，这里get_task_status通常需要结果）
        del row_dict['result_json']
        return row_dict
    return None

def get_history(scene_type=None):
    conn = get_db_connection()
    c = conn.cursor()
    
    # 增加 finish_time 查询
    query = "SELECT task_id, scene_type, predict_time, finish_time, file_path, model_type, accuracy, status FROM prediction_task WHERE 1=1"
    params = []
    
    if scene_type and scene_type.strip():
        query += " AND scene_type = ?"
        params.append(scene_type)
    
    query += " ORDER BY task_id DESC" # 按ID倒序，最新的在前面
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]