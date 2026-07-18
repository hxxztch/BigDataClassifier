from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import threading
import traceback
from werkzeug.utils import secure_filename

# 引入数据库和预测模块
from database import init_db, create_task_entry, update_task_result, update_task_progress, get_history, get_task_status
from spark_utils import SparkClassifier

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# ================= 路径配置 =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'data', 'uploads')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024 # 10GB

# ================= 初始化 =================
init_db()
classifier = SparkClassifier()

def run_async_prediction(task_id, file_path, model_type, scene_type):
    """
    异步执行预测任务
    """
    print(f"🚀 [Task {task_id}] 后台开始处理...")
    try:
        # 调用 Spark 进行预测
        result = classifier.predict(file_path, model_type, scene_type, task_id)
        
        # 检查结果是否包含错误
        if 'error' in result:
            print(f"❌ [Task {task_id}] 预测失败: {result['error']}")
            update_task_result(task_id, 0.0, result, status='failed')
        else:
            acc = result.get('accuracy', 0.0)
            update_task_result(task_id, acc, result, status='completed')
            print(f"✅ [Task {task_id}] 处理完成")
            
    except Exception as e:
        print(f"❌ [Task {task_id}] 系统严重错误: {e}")
        traceback.print_exc()
        update_task_result(task_id, 0.0, {'error': str(e)}, status='failed')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        import time
        save_name = f"{int(time.time())}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], save_name)
        file.save(file_path)
        
        return jsonify({
            'message': 'File uploaded successfully',
            'file_path': file_path,
            'file_name': filename
        })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    file_path = data.get('file_path')
    model_type = data.get('model_type', 'auto')
    scene_type = data.get('scene_type', 'unknown')
    
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 400

    task_id = create_task_entry(file_path, model_type, scene_type)
    
    thread = threading.Thread(target=run_async_prediction, args=(task_id, file_path, model_type, scene_type))
    thread.start()
    
    return jsonify({'task_id': task_id, 'status': 'processing'})

@app.route('/task/<int:task_id>', methods=['GET'])
def get_task(task_id):
    task = get_task_status(task_id)
    if task:
        # === 核心修改：将 progress_info 扁平化，直接放在根节点 ===
        # 这样前端的 data.current_model 和 data.model_progress_status 就能取到值了
        response = {
            'task_id': task['task_id'],
            'status': task['status'],
            'result': task.get('result'),
            'finish_time': task['finish_time'],
            # 直接映射数据库字段到 JSON 根目录
            'current_model': task['current_model'],
            'model_progress_status': task['model_progress_status']
        }
        return jsonify(response)
    else:
        return jsonify({'error': 'Task not found'}), 404

@app.route('/history', methods=['GET'])
def history():
    scene_type = request.args.get('scene_type')
    tasks = get_history(scene_type)
    return jsonify(tasks)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)