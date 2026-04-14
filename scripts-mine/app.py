import os
import subprocess
import uuid
from flask import Flask, request, jsonify, render_template, session, Response, stream_with_context
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
SCRIPTS_DIR = BASE_DIR
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

SCRIPT_MAP = {
    ('gcg', 'none'): 'gcg_none_defense.sh',
    ('autodan', 'none'): 'autodan_none_defense.sh',
    ('pair', 'none'): 'pair_none_defense.sh',
    ('gcg', 'smoothllm'): 'smoothllm_gcg.sh',
    ('autodan', 'smoothllm'): 'smoothllm_autodan.sh',
    ('pair', 'smoothllm'): 'smoothllm_pair.sh',
    ('gcg', 'selfreminder'): 'selfreminder_gcg.sh',
    ('autodan', 'selfreminder'): 'selfreminder_autodan.sh',
    ('pair', 'selfreminder'): 'selfreminder_pair.sh',
    ('gcg', 'safetytraining'): 'safetytraining_gcg.sh',
    ('autodan', 'safetytraining'): 'safetytraining_autodan.sh',
    ('pair', 'safetytraining'): 'safetytraining_pair.sh',
    # 预算测试：按攻击方法区分
    ('budget', 'pair'): 'test_atk_budget_pair.sh',
    ('budget', 'autodan'): 'test_atk_budget_gcg.sh',
    # 其他环境测试
    'capability': 'test_atk_ability_pair.sh',      # 能力
    'alignment': 'target_align_pair',           # 微调
    'system_prompt': 'target_size_pair.sh',      # 模型大小（原系统提示词）
    'virtual': 'virtual_thought.sh'
}

def run_script(script_path):
    """原有阻塞式执行（保留用于非流式接口）"""
    try:
        if not os.access(script_path, os.X_OK):
            os.chmod(script_path, 0o755)
        result = subprocess.run(
            ['bash', script_path],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300
        )
        output = result.stdout + result.stderr
        return output, result.returncode == 0
    except subprocess.TimeoutExpired:
        return "脚本执行超时（超过5分钟）", False
    except Exception as e:
        return f"执行出错：{str(e)}", False

@app.route('/')
def index():
    return render_template('index.html')

# ------------------- 新增流式攻击接口 -------------------
@app.route('/stream_run_attack')
def stream_run_attack():
    attack = request.args.get('attack')
    defense = request.args.get('defense', 'none')
    key = (attack, defense)
    if key not in SCRIPT_MAP:
        return Response(f"event: error\ndata: 未找到对应的脚本：攻击={attack}, 防御={defense}\n\n", mimetype='text/event-stream')
    script_name = SCRIPT_MAP[key]
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    if not os.path.exists(script_path):
        return Response(f"event: error\ndata: 脚本文件不存在：{script_name}\n\n", mimetype='text/event-stream')

    def generate():
        # 发送开始事件（可选）
        yield f"event: start\ndata: 开始执行脚本 {script_name}\n\n"
        try:
            # 确保脚本可执行
            if not os.access(script_path, os.X_OK):
                os.chmod(script_path, 0o755)
            # 使用 Popen 逐行读取输出
            process = subprocess.Popen(
                ['bash', script_path],
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # 合并 stderr 到 stdout
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            for line in iter(process.stdout.readline, ''):
                if line:
                    # 将每一行作为 data 事件发送（去掉末尾换行符）
                    yield f"data: {line.rstrip()}\n\n"
            process.wait()
            return_code = process.returncode
            if return_code == 0:
                yield f"event: end\ndata: 执行成功\n\n"
            else:
                yield f"event: error\ndata: 执行失败，返回码 {return_code}\n\n"
        except Exception as e:
            yield f"event: error\ndata: 执行出错：{str(e)}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/stream_run_environment')
def stream_run_environment():
    env_type = request.args.get('env_type')
    attack_method = request.args.get('attack_method')  # 预算测试专用

    # 确定脚本名
    if env_type == 'budget':
        if not attack_method:
            return Response(f"event: error\ndata: 预算测试必须指定攻击方法\n\n", mimetype='text/event-stream')
        key = (env_type, attack_method)
        if key not in SCRIPT_MAP:
            return Response(f"event: error\ndata: 未找到预算脚本：攻击方法={attack_method}\n\n", mimetype='text/event-stream')
        script_name = SCRIPT_MAP[key]
    else:
        if env_type not in SCRIPT_MAP:
            return Response(f"event: error\ndata: 未找到环境测试脚本：{env_type}\n\n", mimetype='text/event-stream')
        script_name = SCRIPT_MAP[env_type]

    script_path = os.path.join(SCRIPTS_DIR, script_name)
    if not os.path.exists(script_path):
        return Response(f"event: error\ndata: 脚本文件不存在：{script_name}\n\n", mimetype='text/event-stream')

    def generate():
        yield f"event: start\ndata: 开始执行脚本 {script_name}\n\n"
        try:
            if not os.access(script_path, os.X_OK):
                os.chmod(script_path, 0o755)
            process = subprocess.Popen(
                ['bash', script_path],
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            for line in iter(process.stdout.readline, ''):
                if line:
                    yield f"data: {line.rstrip()}\n\n"
            process.wait()
            if process.returncode == 0:
                yield f"event: end\ndata: 执行成功\n\n"
            else:
                yield f"event: error\ndata: 执行失败，返回码 {process.returncode}\n\n"
        except Exception as e:
            yield f"event: error\ndata: 执行出错：{str(e)}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/stream_run_uploaded')
def stream_run_uploaded():
    script_path = session.get('uploaded_script')
    if not script_path or not os.path.exists(script_path):
        return Response(f"event: error\ndata: 没有已上传的脚本或文件已丢失\n\n", mimetype='text/event-stream')

    def generate():
        yield f"event: start\ndata: 开始执行上传的脚本\n\n"
        try:
            if not os.access(script_path, os.X_OK):
                os.chmod(script_path, 0o755)
            process = subprocess.Popen(
                ['bash', script_path],
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            for line in iter(process.stdout.readline, ''):
                if line:
                    yield f"data: {line.rstrip()}\n\n"
            process.wait()
            if process.returncode == 0:
                yield f"event: end\ndata: 执行成功\n\n"
            else:
                yield f"event: error\ndata: 执行失败，返回码 {process.returncode}\n\n"
        except Exception as e:
            yield f"event: error\ndata: 执行出错：{str(e)}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

# 以下接口保持不变（仍使用阻塞式执行）
@app.route('/run_attack', methods=['POST'])
def run_attack():
    data = request.get_json()
    attack = data.get('attack')
    defense = data.get('defense', 'none')
    key = (attack, defense)
    if key not in SCRIPT_MAP:
        return jsonify({'success': False, 'output': f'未找到对应的脚本：攻击={attack}, 防御={defense}'})
    script_name = SCRIPT_MAP[key]
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    if not os.path.exists(script_path):
        return jsonify({'success': False, 'output': f'脚本文件不存在：{script_name}'})
    output, ok = run_script(script_path)
    return jsonify({'success': ok, 'output': output})


@app.route('/run_environment', methods=['POST'])
def run_environment():
    data = request.get_json()
    env_type = data.get('env_type')
    attack_method = data.get('attack_method')

    # 预算测试需要组合键
    if env_type == 'budget':
        if not attack_method:
            return jsonify({'success': False, 'output': '预算测试必须指定攻击方法'})
        key = (env_type, attack_method)
        if key not in SCRIPT_MAP:
            return jsonify({'success': False, 'output': f'未找到预算脚本：攻击方法={attack_method}'})
        script_name = SCRIPT_MAP[key]
    else:
        # 其他环境测试使用字符串键
        if env_type not in SCRIPT_MAP:
            return jsonify({'success': False, 'output': f'未找到环境测试脚本：{env_type}'})
        script_name = SCRIPT_MAP[env_type]

    script_path = os.path.join(SCRIPTS_DIR, script_name)
    if not os.path.exists(script_path):
        return jsonify({'success': False, 'output': f'脚本文件不存在：{script_name}'})

    output, ok = run_script(script_path)
    return jsonify({'success': ok, 'output': output})

@app.route('/run_virtual', methods=['POST'])
def run_virtual():
    script_name = SCRIPT_MAP['virtual']
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    if not os.path.exists(script_path):
        with open(script_path, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('echo "=== 模拟模型思考过程 ===\n"\n')
            f.write('echo "中间层激活值: [0.12, 0.45, 0.78, ...]"\n')
            f.write('echo "注意力分布: 第5层注意力头3集中在token 12"\n')
            f.write('echo "梯度信息: ..."\n')
        os.chmod(script_path, 0o755)
    output, ok = run_script(script_path)
    return jsonify({'success': ok, 'output': output})

@app.route('/upload_script', methods=['POST'])
def upload_script():
    if 'file' not in request.files:
        return jsonify({'success': False, 'output': '没有文件上传'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'output': '文件名为空'})
    if not file.filename.endswith('.sh'):
        return jsonify({'success': False, 'output': '只支持上传 .sh 脚本文件'})
    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(save_path)
    os.chmod(save_path, 0o755)
    session['uploaded_script'] = save_path
    return jsonify({'success': True, 'output': f'脚本上传成功：{filename}'})

@app.route('/run_uploaded', methods=['POST'])
def run_uploaded():
    script_path = session.get('uploaded_script')
    if not script_path or not os.path.exists(script_path):
        return jsonify({'success': False, 'output': '没有已上传的脚本或文件已丢失'})
    output, ok = run_script(script_path)
    return jsonify({'success': ok, 'output': output})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)