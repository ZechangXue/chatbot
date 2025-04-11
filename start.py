import os
import sys
import uvicorn
import shutil
import subprocess
import time
from dotenv import load_dotenv

# 添加当前目录和backend目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(current_dir, 'backend')
sys.path.insert(0, current_dir)
sys.path.insert(0, backend_dir)

# 打印当前Python路径，用于调试
print("Python path:", sys.path)
print("Current directory:", current_dir)
print("Backend directory:", backend_dir)
print("Files in current directory:", os.listdir(current_dir))
print("Files in backend directory:", os.listdir(backend_dir))

# 检查CSV文件是否存在
csv_path = os.path.join(current_dir, 'sample_txns.csv')
if os.path.exists(csv_path):
    print(f"CSV file exists at {csv_path}")
    # 如果在Docker容器中，确保CSV文件在正确的位置
    if os.path.exists('/app') and current_dir != '/app':
        docker_csv_path = '/app/sample_txns.csv'
        if not os.path.samefile(csv_path, docker_csv_path):
            print(f"Copying CSV file to {docker_csv_path}")
            shutil.copy(csv_path, docker_csv_path)
        else:
            print("CSV file already in the correct location")
else:
    print(f"CSV file not found at {csv_path}")

# 加载环境变量
load_dotenv()

def start_backend():
    print("Starting backend service...")
    backend_process = subprocess.Popen(
        ["python", "backend/app.py"],
        env=dict(os.environ, HOST="0.0.0.0", PORT="8000")
    )
    return backend_process

def start_frontend():
    print("Starting frontend service...")
    frontend_process = subprocess.Popen(
        ["npm", "start"],
        cwd="frontend",
        env=dict(os.environ, REACT_APP_API_URL="http://0.0.0.0:8000")
    )
    return frontend_process

def main():
    try:
        backend = start_backend()
        time.sleep(2)  # 等待后端启动
        frontend = start_frontend()
        
        print("\nServices are running:")
        print("Backend API: http://0.0.0.0:8000")
        print("Frontend: http://localhost:3000")
        
        # 保持程序运行
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down services...")
        backend.terminate()
        frontend.terminate()
        print("Services stopped.")

if __name__ == "__main__":
    main() 