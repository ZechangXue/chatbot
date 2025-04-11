import os
import sys
import uvicorn
import shutil

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

if __name__ == "__main__":
    # 启动FastAPI应用
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000) 