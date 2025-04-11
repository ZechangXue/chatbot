import os
import sys
import uvicorn

# 添加当前目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

if __name__ == "__main__":
    # 启动FastAPI应用
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True) 