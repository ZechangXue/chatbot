# 使用Python 3.9作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY backend/requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码和模型文件
COPY backend/ backend/
COPY frontend/ frontend/
COPY models/ models/
COPY start.py .

# 复制样本数据
COPY sample_txns.csv .

# 设置环境变量
ENV PYTHONPATH=/app

# 暴露端口
EXPOSE 8000

# 启动应用
CMD ["python", "start.py"] 