#!/bin/bash

# 构建Docker镜像
echo "Building Docker image..."
docker build -t gambling-prediction-app-api .

# 运行Docker容器
echo "Running Docker container..."
docker run -p 8080:8000 --env-file .env gambling-prediction-app-api

echo "Application is running at http://localhost:8080" 