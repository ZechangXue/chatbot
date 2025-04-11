#!/bin/bash

# 停止并删除现有容器
docker stop chatbot-app || true
docker rm chatbot-app || true

# 构建新镜像
docker build -t chatbot-app .

# 运行新容器
docker run -d \
  --name chatbot-app \
  -p 8000:8000 \
  --env-file .env \
  chatbot-app

# 显示容器状态
docker ps

echo "Deployment completed. Service is running at http://16.170.253.244:8000"

# 重置权限
icacls chatbot-key.pem /reset

# 只给当前用户读取权限
icacls chatbot-key.pem /grant:r "%USERNAME%":R

# 移除其他所有用户的权限
icacls chatbot-key.pem /remove:g "BUILTIN\Users"
icacls chatbot-key.pem /remove:g "Everyone" 