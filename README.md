# 赌博行为预测系统

这是一个基于机器学习的赌博行为预测系统，可以根据用户的基本信息预测其工资水平、赌博存款、使用赌博运营商数量和赌博频率。系统还包含一个智能聊天助手，可以回答用户的问题。

## 功能特点

1. **多维度预测**
   - 工资水平预测
   - 赌博存款预测
   - 赌博运营商使用数量预测
   - 赌博频率预测

2. **智能用户画像**
   - 基于预测结果生成个性化用户画像
   - 提供用户行为特征描述

3. **智能聊天助手**
   - 实时回答用户问题
   - 支持自然语言交互
   - 可配置OpenAI API密钥

## 技术栈

- 后端：FastAPI + Python
- 前端：HTML + TailwindCSS + JavaScript
- 机器学习：XGBoost
- 部署：Docker

## 快速开始

### 环境要求

- Docker
- OpenAI API密钥

### 安装步骤

1. 克隆项目到本地：
   ```bash
   git clone <repository_url>
   cd gambling_prediction_app
   ```

2. 创建环境变量文件：
   ```bash
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

3. 复制模型文件：
   将所有模型文件（.json和.pkl）复制到`models/`目录下。

4. 构建Docker镜像：
   ```bash
   docker build -t gambling-prediction-app .
   ```

5. 运行容器：
   ```bash
   docker run -p 8000:8000 --env-file .env gambling-prediction-app
   ```

6. 访问应用：
   打开浏览器访问 http://localhost:8000

## 使用说明

1. **输入用户信息**
   - 选择国家/地区（英格兰、苏格兰、威尔士、北爱尔兰）
   - 输入年龄段
   - 选择性别
   - 输入收入排名

2. **查看预测结果**
   - 系统会显示四个维度的预测结果
   - 每个预测都包含置信度范围
   - 自动生成用户画像

3. **使用聊天助手**
   - 点击右下角的聊天图标
   - 输入问题并等待回答
   - 支持自然语言交互

## 注意事项

1. 请确保`.env`文件中包含有效的OpenAI API密钥
2. 所有模型文件必须放在正确的目录中
3. 预测结果仅供参考，不应作为唯一决策依据

## 开发者说明

### 项目结构
```
gambling_prediction_app/
├── backend/
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   └── index.html
├── models/
│   ├── label_encoders_*.pkl
│   ├── scalers_*.pkl
│   └── xgboost_model_*.json
├── Dockerfile
└── README.md
```

### API端点

1. `/predict` (POST)
   - 接收用户信息并返回预测结果
   - 包含置信度范围和用户画像

2. `/chat` (POST)
   - 处理用户聊天消息
   - 返回AI助手的回答

## 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 发起Pull Request

## 许可证

MIT License

## 新增功能：数据查询和计算

本应用现在支持通过聊天框进行数据查询和计算功能。用户可以使用自然语言查询交易数据，系统会自动解析查询参数，执行相应的数据过滤和计算操作。

### 支持的查询类型

1. **日期范围查询**：
   - 示例：`Show me transactions from 2025-12-30 to 2025-12-31`
   - 格式：使用 `from [日期]` 或 `between [日期] to [日期]` 格式

2. **交易类型查询**：
   - 示例：`Show me transactions where type is DEBIT`
   - 格式：使用 `type is [类型]` 格式，类型可以是 `DEBIT` 或 `CREDIT`

3. **标签查询**：
   - 示例：`Show me transactions where label is gambling`
   - 格式：使用 `label is [标签]` 格式，标签可以是 `gambling`, `other_discretionary`, `other_transfer` 等

4. **账户查询**：
   - 示例：`Show me transactions where account is 7361d008-82de-4026-8908-aa5d5102ee56`
   - 格式：使用 `account is [账户UUID]` 格式

5. **商户查询**：
   - 示例：`Show me transactions where merchant is mrc_PBEa5KrDCGRCYbENEe26Qw`
   - 格式：使用 `merchant is [商户UUID]` 格式

### 支持的计算操作

1. **求和计算**：
   - 示例：`What is the sum of gambling transactions?`
   - 格式：使用 `sum` 或 `total` 关键词

2. **平均值计算**：
   - 示例：`What is the average amount of DEBIT transactions?`
   - 格式：使用 `average`, `mean` 或 `avg` 关键词

3. **分组计算**：
   - 示例：`Sum of transactions group by type`
   - 格式：使用 `group by [列名]` 格式，列名可以是 `type`, `label` 等

### 组合查询示例

1. `What is the total amount of gambling transactions from 2025-12-30 to 2025-12-31?`
2. `Calculate the average transaction amount where label is gambling`
3. `Sum of DEBIT transactions group by label`
4. `What is the average amount of transactions where type is CREDIT and label is gambling?`

### 技术实现

该功能通过以下组件实现：

1. **数据处理模块**：`data_processor.py`
   - 负责加载和查询CSV数据
   - 使用正则表达式从自然语言查询中提取参数

2. **公式索引库**：`formula_index.py`
   - 提供计算功能，如求和和平均值
   - 支持分组计算

3. **聊天集成**：
   - 在聊天功能中集成数据查询和计算
   - 使用OpenAI API生成基于计算结果的自然语言回复

## EC2部署指南

### 前提条件
- EC2实例（Amazon Linux）
- 已安装Docker
- 已克隆代码仓库
- 已配置安全组（开放8000端口）

### 部署步骤

1. **连接到EC2实例**
```bash
ssh -i chatbot-key.pem ec2-user@16.170.253.244
```

2. **安装依赖**
```bash
# 安装Docker（如果尚未安装）
sudo yum update -y
sudo yum install docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user

# 安装Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

3. **配置环境变量**
```bash
# 创建.env文件
cat > .env << EOL
# 添加您的环境变量
EOL
```

4. **构建和运行Docker容器**
```bash
# 构建镜像
docker build -t chatbot-app .

# 运行容器
docker run -d \
  --name chatbot-app \
  -p 8000:8000 \
  --env-file .env \
  chatbot-app
```

5. **访问服务**
- 后端API: http://16.170.253.244:8000
- 前端界面: http://16.170.253.244:8000

### 维护命令
```bash
# 查看日志
docker logs chatbot-app

# 重启服务
docker restart chatbot-app

# 停止服务
docker stop chatbot-app

# 删除容器
docker rm chatbot-app
```

### 故障排除
1. 检查Docker服务状态
```bash
sudo service docker status
```

2. 检查容器状态
```bash
docker ps -a
```

3. 检查端口监听
```bash
sudo netstat -tuln | grep 8000
``` 