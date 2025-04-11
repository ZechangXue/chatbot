@echo off

echo Building Docker image...
docker build -t gambling-prediction-app-api .

echo Running Docker container...
docker run -p 8080:8000 --env-file .env gambling-prediction-app-api

echo Application is running at http://localhost:8080 