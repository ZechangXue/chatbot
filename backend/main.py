from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import os
from dotenv import load_dotenv
from openai import OpenAI
import httpx
from pathlib import Path
import logging
import json
import asyncio
import re

# Import custom modules for data processing and formula calculation
from .data_processor import DataProcessor
from .formula_index import FormulaIndex

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.warning("API key not found!")

# Create OpenAI client
client = OpenAI(
    api_key=api_key,
)

# Initialize data processor and formula index
try:
    logger.info("Initializing data processor and formula index")
    data_processor = DataProcessor(csv_path="sample_txns.csv")
    formula_index = FormulaIndex()
    logger.info(f"Data processor initialized with DataFrame shape: {data_processor.df.shape if data_processor.df is not None else 'None'}")
except Exception as e:
    logger.error(f"Error initializing data processor: {str(e)}", exc_info=True)
    # 创建空的数据处理器，稍后会尝试重新加载
    data_processor = DataProcessor(csv_path="sample_txns.csv")
    formula_index = FormulaIndex()

# Mount static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("frontend/index.html")

# Define input models
class PredictionInput(BaseModel):
    country: str
    age_band: int
    gender: str
    lsoa_income_rank: int

class ChatInput(BaseModel):
    message: str

class DataQueryInput(BaseModel):
    query: str

# Define country configurations
COUNTRIES = {
    'england': {'defaults': {'age_band': 27, 'gender': 'M', 'lsoa_income_rank': 14264}},
    'scotland': {'defaults': {'age_band': 28, 'gender': 'M', 'lsoa_income_rank': 856}},
    'wales': {'defaults': {'age_band': 28, 'gender': 'M', 'lsoa_income_rank': 3809}},
    'ni': {'defaults': {'age_band': 28, 'gender': 'M', 'lsoa_income_rank': 461}}
}

# Load model components
def load_components(country: str, predictor_type: str):
    base_path = Path("models")
    logger.info(f"Loading models from {base_path}")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    try:
        label_encoder_path = base_path / f'label_encoders_{predictor_type}_{country}.pkl'
        scaler_path = base_path / f'scaler_{predictor_type}_{country}.pkl'
        model_path = base_path / f'xgboost_model_{predictor_type}_{country}.json'
        
        logger.info(f"Loading label encoder from {label_encoder_path}")
        label_encoder = joblib.load(label_encoder_path)
        
        logger.info(f"Loading scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)
        
        logger.info(f"Loading model from {model_path}")
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        
        return label_encoder, scaler, model
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        logger.error(f"Files in models directory: {list(base_path.glob('*'))}")
        raise HTTPException(status_code=500, detail=f"Error loading models for {country}/{predictor_type}: {str(e)}")

# Calculate confidence levels
def calculate_confidence_levels(pred_proba):
    pred_class = int(np.argmax(pred_proba))  # Convert to Python int type
    one_band = [int(x) for x in sorted([pred_class])]
    two_band = [int(x) for x in sorted([pred_class] + ([pred_class + 1] if pred_class + 1 < len(pred_proba) else [pred_class - 1]))]
    three_band = [int(x) for x in sorted(one_band + ([pred_class - 1] if pred_class - 1 >= 0 else []) + 
                       ([pred_class + 1] if pred_class + 1 < len(pred_proba) else []))]

    # Ensure probability values don't exceed 100%
    confidence_levels = [
        {"bands": one_band, "confidence": min(100.0, float(round(sum(pred_proba[one_band]) * 100, 2)))},
        {"bands": two_band, "confidence": min(100.0, float(round(sum(pred_proba[two_band]) * 100, 2)))}
    ]
    
    if pred_class > 0 and pred_class < len(pred_proba) - 1:
        confidence_levels.append({
            "bands": three_band,
            "confidence": min(100.0, float(round(sum(pred_proba[three_band]) * 100, 2)))
        })
    
    return confidence_levels

# Generate user persona
def generate_user_persona(predictions: Dict):
    # Convert prediction levels to more understandable descriptions
    salary_level = predictions['salary'][0]['bands'][0]
    gambling_level = predictions['gambling'][0]['bands'][0]
    operator_count = predictions['operator'][0]['bands'][0]
    frequency_level = predictions['frequency'][0]['bands'][0]
    
    salary_desc = ["Low income", "Medium income", "High income"][min(salary_level, 2)]
    gambling_desc = ["Low stakes", "Medium stakes", "High stakes"][min(gambling_level, 2)]
    operator_desc = f"Uses {operator_count} gambling platforms"
    frequency_desc = ["Occasional participation", "Regular participation", "Frequent participation"][min(frequency_level, 2)]
    
    prompt = f"""Please generate a brief user persona description based on the following characteristics (respond in English):
    - Income level: {salary_desc}
    - Betting amount: {gambling_desc}
    - Gambling platforms: {operator_desc}
    - Participation frequency: {frequency_desc}
    
    Please describe this user's characteristics and possible behavior patterns in 2-3 sentences. Focus on the user's gambling behavior and risk level."""
    
    try:
        # Use OpenAI API to generate user persona
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional user behavior analyst, focusing on analyzing user gambling behavior patterns and risk assessment. Please use professional but easy-to-understand language to describe user characteristics."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Persona generation error: {str(e)}", exc_info=True)
        return "Unable to generate user persona at this time. Please try again later."

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        logger.info(f"Received prediction request: {input_data}")
        
        # Prepare input data
        df = pd.DataFrame({
            'age_band': [input_data.age_band],
            'gender': [input_data.gender.upper()],
            'lsoa_income_rank': [input_data.lsoa_income_rank]
        })
        
        country = input_data.country.lower()
        if country not in COUNTRIES:
            raise HTTPException(status_code=400, detail="Invalid country")

        logger.info(f"Loading models for country: {country}")
        # Load all models
        components = {}
        for predictor_type in ['salary', 'gambling', 'operator', 'frequency']:
            components[(country, predictor_type)] = load_components(country, predictor_type)

        logger.info("Processing gender encoding")
        # Process gender encoding
        df['gender'] = components[(country, 'salary')][0]['gender'].transform(df['gender'])
        
        logger.info("Scaling input data")
        # Standardize data
        input_scaled = components[(country, 'salary')][1].transform(
            df[['age_band', 'gender', 'lsoa_income_rank']].values
        )

        # Predict salary
        logger.info("Predicting salary")
        salary_proba = components[(country, 'salary')][2].predict_proba(input_scaled)[0]
        
        # Predict gambling deposit
        logger.info("Predicting gambling")
        gambling_proba = components[(country, 'gambling')][2].predict_proba(input_scaled)[0]
        gambling_class = int(np.argmax(gambling_proba))  # Convert to Python int type

        # Combine gambling prediction results
        gambling_pred_reshaped = np.array([[gambling_class]])
        input_with_gambling = np.hstack([input_scaled, gambling_pred_reshaped])
        
        # Predict operator count
        logger.info("Predicting operator")
        operator_proba = components[(country, 'operator')][2].predict_proba(input_with_gambling)[0]
        
        # Predict gambling frequency
        logger.info("Predicting frequency")
        frequency_proba = components[(country, 'frequency')][2].predict_proba(input_with_gambling)[0]

        # Organize prediction results
        predictions = {
            "salary": calculate_confidence_levels(salary_proba),
            "gambling": calculate_confidence_levels(gambling_proba),
            "operator": calculate_confidence_levels(operator_proba),
            "frequency": calculate_confidence_levels(frequency_proba)
        }
        
        logger.info("Generating user persona")
        # Generate user persona
        persona = generate_user_persona(predictions)

        logger.info("Returning prediction results")
        return JSONResponse(content={
            "predictions": predictions,
            "persona": persona
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query_data")
async def query_data(query_input: DataQueryInput):
    """
    Process a natural language query to extract data from the CSV file
    and perform calculations based on the query
    """
    try:
        logger.info(f"Received data query: {query_input.query}")
        
        # Extract query parameters from the natural language query
        params = data_processor.extract_query_params(query_input.query)
        
        # If no parameters were extracted, return an error
        if not params:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not extract query parameters from the input"}
            )
        
        # Query the data based on the extracted parameters
        filtered_df = data_processor.query_data(params)
        
        # If no data was found, return an error
        if filtered_df.empty:
            return JSONResponse(
                status_code=404,
                content={"error": "No data found matching the query parameters"}
            )
        
        # Perform calculation if specified
        result = {}
        if 'calculation' in params:
            # Get the column to perform calculation on (default to 'amount')
            column = 'amount'  # Default column
            
            # Get the group by column if specified
            group_by = params.get('group_by', None)
            
            # Perform the calculation
            calc_result = formula_index.calculate(
                filtered_df, 
                params['calculation'], 
                column=column, 
                group_by=group_by
            )
            
            result = {
                "query": query_input.query,
                "parameters": params,
                "calculation_result": calc_result,
                "row_count": len(filtered_df)
            }
        else:
            # Return the filtered data without calculation
            result = {
                "query": query_input.query,
                "parameters": params,
                "data": filtered_df.head(50).to_dict(orient='records'),  # Limit to 50 rows
                "row_count": len(filtered_df)
            }
        
        logger.info(f"Query processed successfully")
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Error processing data query: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing data query: {str(e)}"}
        )

@app.post("/chat")
async def chat_post(chat_input: ChatInput):
    try:
        logger.info(f"Received chat message: {chat_input.message}")
        if not api_key:
            logger.error("API key not found in environment variables")
            raise HTTPException(status_code=500, detail="API key not configured")
        
        # 添加调试日志
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Sample CSV path: {os.path.abspath('sample_txns.csv')}")
        logger.info(f"Sample CSV exists: {os.path.exists('sample_txns.csv')}")
        
        # 检查是否是数据查询
        is_data_query = data_processor.is_data_query(chat_input.message)
        logger.info(f"Is data query: {is_data_query}")
        
        if is_data_query:
            try:
                # Extract query parameters
                params = data_processor.extract_query_params(chat_input.message)
                logger.info(f"Extracted parameters: {params}")
                
                # If parameters were extracted, process as a data query
                if params:
                    logger.info(f"Processing as data query with parameters: {params}")
                    
                    # 检查数据处理器状态
                    if data_processor.df is None:
                        logger.error("DataFrame is None, attempting to reload data")
                        data_processor.load_data()
                    
                    logger.info(f"DataFrame shape before query: {data_processor.df.shape if data_processor.df is not None else 'None'}")
                    
                    # Query the data
                    filtered_df = data_processor.query_data(params)
                    logger.info(f"Filtered DataFrame shape: {filtered_df.shape}")
                    logger.info(f"Filtered DataFrame sample: {filtered_df.head(3).to_dict() if not filtered_df.empty else 'Empty'}")
                    
                    # Prepare result based on the query
                    query_result = {}
                    if 'calculation' in params:
                        # Get the column to perform calculation on (default to 'amount')
                        column = params.get('column', 'amount')
                        
                        # Get the group by column if specified
                        group_by = params.get('group_by', None)
                        
                        logger.info(f"Performing calculation: {params['calculation']} on column: {column}, group_by: {group_by}")
                        
                        # Perform the calculation
                        calc_result = formula_index.calculate(
                            filtered_df, 
                            params['calculation'], 
                            column=column, 
                            group_by=group_by
                        )
                        
                        logger.info(f"Calculation result: {calc_result}")
                        
                        if calc_result.get("error"):
                            return JSONResponse(content={"response": f"计算出错：{calc_result['error']}"})
                        
                        if calc_result["result_type"] == "single":
                            if params['calculation'] == 'sum':
                                response = f"总金额为: {calc_result['result']:.2f}"
                            elif params['calculation'] == 'mean':
                                response = f"平均金额为: {calc_result['result']:.2f}"
                            elif params['calculation'] == 'max':
                                record = calc_result.get('record', {})
                                date = record.get('dateObj', '未知日期')
                                response = f"最大金额为: {calc_result['result']:.2f}，发生在 {date}"
                            elif params['calculation'] == 'min':
                                record = calc_result.get('record', {})
                                date = record.get('dateObj', '未知日期')
                                response = f"最小金额为: {calc_result['result']:.2f}，发生在 {date}"
                            elif params['calculation'] == 'count':
                                response = f"交易笔数为: {calc_result['result']}"
                            elif params['calculation'] == 'median':
                                response = f"中位数金额为: {calc_result['result']:.2f}"
                            elif params['calculation'] == 'std':
                                response = f"标准差为: {calc_result['result']:.2f}"
                            elif params['calculation'] == 'variance':
                                response = f"方差为: {calc_result['result']:.2f}"
                            elif params['calculation'] == 'range':
                                response = f"金额范围为: {calc_result['result']:.2f}，最大值: {calc_result['max']:.2f}，最小值: {calc_result['min']:.2f}"
                            else:
                                response = f"计算结果为: {calc_result['result']:.2f}"
                        else:
                            # 处理分组结果
                            group_results = calc_result.get('results', [])
                            if group_results:
                                if params['calculation'] == 'sum':
                                    response = f"按 {params['group_by']} 分组的总金额如下:\n"
                                    for item in group_results:
                                        group_value = item[params['group_by']]
                                        amount = item[params['column']]
                                        response += f"- {group_value}: {amount:.2f}\n"
                                elif params['calculation'] == 'mean':
                                    response = f"按 {params['group_by']} 分组的平均金额如下:\n"
                                    for item in group_results:
                                        group_value = item[params['group_by']]
                                        amount = item[params['column']]
                                        response += f"- {group_value}: {amount:.2f}\n"
                                elif params['calculation'] == 'count':
                                    response = f"按 {params['group_by']} 分组的交易笔数如下:\n"
                                    for item in group_results:
                                        group_value = item[params['group_by']]
                                        count = item[params['column']]
                                        response += f"- {group_value}: {count}\n"
                                elif params['calculation'] == 'max':
                                    response = f"按 {params['group_by']} 分组的最大金额如下:\n"
                                    for item in group_results:
                                        group_value = item[params['group_by']]
                                        amount = item[params['column']]
                                        response += f"- {group_value}: {amount:.2f}\n"
                                elif params['calculation'] == 'min':
                                    response = f"按 {params['group_by']} 分组的最小金额如下:\n"
                                    for item in group_results:
                                        group_value = item[params['group_by']]
                                        amount = item[params['column']]
                                        response += f"- {group_value}: {amount:.2f}\n"
                                else:
                                    response = f"按 {params['group_by']} 分组的计算结果如下:\n"
                                    for item in group_results:
                                        group_value = item[params['group_by']]
                                        amount = item[params['column']]
                                        response += f"- {group_value}: {amount:.2f}\n"
                            else:
                                response = f"分组计算未返回任何结果"
                        
                        query_result = {
                            "parameters": params,
                            "calculation_result": calc_result,
                            "row_count": len(filtered_df)
                        }
                    else:
                        # Return comprehensive data summary when no specific calculation is requested
                        max_record = filtered_df.loc[filtered_df['amount'].idxmax()]
                        min_record = filtered_df.loc[filtered_df['amount'].idxmin()]
                        
                        query_result = {
                            "parameters": params,
                            "row_count": len(filtered_df),
                            "data_summary": {
                                "total_amount": float(filtered_df['amount'].sum()),
                                "average_amount": float(filtered_df['amount'].mean()),
                                "max_amount": {
                                    "value": float(max_record['amount']),
                                    "date": max_record['dateObj']
                                },
                                "min_amount": {
                                    "value": float(min_record['amount']),
                                    "date": min_record['dateObj']
                                },
                                "transaction_count": len(filtered_df)
                            }
                        }
                        
                        response = (
                            f"在指定时间范围内找到 {len(filtered_df)} 笔交易：\n"
                            f"总金额：{query_result['data_summary']['total_amount']:.2f}\n"
                            f"平均金额：{query_result['data_summary']['average_amount']:.2f}\n"
                            f"最大金额：{query_result['data_summary']['max_amount']['value']:.2f} (发生在 {query_result['data_summary']['max_amount']['date']})\n"
                            f"最小金额：{query_result['data_summary']['min_amount']['value']:.2f} (发生在 {query_result['data_summary']['min_amount']['date']})"
                        )
                        
                        logger.info(f"Data summary: {query_result['data_summary']}")
                    
                    # Convert the query result to a string for OpenAI
                    query_result_str = json.dumps(query_result, indent=2)
                    logger.info(f"Query result for OpenAI: {query_result_str}")
                    
                    # Use OpenAI to generate a response based on the query result
                    messages = [
                        {"role": "system", "content": "You are a financial data analyst assistant. You help users analyze transaction data and provide insights. When responding to queries about transaction data, use the provided data analysis results to formulate your response. Present the information in a clear, concise manner. If the data shows interesting patterns or insights, point them out. IMPORTANT: Always include the actual numerical results in your response. If the user query is in Chinese, respond in Chinese; otherwise respond in English."},
                        {"role": "user", "content": chat_input.message},
                        {"role": "system", "content": f"Here is the result of the data analysis based on the user's query:\n\n{query_result_str}\n\nMake sure to include the specific numerical results in your response. If the calculation was a sum or average, explicitly mention the value. DO NOT say you don't have access to the data - you DO have the data and the results are provided above."}
                    ]
                    
                    logger.info("Sending request to OpenAI API")
                    
                    # Make request to OpenAI API
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        max_tokens=500,
                        temperature=0.7,
                        stream=True
                    )
                    
                    # Stream the response
                    async def generate():
                        for chunk in response:
                            if chunk.choices and len(chunk.choices) > 0:
                                delta = chunk.choices[0].delta
                                if hasattr(delta, 'content') and delta.content:
                                    yield f"data: {json.dumps({'choices': [{'delta': {'content': delta.content}}]})}\n\n"
                        yield "data: [DONE]\n\n"
                    
                    logger.info("Successfully streaming response for data query")
                    return StreamingResponse(generate(), media_type="text/event-stream")
                else:
                    logger.warning("No parameters extracted from query")
                
            except Exception as e:
                logger.error(f"Error processing data query in chat: {str(e)}", exc_info=True)
                # Continue with normal chat if data query processing fails
        
        try:
            logger.info("Making request to OpenAI API for regular chat...")
            
            # Use OpenAI API to handle chat request (streaming response)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional gambling consultant, focusing on responsible gambling and risk management advice. Please answer the user's questions in English."},
                    {"role": "user", "content": chat_input.message}
                ],
                max_tokens=500,
                temperature=0.7,
                stream=True
            )
            
            # Convert streaming response to SSE format
            async def generate():
                for chunk in response:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            yield f"data: {json.dumps({'choices': [{'delta': {'content': delta.content}}]})}\n\n"
                yield "data: [DONE]\n\n"
            
            logger.info("Successfully streaming response from OpenAI API")
            return StreamingResponse(generate(), media_type="text/event-stream")
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"response": f"Sorry, I cannot answer your question right now. Please try again later. Error: {str(e)}"}
            )
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"response": f"The system is experiencing some issues. Please try again later. Error: {str(e)}"}
        )

@app.get("/chat")
async def chat_get(message: str):
    try:
        logger.info(f"Received chat message via GET: {message}")
        if not api_key:
            logger.error("API key not found in environment variables")
            raise HTTPException(status_code=500, detail="API key not configured")
        
        try:
            logger.info("Making request to OpenAI API...")
            
            # Use OpenAI API to handle chat request (non-streaming response)
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional gambling consultant, focusing on responsible gambling and risk management advice. Please answer the user's questions in English."},
                    {"role": "user", "content": message}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            logger.info("Successfully received response from OpenAI API")
            return {"response": completion.choices[0].message.content}
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}", exc_info=True)
            return {"response": "Sorry, I cannot answer your question right now. Please try again later."}
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}", exc_info=True)
        return {"response": "The system is experiencing some issues. Please try again later."}

@app.get("/test_query")
async def test_query():
    """
    测试端点，用于验证数据查询功能是否正常工作
    """
    try:
        logger.info("Testing data query functionality")
        
        # 检查数据处理器状态
        if data_processor.df is None:
            logger.error("DataFrame is None, attempting to reload data")
            data_processor.load_data()
        
        # 打印数据处理器状态
        logger.info(f"DataFrame shape: {data_processor.df.shape if data_processor.df is not None else 'None'}")
        logger.info(f"DataFrame columns: {data_processor.df.columns.tolist() if data_processor.df is not None else 'None'}")
        
        # 测试查询：2025-12-31的所有交易
        test_params = {
            'start_date': '2025-12-31',
            'end_date': '2025-12-31'
        }
        
        # 执行查询
        filtered_df = data_processor.query_data(test_params)
        logger.info(f"Test query returned {len(filtered_df)} rows")
        
        # 计算平均值
        if not filtered_df.empty:
            mean_result = formula_index.calculate(filtered_df, 'mean', column='amount')
            sum_result = formula_index.calculate(filtered_df, 'sum', column='amount')
            
            return {
                "status": "success",
                "message": "Data query test successful",
                "data_loaded": True,
                "row_count": len(filtered_df),
                "mean_result": mean_result,
                "sum_result": sum_result,
                "sample_data": filtered_df.head(5).to_dict(orient='records')
            }
        else:
            return {
                "status": "warning",
                "message": "Query returned no data",
                "data_loaded": data_processor.df is not None,
                "test_params": test_params
            }
    
    except Exception as e:
        logger.error(f"Error in test query: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Error in test query: {str(e)}",
            "data_loaded": data_processor.df is not None
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 