from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi import UploadFile, File
from fastapi.responses import FileResponse
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import mlflow.pyfunc
import pandas as pd
import time
import os
import boto3
import uuid
from langgraph.graph import StateGraph, END
from typing import TypedDict
import openai
import json


app = FastAPI(title="Churn Prediction Inference")

app.mount("/static", StaticFiles(directory="static"), name="static")

instrumentator = Instrumentator()

API_KEY = os.getenv("API_KEY")
client = openai.OpenAI(
    api_key=API_KEY,
    base_url="https://api.vsellm.ru/v1"
)


def interpretation_node(df):
    compressed = df['df'].head(3)

    messages = [
        {"role": "system", "content": "Делай выводы кратко"},
        {"role": "user", "content": "Проанализируй churn данные."},
        {"role": "user", "content": f"Данные: {compressed}"}
    ]

    response = client.chat.completions.create(
        model="openai/gpt-4.1-nano",
        messages=messages,
        max_tokens=100
    )

    return {"interpretation": response.choices[0].message.content}


class PipelineState(TypedDict):
    df: any
    interpretation: str


graph = StateGraph(PipelineState)
graph.add_node("interpret", interpretation_node)

graph.set_entry_point("interpret")
graph.add_edge("interpret", END)

workflow = graph.compile()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
S3_ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL")
S3_BUCKET = "churn-results"

PREDICTION_LATENCY = Gauge(
    "model_prediction_latency_seconds",
    "Time spent performing model inference (seconds)"
)


instrumentator.instrument(app).expose(app, endpoint="/metrics")
templates = Jinja2Templates(directory="templates")


class Model:
    def __init__(self):
        self.model = None

    def load(self, experiment_name: str, alias: str):
        self.model = mlflow.pyfunc.load_model(model_uri=f"models:/{experiment_name}@{alias}")

    def predict(self, df: pd.DataFrame):
        return self.model.predict(df)


churn_model = Model()
churn_model.load("ChurnPipeline", "staging")
s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})


def prepare_dashboard_data(df):
    """
    Готовит данные для 4-х графиков дашборда на основе DataFrame.
    """
    # 0. Нормализация данных
    # Приводим предсказания к единому формату (1/0 или 'yes'/'no')
    # Предполагаем, что 'yes'/'1' это отток.
    df['is_churn'] = df['churn_pred'].astype(str).str.lower().isin(['yes', '1', 'true'])
    
    # Приводим планы к 'yes'/'no' (на случай если там 1/0)
    df['international_plan'] = df['international_plan'].astype(str).str.lower().replace({'1': 'yes', '0': 'no'})
    df['voice_mail_plan'] = df['voice_mail_plan'].astype(str).str.lower().replace({'1': 'yes', '0': 'no'})

    # --- ГРАФИК 1: Доля оттока ---
    churn_counts = df['is_churn'].value_counts()
    count_churn = int(churn_counts.get(True, 0))
    count_stay = int(churn_counts.get(False, 0))
    
    # --- ГРАФИК 2: International Plan (Crosstab) ---
    # Группируем: [Plan(no/yes)] -> [Churn(False/True)]
    ct_intl = pd.crosstab(df['international_plan'], df['is_churn'])
    
    # Безопасное извлечение (в файле может не быть людей с планом)
    def get_val(crosstab, plan_val, churn_val):
        try:
            return int(crosstab.loc[plan_val, churn_val])
        except KeyError:
            return 0

    intl_data = {
        'labels': ['Нет плана', 'Есть план'],
        'stayed': [get_val(ct_intl, 'no', False), get_val(ct_intl, 'yes', False)],
        'churned': [get_val(ct_intl, 'no', True), get_val(ct_intl, 'yes', True)]
    }

    # --- ГРАФИК 3: Voice Mail Plan (Crosstab) ---
    ct_vmail = pd.crosstab(df['voice_mail_plan'], df['is_churn'])
    
    vmail_data = {
        'labels': ['Нет VMail', 'Есть VMail'],
        'stayed': [get_val(ct_vmail, 'no', False), get_val(ct_vmail, 'yes', False)],
        'churned': [get_val(ct_vmail, 'no', True), get_val(ct_vmail, 'yes', True)]
    }

    # --- ГРАФИК 4: Calls vs Churn (Grouped Bar) ---
    # Получаем распределение звонков для ушедших и оставшихся
    churners = df[df['is_churn'] == True]
    stayers = df[df['is_churn'] == False]
    
    # Считаем value_counts для звонков (от 0 до 9+)
    calls_churn = churners['number_customer_service_calls'].value_counts().sort_index()
    calls_stay = stayers['number_customer_service_calls'].value_counts().sort_index()
    
    call_labels = [str(i) for i in range(10)] # "0"..."9"
    churn_values = []
    stay_values = []
    
    for i in range(9):
        churn_values.append(int(calls_churn.get(i, 0)))
        stay_values.append(int(calls_stay.get(i, 0)))
    
    # 9+ звонков
    mask_churn_9plus = churners['number_customer_service_calls'] >= 9
    mask_stay_9plus = stayers['number_customer_service_calls'] >= 9
    
    churn_values.append(int(mask_churn_9plus.sum()))
    stay_values.append(int(mask_stay_9plus.sum()))
    call_labels[-1] = "9+"

    calls_data = {
        'labels': call_labels,
        'churned': churn_values,
        'stayed': stay_values
    }

    # Собираем итоговый словарь
    return {
        'total_rows': len(df),
        'churn_rate': round((count_churn / len(df) * 100), 1) if len(df) > 0 else 0,
        'charts': {
            'ratio': {'labels': ['Остались', 'Ушли'], 'data': [count_stay, count_churn]},
            'intl': intl_data,
            'vmail': vmail_data,
            'calls': calls_data
        }
    }


@app.post("/predict", response_class=HTMLResponse)
async def predict_file(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        start_time = time.time()
        preds = churn_model.predict(df)
        elapsed = time.time() - start_time

        PREDICTION_LATENCY.set(elapsed)
        df["churn_pred"] = preds

        # Запуск LangGraph Workflow
        result = workflow.invoke({"df": df})

        # --- НОВОЕ: Подготовка данных для графиков ---
        # Вызываем функцию обработки (код которой был выше)
        dash_data = prepare_dashboard_data(df)

        # Сериализуем в JSON-строку для передачи в JavaScript
        dashboard_json = json.dumps(dash_data)
        # ---------------------------------------------

        filename = f"{uuid.uuid4().hex}.csv"
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        s3.put_object(
            Bucket=S3_BUCKET,
            Key=filename,
            Body=csv_bytes,
            ContentType="text/csv"
        )

        # download_url = s3.generate_presigned_url(
        #     ClientMethod="get_object",
        #     Params={"Bucket": S3_BUCKET, "Key": filename},
        #     ExpiresIn=3600
        # )

        preview = df.head(5).to_dict(orient="records")

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "preview": preview,
                "recommendations": result["interpretation"],
                "uploaded_filename": filename,
                "dashboard_data": dashboard_json
            }
        )
    except Exception as e:
        # Логируем ошибку, чтобы проще было отлаживать
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/webhook/")
async def receive_webhook(req: Request):
    payload = await req.json()
    print("Before load:", id(churn_model.model))
    if payload.get("entity") == "model_version_alias" and payload.get("action") == "created":
        data = payload.get("data", {})
        if data.get("alias") == "prod":
            churn_model.load(data.get("name"), "prod")
            print("After load:", id(churn_model.model))
            print(f"Model '{data.get('name')}' loaded with alias 'prod'")
    return {"status": "success"}
