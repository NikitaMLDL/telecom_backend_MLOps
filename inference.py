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
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing import TypedDict
import requests


app = FastAPI(title="Churn Prediction Inference")

app.mount("/static", StaticFiles(directory="static"), name="static")

instrumentator = Instrumentator()


YOUR_LOGIN = os.getenv("YOUR_LOGIN")
YOUR_PASSWORD = os.getenv("YOUR_PASSWORD")
proxy_host = "196.19.5.245"
proxy_port = "8000"

proxy_url = f"http://{YOUR_LOGIN}:{YOUR_PASSWORD}@{proxy_host}:{proxy_port}"

model_proxies = {
    "http": proxy_url,
    "https": proxy_url
}


AI_STUDIO_KEY = os.getenv("AI_STUDIO_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=AI_STUDIO_KEY,
    temperature=0.4,
    max_output_tokens=200,
    transport="rest"
)


def query_model(prompt):
    session = requests.Session()
    session.proxies.update(model_proxies)

    # try:
    response = llm.generate(
        [{"role": "user", "content": prompt}],
        transport="rest",
        requests_session=session
    )
    print("Ответ: ", response.generations[0][0].text)
    return response.generations[0][0].text
    # except Exception:
    #     return None


def interpretation_node(state):
    df = state["df"]
    sample = df.head(5).to_dict(orient="records")

    prompt = f"""
Ты эксперт по удержанию клиентов. У тебя есть результаты предсказания churn:
{sample}

Опиши в 3 коротких пунктах:
1. Какие признаки влияют на уход клиентов.
2. Конкретные стратегии удержания.
3. Короткое резюме.
"""

    result = query_model(prompt)
    return {"interpretation": result}


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


@app.post("/predict", response_class=HTMLResponse)
async def predict_file(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        start_time = time.time()
        preds = churn_model.predict(df)
        elapsed = time.time() - start_time

        PREDICTION_LATENCY.set(elapsed)
        df["churn_pred"] = preds
        result = workflow.invoke({"df": df})

        # Сохраняем полный файл
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
                "recommendations": result["interpretation"]
            }
        )
    except Exception as e:
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
