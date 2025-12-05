import uuid
import os
import pandas as pd
import mlflow.pyfunc
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import time
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge
import boto3

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing import TypedDict
from dotenv import load_dotenv


load_dotenv()
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://94.228.117.198:9000"


AI_STUDIO_KEY = os.getenv("AI_STUDIO_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=AI_STUDIO_KEY,
    temperature=0.4,
    max_output_tokens=200,
)

templates = Jinja2Templates(directory="templates")


class MLAgent:
    def __init__(self, experiment_name: str, alias: str = "staging"):
        mlflow.set_tracking_uri("http://94.228.117.198:5000")
        self.model = mlflow.pyfunc.load_model(f"models:/{experiment_name}@{alias}")

    def predict(self, df: pd.DataFrame):
        return self.model.predict(df)


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

    result = llm.invoke(prompt)
    return {"interpretation": result.content}


# def evaluation_node(state):
#     recs = state["interpretation"]

#     prompt = f"""
# Проведи оценку рекомендаций:
# {recs}

# Дай корректировки, если что-то нереалистично.
# """

#     result = llm.invoke(prompt)
#     return {"evaluation": result.content}


class PipelineState(TypedDict):
    df: any
    interpretation: str

graph = StateGraph(PipelineState)

graph.add_node("interpret", interpretation_node)
# graph.add_node("evaluate", evaluation_node)

graph.set_entry_point("interpret")
graph.add_edge("interpret", END)
# graph.add_edge("evaluate", END)

workflow = graph.compile()


app = FastAPI(title="Churn Multi-Agent ChatBot")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.state.last_result_file = None

ml_agent = MLAgent("ChurnPipeline", "staging")

instrumentator = Instrumentator()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
S3_ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL")
S3_BUCKET = "churn-results"

PREDICTION_LATENCY = Gauge(
    "model_prediction_latency_seconds",
    "Time spent performing model inference (seconds)"
)


instrumentator.instrument(app).expose(app, endpoint="/metrics")

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict_file(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        start_time = time.time()
        preds = ml_agent.predict(df)
        elapsed = time.time() - start_time

        PREDICTION_LATENCY.set(elapsed)

        df["churn_pred"] = preds
        result = workflow.invoke({"df": df})

        # Сохраняем полный файл
        filename = f"pred_{uuid.uuid4().hex}.csv"
        os.makedirs("results", exist_ok=True)
        path = os.path.join("results", filename)
        df.to_csv(path, index=False)
        app.state.last_result_file = path

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

        # 4️⃣ Возвращаем в чат
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


@app.get("/download")
def download():
    path = app.state.last_result_file
    if path and os.path.exists(path):
        return FileResponse(path, filename="predictions.csv")
    return {"error": "No predictions available yet"}
