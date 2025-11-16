from fastapi import FastAPI, HTTPException, Request
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


app = FastAPI(title="Churn Prediction Inference")

instrumentator = Instrumentator()

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

        # Сохраняем полный файл
        filename = f"{uuid.uuid4().hex}.csv"
        csv_bytes = df.to_csv(index=False).encode("utf-8")

        s3.put_object(
            Bucket=S3_BUCKET,
            Key=filename,
            Body=csv_bytes,
            ContentType="text/csv"
        )

        download_url = s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": S3_BUCKET, "Key": filename},
            ExpiresIn=3600
        )

        preview = df.head(5).to_dict(orient="records")

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "preview": preview,
                "download_url": download_url
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
