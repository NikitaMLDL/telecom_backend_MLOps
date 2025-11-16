import uuid
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi import UploadFile, File
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import os

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://94.228.117.198:9000"

templates = Jinja2Templates(directory="templates")


class Model:
    def __init__(self, experiment_name: str, alias: str = "staging"):
        mlflow.set_tracking_uri("http://94.228.117.198:5000")
        self.model = mlflow.pyfunc.load_model(model_uri=f"models:/{experiment_name}@{alias}")

    def predict(self, df: pd.DataFrame):
        return self.model.predict(df)


app = FastAPI(title="Churn Prediction Inference")

churn_model = Model(experiment_name="ChurnPipeline", alias="staging")


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})


@app.post("/predict", response_class=HTMLResponse)
async def predict_file(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        # Предсказание churn
        preds = churn_model.predict(df)

        df["churn_pred"] = preds

        # Сохраняем полный файл
        filename = f"pred_{uuid.uuid4().hex}.csv"
        os.makedirs("results", exist_ok=True)
        path = os.path.join("results", filename)
        df.to_csv(path, index=False)
        app.state.last_result_file = path

        # Берём preview (первые 5 строк)
        preview = df.head(5).to_dict(orient="records")

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "preview": preview}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download")
def download():
    path = app.state.last_result_file
    if path and os.path.exists(path):
        return FileResponse(path, filename="predictions.csv")
    return {"error": "No predictions available yet"}