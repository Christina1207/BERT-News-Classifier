from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import time

app = FastAPI(title="News Classifier", version="1.0")

# Load model once at startup (not per request)
clf = pipeline(
    "text-classification",
    model="./checkpoints/best",
    device=-1,    # -1 = CPU for serving; change to 0 for GPU
    top_k=1,
)

LABEL_MAP = {
    "LABEL_0": "World",
    "LABEL_1": "Sports",
    "LABEL_2": "Business",
    "LABEL_3": "Sci/Tech",
}


class TextInput(BaseModel):
    text: str


class PredictionOut(BaseModel):
    label: str
    score: float
    latency_ms: float


@app.post("/predict", response_model=PredictionOut)
def predict(body: TextInput):
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="Empty input")
    t0 = time.perf_counter()
    result = clf(body.text)[0]
    latency = (time.perf_counter() - t0) * 1000
    return {
        "label": LABEL_MAP.get(result["label"], result["label"]),
        "score": round(result["score"], 4),
        "latency_ms": round(latency, 1),
    }


@app.get("/health")
def health(): return {"status": "ok"}