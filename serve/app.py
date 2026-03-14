from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time

MODEL_PATH = "./checkpoints/best"
LABEL_MAP  = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

tokenizer = None
model     = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model
    print("Loading tokenizer and model from", MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    print("Model loaded and ready.")
    yield
    print("Shutting down.")

app = FastAPI(title="News Classifier", version="1.0", lifespan=lifespan)

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

    inputs = tokenizer(
        body.text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length",
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    pred_id = int(torch.argmax(logits, dim=-1))
    score   = float(torch.softmax(logits, dim=-1)[0][pred_id])
    latency = (time.perf_counter() - t0) * 1000

    return {
        "label":      LABEL_MAP[pred_id],
        "score":      round(score, 4),
        "latency_ms": round(latency, 1),
    }

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {"status": "ok"}