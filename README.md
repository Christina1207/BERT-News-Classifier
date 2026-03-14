# BERT New Classifier
 
Fine-tuning `bert-base-uncased` for 4-class news topic classification on the [AG News](https://huggingface.co/datasets/ag_news) dataset. Achieves **94.2% accuracy** and **94.1% macro-F1** on the held-out test set. Includes a FastAPI inference server.
 
---
 
## Table of Contents
 
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [API Usage](#api-usage)
- [Results](#results)
- [License](#license)
 
---
 
## Overview
 
| | |
|---|---|
| **Task** | Multi-class text classification (4 classes) |
| **Model** | `bert-base-uncased` (110M params) |
| **Dataset** | AG News — 120k train / 7.6k test |
| **Classes** | World · Sports · Business · Sci/Tech |
| **Best val loss** | 0.1823 @ epoch 3 |
| **Test accuracy** | 94.2% |
| **Inference latency** | ~45ms / request (CPU) |
| **Tracking** | Weights & Biases |
 
---
 
## Repository Structure
 
```
bert-news-classifier/
├── data/
│   └── preprocess.py        # AG News loading, cleaning, tokenization
├── models/
│   └── train.py             # Fine-tuning script (BERT + HuggingFace Trainer)
├── serve/
│   └── app.py               # FastAPI inference endpoint
├── tests/
│   └── test_dataset.py      # Unit tests for preprocessing
├── checkpoints/             # Saved model checkpoints (gitignored)
├── requirements.txt
└── README.md
```
 
---
 
## Setup & Installation
 
**Requirements:** Python 3.10+
 
```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/bert-news-classifier
cd bert-news-classifier
 
# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate         # Windows
 
# 3. Install dependencies
pip install -r requirements.txt
```
 
---
 
## Dataset
 
[AG News](https://huggingface.co/datasets/ag_news) is a news topic classification benchmark with 4 categories.
 
| Split | Samples |
|---|---|
| Train | 108,000 |
| Validation | 12,000 (10% stratified split from train, seed=42) |
| Test | 7,600 (official) |
 
**Preprocessing steps:**
- Strip HTML entities (`&amp;`, `&#160;`, etc.)
- Normalise whitespace
- Tokenize with `bert-base-uncased` tokenizer, truncate to 128 tokens
 
Run preprocessing standalone to verify:
 
```bash
python data/preprocess.py
```
 
---
 
## Training
 
Training was done on a Google Colab T4 GPU (15.8GB VRAM, CUDA 12.2). Runtime: ~90 minutes.
 
```bash
python models/train.py --lr 2e-5 --epochs 3 --batch_size 32 --seed 42
```
 
**Key hyperparameters:**
 
| Parameter | Value | Notes |
|---|---|---|
| `learning_rate` | `2e-5` | Most impactful HP; tuned over [1e-5, 5e-5] |
| `weight_decay` | `0.01` | L2 regularisation via AdamW |
| `warmup_ratio` | `0.1` | Linear warmup over first 10% of steps |
| `max_length` | `128` | Covers 95th percentile of AG News lengths |
| `seed` | `42` | Set across Python, NumPy, PyTorch CPU+GPU |
 
Experiment runs are tracked with [Weights & Biases](https://wandb.ai).
 
Checkpoints are saved per epoch under `./checkpoints/`. The best checkpoint (lowest `eval_loss`) is saved to `./checkpoints/best`.
 
---
 
## Evaluation
 
After training, run the evaluation script from a Python session or Colab cell:
 
```python
from transformers import Trainer, AutoModelForSequenceClassification
from data.preprocess import load_and_prepare
import numpy as np
from sklearn.metrics import classification_report
 
dataset = load_and_prepare()
model   = AutoModelForSequenceClassification.from_pretrained("./checkpoints/best")
trainer = Trainer(model=model)
 
preds_out = trainer.predict(dataset["test"])
y_pred    = np.argmax(preds_out.predictions, axis=-1)
y_true    = preds_out.label_ids
 
print(classification_report(y_true, y_pred,
      target_names=["World", "Sports", "Business", "Sci/Tech"]))
```
 
To run unit tests:
 
```bash
pytest tests/test_dataset.py -v
```
 
---
 
## API Usage
 
Start the inference server locally:
 
```bash
uvicorn serve.app:app --host 0.0.0.0 --port 8000 --reload
```
 
**Endpoints:**
 
### `POST /predict`
 
Classify a news headline or article snippet.
 
**Request:**
```json
{ "text": "NASA launches new telescope to study black holes" }
```
 
**Response:**
```json
{
  "label": "Sci/Tech",
  "score": 0.9412,
  "latency_ms": 45.3
}
```
 
**curl example:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "NASA launches new telescope to study black holes"}'
```
 
### `GET /health`
 
```bash
curl http://localhost:8000/health
# {"status": "ok"}
```
 
Interactive docs (Swagger UI) are available at **http://localhost:8000/docs** when the server is running.
 
---
 
## Results
 
| Metric | Value |
|---|---|
| Test Accuracy | ~94.2% |
| Macro F1 | ~94.1% |
| Best eval loss | ~0.1823 |
| Checkpoint | `checkpoints/best` |
 
**Confusion matrix:** The most common error is Business samples misclassified as Sci/Tech (~60 samples out of 1,900). Technology company financial news shares vocabulary with the Sci/Tech category. Concatenating the article description to the headline reduced this confusion by ~30%.
 
---
 
## License
 
This project is licensed under the **Apache 2.0 License** — see [LICENSE](LICENSE) for details.
 
- `bert-base-uncased`: Apache 2.0 ([HuggingFace](https://huggingface.co/bert-base-uncased))
- AG News dataset: public research use
 