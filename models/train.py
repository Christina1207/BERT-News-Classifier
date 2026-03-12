import argparse, random, os
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score
import wandb

# make our preprocessing importable
import sys; sys.path.insert(0, ".")
from data.preprocess import load_and_prepare


def set_seed(seed: int):
    """Seed all RNG sources for reproducibility."""
    random.seed(seed)                   # Python stdlib
    np.random.seed(seed)                # NumPy
    torch.manual_seed(seed)             # PyTorch CPU
    torch.cuda.manual_seed_all(seed)    # PyTorch GPU
    os.environ["PYTHONHASHSEED"] = str(seed)
  

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": f1}


def main(args):
    set_seed(args.seed)

    wandb.init(
        project="bert-news-classifier",
        config=vars(args),
        name=f"lr{args.lr}_ep{args.epochs}_bs{args.batch_size}"
    )

    dataset = load_and_prepare(seed=args.seed)

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=4
    )

    training_args = TrainingArguments(
        output_dir="./checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        learning_rate=args.lr,            # most impactful HP
        weight_decay=0.01,               # L2 regularisation
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("./checkpoints/best")
    print("Training complete. Best model saved to ./checkpoints/best")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()
    main(args)