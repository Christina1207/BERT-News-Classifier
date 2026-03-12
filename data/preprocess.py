import re
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128   # fits free Colab VRAM; avoids OOM on T4

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def clean_text(text: str) -> str:
    """Remove HTML entities and normalise whitespace."""
    text = re.sub(r'&[a-z]+;', ' ', text)   # &amp; &lt; etc.
    text = re.sub(r'&#\d+;', ' ', text)       # numeric entities
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_batch(batch):
    """Tokenize a batch; truncate to MAX_LENGTH, pad to same length."""
    cleaned = [clean_text(t) for t in batch["text"]]
    return tokenizer(
        cleaned,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )


def load_and_prepare(val_frac: float = 0.1, seed: int = 42):
    """Load AG News, split val from train, tokenize all splits."""
    raw = load_dataset("ag_news")

    # carve validation from training set
    split = raw["train"].train_test_split(
        test_size=val_frac, seed=seed, stratify_by_column="label"
    )
    raw["train"] = split["train"]
    raw["validation"] = split["test"]

    tokenized = raw.map(tokenize_batch, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch",
        columns=["input_ids", "attention_mask", "labels"])
    return tokenized


if __name__ == "__main__":
    ds = load_and_prepare()
    print(ds)
    print("Sample:", ds["train"][0])