import sys; sys.path.insert(0, ".")
from data.preprocess import clean_text, tokenize_batch, MAX_LENGTH


def test_html_entity_removal():
    """HTML entities must be stripped before tokenisation."""
    result = clean_text("AT&amp;T reports Q3 earnings &lt;b&gt;")
    assert "&" not in result
    assert "AT" in result   # content preserved


def test_tokenization_length():
    """All tokenised sequences must be exactly MAX_LENGTH tokens."""
    batch = {"text": [
        "Breaking news: global markets fall sharply amid recession fears.",
        "Short text.",
        "x " * 300,   # very long — should truncate
    ]}
    out = tokenize_batch(batch)
    for ids in out["input_ids"]:
        assert len(ids) == MAX_LENGTH, \
            f"Expected {MAX_LENGTH} tokens, got {len(ids)}"


def test_attention_mask_shape():
    """Attention masks must match input_ids shape."""
    batch = {"text": ["Test sentence one.", "Test sentence two."]}
    out = tokenize_batch(batch)
    assert len(out["input_ids"]) == len(out["attention_mask"])
    assert len(out["input_ids"][0]) == len(out["attention_mask"][0])