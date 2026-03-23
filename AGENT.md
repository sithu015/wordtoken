# AGENT.md — AI Coding Agent Instructions

This file provides structured guidance for AI coding agents (GitHub Copilot, Cursor, Claude, etc.) working on this Myanmar NLP API project.

---

## Project Purpose

This is a **FastAPI backend server** that exposes REST API endpoints for:
1. **Myanmar Word Segmentation** — syllable-level Myanmar text → segmented word tokens
2. **POS Tagging** — part-of-speech label assignment per segmented word
3. **Joint Processing** — both tasks performed simultaneously via a single neural model

The underlying model is [`sithu015/XLM-RoBERTa-BiLSTM-CRF-Joint`](https://huggingface.co/sithu015/XLM-RoBERTa-BiLSTM-CRF-Joint), a fine-tuned XLM-RoBERTa with BiLSTM + CRF head for joint sequence labeling.

---

## Architecture Decisions

| Decision                        | Rationale                                                        |
|---------------------------------|------------------------------------------------------------------|
| FastAPI over Flask              | Async support, automatic OpenAPI docs, Pydantic validation       |
| Model loaded once at startup    | Avoid per-request cold start; store in `app.state`               |
| CRF decoding on CPU by default  | Inference is fast enough; GPU optional via `DEVICE=cuda`         |
| BIO/BIOES tagging scheme        | Standard for joint segmentation+POS in Myanmar NLP literature    |
| Pydantic v2 schemas             | Strict input validation, automatic error responses               |

---

## Code Conventions

- **Language:** Python 3.9+
- **Formatter:** `black` (line length 88)
- **Linter:** `ruff`
- **Type hints:** Required on all function signatures
- **Docstrings:** Google-style on all public functions and classes
- **Async:** Use `async def` for all FastAPI route handlers

---

## Key Files & Responsibilities

### `app/model.py`
- Load tokenizer and model from HuggingFace Hub at startup
- Implement `predict(text: str) -> list[dict]` for single inference
- Implement `batch_predict(texts: list[str]) -> list[list[dict]]` for batch
- Handle BIO/BIOES tag decoding to `(word, pos_tag)` pairs
- Cache model in module-level singleton — do NOT reload per request

```python
# Example structure
class MyanmarNLPModel:
    def __init__(self, model_name: str, device: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def predict(self, text: str) -> list[dict]:
        # Tokenize → inference → decode BIO tags → return word/pos pairs
        ...
```

---

### `app/main.py`
- Initialize FastAPI app with metadata (title, version, description)
- Load model into `app.state.model` using `@app.on_event("startup")`
- Register routers from `app/routes/`
- Add CORS middleware for cross-origin frontend access
- Add request logging middleware

---

### `app/schemas.py`
- `SegmentRequest`: `text: str` (min_length=1, max_length=2000)
- `SegmentResponse`: `input: str`, `words: list[str]`, `processing_time_ms: float`
- `POSRequest`: `text: str`
- `POSToken`: `word: str`, `pos: str`
- `POSResponse`: `input: str`, `tokens: list[POSToken]`, `processing_time_ms: float`
- `BatchRequest`: `texts: list[str]` (max 32 items)
- `HealthResponse`: `status: str`, `model: str`, `device: str`

---

### `app/utils.py`
- `decode_bio_tags(tokens, labels) -> list[tuple[str, str]]`
  - Convert BIO label sequence to `(word, pos_tag)` pairs
  - Handle subword token merging from XLM-RoBERTa tokenization
- `merge_subword_tokens(tokens, labels) -> list[tuple[str, str]]`
  - Merge `##`-prefixed or `▁`-prefixed subword pieces back to whole words

---

## BIO Tag Decoding Logic

The model outputs BIO-scheme labels combining segmentation and POS:

```
B-NOUN  → Begin of a NOUN word
I-NOUN  → Inside continuation of a NOUN word
B-VERB  → Begin of a VERB word
B-PART  → Begin of a PART (particle) word
O       → Non-word / punctuation
```

Decoding algorithm:
1. Iterate over `(token, label)` pairs
2. On `B-*` → start new word, record POS tag
3. On `I-*` → append token to current word (subword continuation)
4. On `O`   → finalize current word if open; skip punctuation or include separately
5. Return `list[{"word": str, "pos": str}]`

---

## Testing Guidelines

- Use `pytest` with `httpx.AsyncClient` for async endpoint testing
- Mock model inference in unit tests to avoid loading full model
- Integration tests should load the actual model (mark with `@pytest.mark.integration`)
- Test cases must include:
  - Empty string → 422 validation error
  - Single word Myanmar text
  - Full sentence with punctuation (။)
  - Mixed Myanmar + English text
  - Very long text (>500 chars) → graceful truncation

---

## Adding New Endpoints

When adding a new endpoint, follow this checklist:

1. [ ] Define request/response schemas in `app/schemas.py`
2. [ ] Add route handler in `app/routes/` (new file if new domain)
3. [ ] Register router in `app/main.py`
4. [ ] Write unit test in `tests/test_api.py`
5. [ ] Update `README.md` API Endpoints section
6. [ ] Update OpenAPI description string in the route decorator

---

## Performance Considerations

- **Batch size:** Default max batch = 32 sentences; configurable via `MAX_BATCH_SIZE` env var
- **Sequence length:** Truncate at `MAX_LENGTH=512` tokens; log a warning if truncation occurs
- **Concurrency:** FastAPI async routes handle concurrent requests; model inference is synchronous — wrap in `asyncio.run_in_executor` if CPU-bound blocking becomes an issue
- **Caching:** Consider adding Redis caching for repeated identical inputs in production

---

## Common Issues & Fixes

| Issue                              | Fix                                                               |
|------------------------------------|-------------------------------------------------------------------|
| Myanmar text garbled in JSON       | Ensure `ensure_ascii=False` in JSONResponse                       |
| Subword tokens not merging correctly | Check `▁` (SentencePiece) prefix handling in `merge_subword_tokens` |
| Model not found on startup         | Verify `MODEL_NAME` env var; check HuggingFace token if private   |
| OOM on long texts                  | Reduce `MAX_LENGTH`; use `torch.no_grad()` during inference       |
| CRF decode slow                    | Enable `DEVICE=cuda` or reduce batch size                         |

---

## Out of Scope

Do **not** implement the following in this repository:
- Model training or fine-tuning code
- Data preprocessing pipelines
- Frontend UI components
- Database integration (stateless API only)
- Authentication (add separately as middleware if needed)
