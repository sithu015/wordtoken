# Myanmar Word Segmentation & POS Tagging API

A FastAPI-based backend server for Myanmar (Burmese) **Word Segmentation** and **Part-of-Speech (POS) Tagging** using the fine-tuned [`sithu015/MyanBERTa-BiLSTM-CRF-Joint`](https://huggingface.co/sithu015/MyanBERTa-BiLSTM-CRF-Joint) model.

This repository now includes the application scaffold, actual Hugging Face artifact loading path, validation layer, test suite, and deployment/configuration files needed to run the API. On first startup the service downloads the fine-tuned checkpoint (`best_model.pt`, about 430 MB) from Hugging Face. If you explicitly enable fallback mode, the API can still boot with a lightweight heuristic segmenter when the model is unavailable.

---

## 🧠 Model Overview

| Property        | Detail                                      |
|----------------|---------------------------------------------|
| Base Model     | `UCSYNLP/MyanBERTa`                         |
| Architecture   | MyanBERTa + BiLSTM + CRF (Joint)           |
| Tasks          | Word Segmentation + POS Tagging (Joint)     |
| Language       | Myanmar (Burmese)                           |
| Hub            | [sithu015/MyanBERTa-BiLSTM-CRF-Joint](https://huggingface.co/sithu015/MyanBERTa-BiLSTM-CRF-Joint) |

---

## 📁 Project Structure

```
myanmar-nlp-api/
├── app/
│   ├── __init__.py
│   ├── auth.py              # API key authentication dependency
│   ├── config.py            # Environment-backed settings
│   ├── main.py              # FastAPI app entry point
│   ├── model.py             # Hugging Face-backed inference service
│   ├── routes/
│   │   ├── health.py        # Health check routes
│   │   └── nlp.py           # Segmentation/POS routes
│   ├── schemas.py           # Pydantic request/response schemas
│   └── utils.py             # Pre/post processing utilities
├── tests/
│   ├── conftest.py          # Shared async client fixture
│   └── test_api.py          # Unit tests
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── .gitignore
├── .dockerignore
├── deploy/
│   ├── Caddyfile            # Caddy reverse proxy for production HTTPS
│   ├── remote_deploy.sh     # Remote deployment script used by GitHub Actions
│   └── wordtoken.service    # systemd unit for Linux servers
├── .github/workflows/
│   └── deploy.yml           # Test + production deployment workflow
├── Dockerfile
├── .env.example
├── AGENT.md
└── README.md
```

---

## ⚙️ Requirements

- Python 3.9+
- PyTorch ≥ 2.0
- Transformers ≥ 4.35
- FastAPI
- Uvicorn

Install all dependencies:

```bash
pip install -r requirements.txt
```

For local development tools and tests:

```bash
pip install -r requirements-dev.txt
```

**`requirements.txt`:**
```
fastapi==0.135.1
uvicorn[standard]==0.42.0
transformers==5.3.0
huggingface-hub==1.7.2
torch==2.10.0
pytorch-crf==0.7.2
numpy==2.4.3
python-dotenv==1.2.2
pydantic==2.12.5
protobuf==7.34.1
```

---

## 🚀 Quick Start

Production docs are served directly by the API:

- `https://wordtoken.ygn.app/` — public product overview
- `https://wordtoken.ygn.app/wiki` — operations wiki
- `https://wordtoken.ygn.app/docs` — Swagger UI
- `https://wordtoken.ygn.app/redoc` — ReDoc

### 1. Clone the repository
```bash
git clone https://github.com/your-username/myanmar-nlp-api.git
cd myanmar-nlp-api
```

### 2. Set up environment
```bash
cp .env.example .env
# Edit .env as needed
```

### 3. Run the server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

When `API_KEYS` is configured, send a valid `X-API-Key` header on every
`/api/v1/*` request.

### 4. Run with Docker
```bash
docker build -t myanmar-nlp-api .
docker run -p 8000:8000 myanmar-nlp-api
```

### 5. Run with Docker Compose
```bash
cp .env.example .env
docker compose up -d --build
```

The first container startup downloads the Hugging Face checkpoint, so the initial boot can take several minutes.

### 6. Run with systemd on a Linux server
```bash
sudo useradd --system --create-home --home-dir /opt/wordtoken wordtoken
sudo rsync -az . /opt/wordtoken/
sudo python3 -m venv /opt/wordtoken/.venv
sudo /opt/wordtoken/.venv/bin/pip install -r /opt/wordtoken/requirements.txt
sudo cp deploy/wordtoken.service /etc/systemd/system/wordtoken.service
sudo systemctl daemon-reload
sudo systemctl enable --now wordtoken
```

This path avoids large Docker image exports and keeps the Hugging Face cache on disk at `/opt/wordtoken/.cache/huggingface`.

### 7. Put Caddy in front for HTTPS
```bash
sudo apt-get update
sudo apt-get install -y caddy
sudo cp deploy/Caddyfile /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

The bundled [Caddyfile](/Users/sithuaung/.codex/worktrees/2e6b/wordtoken/deploy/Caddyfile) terminates TLS for `wordtoken.ygn.app` and reverse proxies traffic to the API on `127.0.0.1:8000`.

### 8. GitHub Actions production deploys
Pushes to `main` trigger [.github/workflows/deploy.yml](/Users/sithuaung/.codex/worktrees/2e6b/wordtoken/.github/workflows/deploy.yml), which:

1. runs the test suite
2. syncs the repository to the server over SSH
3. executes [deploy/remote_deploy.sh](/Users/sithuaung/.codex/worktrees/2e6b/wordtoken/deploy/remote_deploy.sh)
4. waits for the production health endpoint to recover

Repository variables required by the workflow:

- `DEPLOY_HOST`
- `DEPLOY_USER`
- `DEPLOY_PORT`
- `DEPLOY_PATH`
- `SITE_URL`

Repository secret required by the workflow:

- `DEPLOY_SSH_KEY`

### 9. API key auth
The API supports header-based auth through `X-API-Key`.

- Leave `API_KEYS` empty to disable auth during local development.
- Set `API_KEYS` to a comma-separated list to enable auth.
- Only `/api/v1/*` routes require the key by default.
- `/`, `/wiki`, `/docs`, `/redoc`, and `/health` remain public.

### 10. MyanBERTa tokenizer compatibility
The `UCSYNLP/MyanBERTa` tokenizer ships a `tokenizer_config.json` layout that breaks
on recent `transformers` releases. The runtime now mirrors the training notebook:

- it snapshots the tokenizer assets locally
- removes the incompatible `post_processor` and `model` fields
- loads the patched tokenizer before running inference

This workaround is automatic. You do not need to patch the tokenizer manually.

---

## 📡 API Endpoints

### `POST /api/v1/segment`
Myanmar စာကြောင်းကို Word Segmentation ပြုလုပ်သည်။

**Request:**
```json
{
  "text": "ကျွန်တော်သည်ကျောင်းသွားသည်"
}
```

**Header when auth is enabled:**
```text
X-API-Key: YOUR_API_KEY
```

**Response:**
```json
{
  "input": "ကျွန်တော်သည်ကျောင်းသွားသည်",
  "words": ["ကျွန်တော်", "သည်", "ကျောင်း", "သွား", "သည်"],
  "processing_time_ms": 45.2
}
```

---

### `POST /api/v1/pos`
Word Segmentation နှင့် POS Tagging တပြိုင်နက် ပြုလုပ်သည်။

**Request:**
```json
{
  "text": "ကျွန်တော်သည်ကျောင်းသွားသည်"
}
```

**Header when auth is enabled:**
```text
X-API-Key: YOUR_API_KEY
```

**Response:**
```json
{
  "input": "ကျွန်တော်သည်ကျောင်းသွားသည်",
  "tokens": [
    {"word": "ကျွန်တော်", "pos": "PRN"},
    {"word": "သည်",       "pos": "PART"},
    {"word": "ကျောင်း",   "pos": "NOUN"},
    {"word": "သွား",      "pos": "VERB"},
    {"word": "သည်",       "pos": "PART"}
  ],
  "processing_time_ms": 48.7
}
```

---

### `POST /api/v1/batch`
Batch processing — စာကြောင်းများစွာကို တစ်ကြိမ်တည်း ပြုလုပ်သည်။

**Request:**
```json
{
  "texts": [
    "ကျွန်တော်သည်ကျောင်းသွားသည်",
    "မြန်မာဘာသာသည်လှပသောဘာသာတစ်ခုဖြစ်သည်"
  ]
}
```

**Header when auth is enabled:**
```text
X-API-Key: YOUR_API_KEY
```

---

### `GET /health`
Server health check endpoint။

**Response:**
```json
{
  "status": "ok",
  "model": "sithu015/MyanBERTa-BiLSTM-CRF-Joint",
  "device": "cpu"
}
```

---

## 🐳 Docker Deployment

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY README.md ./

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

For a persistent deployment with cached model artifacts, use [compose.yaml](/Users/sithuaung/.codex/worktrees/2e6b/wordtoken/compose.yaml).

---

## 🌍 Environment Variables

| Variable          | Default                                          | Description              |
|------------------|--------------------------------------------------|--------------------------|
| `MODEL_NAME`     | `sithu015/MyanBERTa-BiLSTM-CRF-Joint`           | HuggingFace model ID     |
| `DEVICE`         | `cpu`                                            | `cpu` or `cuda`          |
| `MAX_LENGTH`     | `300`                                            | Max token sequence length |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Comma-separated trusted origins |
| `ENABLE_FALLBACK_MODEL` | `false`                                   | Use heuristic fallback if model load fails |
| `MODEL_REVISION` | `main`                                          | Hugging Face revision/tag |
| `HF_TOKEN`       | empty                                           | Optional Hugging Face token |
| `HOST`           | `0.0.0.0`                                        | Server host              |
| `PORT`           | `8000`                                           | Server port              |
| `API_KEYS`       | empty                                            | Comma-separated valid API keys for `X-API-Key` auth |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Citation

Model developed by **Sithu Aung**. If you use this API in research, please cite the original model:

```
@misc{sithu015-myanberta-bilstm-crf-joint,
  author    = {Sithu Aung},
  title     = {MyanBERTa-BiLSTM-CRF-Joint for Myanmar NLP},
  year      = {2026},
  publisher = {HuggingFace},
  url       = {https://huggingface.co/sithu015/MyanBERTa-BiLSTM-CRF-Joint}
}
