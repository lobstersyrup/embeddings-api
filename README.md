# Embeddings API

OpenAI-compatible Embeddings API server using [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Single-file, local inference -- no external API calls.

## Quick Start

```bash
git clone https://github.com/lobstersyrup/embeddings-api.git
cd embeddings-api
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python embeddings_server.py
```

Server starts on `http://0.0.0.0:8882`. The model (~90MB) is downloaded automatically on first run.

## API

### POST /v1/embeddings

OpenAI-compatible embeddings endpoint. Accepts a single string or list of strings.

```bash
# Single string
curl http://localhost:8882/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world"}'

# Batch
curl http://localhost:8882/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["hello", "world"]}'
```

**Response:** 384-dim normalized embeddings in OpenAI format:
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.034, -0.019, ...]
    }
  ],
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "usage": {"prompt_tokens": 2, "total_tokens": 2}
}
```

### GET /v1/models

Returns the currently loaded model.

### GET /health

Health check with uptime, memory usage, embedding dims, and server config.

## Configuration

All via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDINGS_HOST` | `0.0.0.0` | Bind address |
| `EMBEDDINGS_PORT` | `8882` | Bind port |
| `EMBEDDINGS_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HF model name |
| `EMBEDDINGS_MAX_CONCURRENT` | `4` | Max concurrent embedding requests |
| `EMBEDDINGS_DIMS` | `0` | Override embedding dims (0 = model default) |

## Systemd

```bash
cp embeddings-api.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now embeddings-api.service
```

Pass custom environment variables by adding lines to the `[Service]` block:

```ini
Environment=EMBEDDINGS_PORT=8883
Environment=EMBEDDINGS_MAX_CONCURRENT=2
```
