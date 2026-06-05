tmux new -s serve
vllm serve ./Qwen3-0.6B-W4A16 --dtype=bfloat16 --max-model-len 4096

curl http://localhost:8000/v1/models


/v1/models, /v1/chat/completions, /v1/completions, /v1/embeddings

VLLM_URL = "http://localhost:8000"

from openai import OpenAI
client = OpenAI(base_url=f"{VLLM_URL}/v1", api_key="unused")

curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "./Qwen3-0.6B-W4A16",
    "prompt": "Machine learning is a branch of",
    "max_tokens": 60
  }'


git@github.com:gmtreacy/vllmcourse.git



  
