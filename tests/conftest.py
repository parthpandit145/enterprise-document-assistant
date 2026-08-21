import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The tests assert against the documented defaults. `load_dotenv` runs with
# override=False, so setting these here means a developer's local `.env`
# cannot change the outcome of the suite.
os.environ.update(
    {
        "LLM_PROVIDER": "anthropic",
        "LLM_MODEL": "claude-opus-5",
        "LLM_EFFORT": "medium",
        "LLM_MAX_TOKENS": "4096",
        "EMBEDDING_PROVIDER": "huggingface",
        "CHUNK_SIZE": "1000",
        "CHUNK_OVERLAP": "200",
        "TOP_K": "5",
        "SEARCH_TYPE": "similarity",
        "SCORE_THRESHOLD": "0.15",
        "MAX_CONTEXT_CHARS": "12000",
        "TOKENIZERS_PARALLELISM": "false",
    }
)
