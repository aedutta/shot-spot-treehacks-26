from fastapi import FastAPI
import os

app = FastAPI()

@app.get("/api/debug-env")
def test_env():
    keys = list(os.environ.keys())
    modal_vars = {k: v for k, v in os.environ.items() if "MODAL" in k}
    return {
        "MODAL_KEYS_FOUND": list(modal_vars.keys()),
        "MODAL_TOKEN_ID_VAL_LEN": len(os.environ.get("MODAL_TOKEN_ID", "")),
        "ALL_KEYS_COUNT": len(keys),
        "ALL_KEYS_SAMPLE": keys[:10]  # Just sample keys to avoid leaking too much
    }
