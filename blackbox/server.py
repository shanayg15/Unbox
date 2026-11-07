"""Lane A, task A1: black-box server"""
from fastapi import FastAPI

app = FastAPI(title="Unbox black box")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "blackbox"}
