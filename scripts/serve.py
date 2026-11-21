#!/usr/bin/env python
"""SageMaker inference server (``docker run <image> serve``).

Implements the two routes SageMaker hosting requires on port 8080: ``GET /ping`` and
``POST /invocations``. The model is read from ``/opt/ml/model`` (SageMaker extracts
model.tar.gz there). Request/response schema: see ``graph_fraud.inference``.

Local test:
    MODEL_DIR=artifacts/graphsage python scripts/serve.py
    python - <<'PY'
    import json, requests
    from graph_fraud.data import EllipticTables
    from graph_fraud.payload import build_request
    req = build_request(EllipticTables.load('data/processed'), [70335036], num_hops=2)
    print(requests.post('http://localhost:8080/invocations', json=req).json())
    PY
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from graph_fraud.artifacts import LoadedModel  # noqa: E402
from graph_fraud.inference import BadRequest, score_payload  # noqa: E402

log = logging.getLogger("serve")
MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/ml/model")


def create_app(model_dir: str = MODEL_DIR) -> FastAPI:
    app = FastAPI(title="graph-fraud-scoring")
    lm = LoadedModel(model_dir)
    log.info("loaded %s model from %s (threshold %.3f)", lm.model_type, model_dir, lm.threshold)

    @app.get("/ping")
    def ping():
        return JSONResponse({"status": "ok", "model_type": lm.model_type})

    @app.post("/invocations")
    async def invocations(request: Request):
        try:
            payload = await request.json()
        except ValueError as exc:
            raise HTTPException(status_code=415, detail="body must be JSON") from exc
        try:
            return JSONResponse(score_payload(lm, payload))
        except BadRequest as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(create_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), workers=1)
