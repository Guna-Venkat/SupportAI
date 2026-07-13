"""
app.py
======
Production-ready REST API for SupportAI ticketing system.
Exposes endpoints for health, prediction, semantic search, local explanations, and versioning.
"""

import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from src.evaluation.explainability import TicketExplainer
from src.models.transformer.decision_engine import DecisionEngine
from src.models.transformer.retrieval import SemanticRetriever
from src.utils.config import load_config
from src.utils.constants import OUTPUT_DIR
from src.utils.logging_utils import get_logger, setup_logging

# Configure structured JSON logging for the production API
setup_logging(level=logging.INFO, use_json=True)
logger = get_logger(__name__)


def get_git_commit() -> str:
    """Safely retrieves the current Git commit SHA."""
    git_bin = shutil.which("git")
    if not git_bin:
        return "unknown"
    try:
        return (
            subprocess.check_output(  # noqa: S603
                [git_bin, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


# --- Prometheus Metrics Definitions ---

REQUESTS_TOTAL = Counter(
    "supportai_requests_total",
    "Total HTTP requests received by SupportAI API",
    ["endpoint", "method", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "supportai_request_latency_seconds",
    "HTTP request processing latency in seconds",
    ["endpoint", "method"],
)

PREDICTION_CONFIDENCE = Histogram(
    "supportai_prediction_confidence",
    "Confidence score distribution of intent predictions",
    ["intent"],
)

ROUTING_DECISIONS = Counter(
    "supportai_routing_decision_total",
    "Total routing decisions categorized by destination route",
    ["route"],
)

MODEL_VERSION = Gauge(
    "supportai_model_version",
    "Model and system metadata information",
    ["version", "project", "git_commit", "experiment_id"],
)


# Global instances loaded at startup and cached in app state
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager to initialize and clean up models at app startup and shutdown."""
    logger.info("Initializing SupportAI REST API models on startup...")

    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        logger.error("Failed to load configuration: %s", e)
        config = {}

    git_sha = get_git_commit()
    exp_id = str(config.get("experiment_id", "default"))
    MODEL_VERSION.labels(
        version="0.1.0", project="SupportAI", git_commit=git_sha, experiment_id=exp_id
    ).set(1.0)

    if os.environ.get("TESTING") == "true":
        logger.info("Running in TESTING mode: overriding LLM configuration to tiny-random-gpt2")
        config["llm"] = {
            "enabled": True,
            "provider": "huggingface",
            "model_id": "hf-internal-testing/tiny-random-gpt2",
            "max_new_tokens": 10,
            "temperature": 0.0,
        }

    model_dir = OUTPUT_DIR / "models" / "best_model"

    retriever_index_dir = OUTPUT_DIR / "retrieval_index"

    # 1. Initialize Decision Engine
    try:
        if not model_dir.exists() or not retriever_index_dir.exists():
            logger.warning("Classifier or retrieval models missing! Starting in DEGRADED mode.")
            app.state.decision_engine = None
        else:
            app.state.decision_engine = DecisionEngine(
                model_dir=model_dir,
                retriever_index_dir=retriever_index_dir,
                config=config,
            )
    except Exception as e:
        logger.exception("Failed to initialize DecisionEngine: %s", e)
        app.state.decision_engine = None

    # 2. Initialize Semantic Retriever
    try:
        if not retriever_index_dir.exists():
            logger.warning("Retrieval index missing!")
            app.state.retriever = None
        else:
            app.state.retriever = SemanticRetriever()
            app.state.retriever.load_index(retriever_index_dir)
    except Exception as e:
        logger.exception("Failed to initialize SemanticRetriever: %s", e)
        app.state.retriever = None

    # 3. Initialize Ticket Explainer
    try:
        if not model_dir.exists():
            logger.warning("Classifier model missing for explainability!")
            app.state.explainer = None
        else:
            app.state.explainer = TicketExplainer(model_dir=model_dir)
    except Exception as e:
        logger.exception("Failed to initialize TicketExplainer: %s", e)
        app.state.explainer = None

    logger.info("Model loading complete.")
    yield
    logger.info("Shutting down REST API...")


app = FastAPI(
    title="SupportAI REST API",
    description=(
        "Lightweight routing, retrieval, and explainability API " "for customer support tickets."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start_time = time.time()

    # Pre-populate state attributes to avoid hasattr/getattr failures in middleware
    request.state.intent = None
    request.state.confidence = None
    request.state.route = None

    response = None
    error_msg = None
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        status_code = 500
        error_msg = str(e)
        raise e
    finally:
        latency = time.time() - start_time
        endpoint = request.url.path
        method = request.method

        if response:
            response.headers["X-Request-ID"] = request_id

        # Update metrics for endpoints we want to observe
        if endpoint not in ("/metrics", "/health", "/version"):
            # Update general HTTP metrics
            REQUESTS_TOTAL.labels(endpoint=endpoint, method=method, status_code=status_code).inc()
            REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(latency)

            # Retrieve git sha and experiment id safely
            git_commit_sha = get_git_commit()
            try:
                config = load_config()
            except Exception:
                config = {}
            experiment_id = str(config.get("experiment_id", "default"))

            # Log trace in JSONL
            trace = {
                "request_id": request_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code,
                "latency_seconds": latency,
                "version": "0.1.0",
                "git_commit": git_commit_sha,
                "experiment_id": experiment_id,
            }

            # If it is a /predict request, add prediction metadata from request state
            if endpoint == "/predict":
                intent = getattr(request.state, "intent", None)
                confidence = getattr(request.state, "confidence", None)
                route = getattr(request.state, "route", None)
                if intent is not None:
                    trace["intent"] = intent
                if confidence is not None:
                    trace["confidence"] = confidence
                if route is not None:
                    trace["route"] = route

            if error_msg:
                trace["error"] = error_msg

            # Write to logs/traces.jsonl
            traces_file = Path("logs/traces.jsonl")
            try:
                traces_file.parent.mkdir(parents=True, exist_ok=True)
                with open(traces_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(trace) + "\n")
            except Exception as e:
                logger.error("Failed to write to traces.jsonl: %s", e)

        if response is None:
            response = Response(content="Internal Server Error", status_code=500)

    return response


@app.get("/metrics")
def get_metrics():
    """Exposes Prometheus scraper metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", response_class=HTMLResponse)
def get_demo():
    """Serves the SupportAI web demo dashboard."""
    template_path = Path(__file__).parent / "templates" / "demo.html"
    try:
        with open(template_path, encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=status.HTTP_200_OK)
    except Exception as e:
        logger.exception("Failed to load demo template: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Demo page template is missing or unreadable.",
        ) from e


# --- Input/Output Validation Schemas ---


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=3, description="Support ticket text.")


class PredictResponse(BaseModel):
    intent: str
    confidence: float
    route: str
    retrieved_docs: list[dict[str, Any]]
    llm_used: bool
    reply: str


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Search query.")
    top_k: int = Field(3, ge=1, le=10, description="Number of results to retrieve.")


class RetrieveResponse(BaseModel):
    query: str
    results: list[dict[str, Any]]


class ExplainRequest(BaseModel):
    text: str = Field(..., min_length=3, description="Ticket text to explain.")
    num_features: int = Field(10, ge=1, le=20, description="Number of features to explain.")
    num_samples: int = Field(100, ge=10, le=500, description="LIME perturbation samples count.")


class ExplainResponse(BaseModel):
    predicted_class: str
    predicted_probability: float
    attributions: list[list[Any]]
    explanation_html: str


# --- Endpoints ---


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> dict[str, Any]:
    """Returns application readiness and status of loaded model components."""
    status_str = "healthy"

    classifier_ok = app.state.decision_engine is not None
    retriever_ok = app.state.retriever is not None
    explainer_ok = app.state.explainer is not None

    if not (classifier_ok and retriever_ok and explainer_ok):
        status_str = "degraded"

    return {
        "status": status_str,
        "classifier_loaded": classifier_ok,
        "retriever_loaded": retriever_ok,
        "explainer_loaded": explainer_ok,
    }


@app.get("/version")
def get_version() -> dict[str, str]:
    """Returns API details and current semantic version."""
    return {
        "project": "SupportAI",
        "version": "0.1.0",
        "description": "Production REST API for support ticket intelligence.",
    }


@app.post("/predict", response_model=PredictResponse)
def predict_route(request: Request, body: PredictRequest):
    """Processes ticket through the Decision Engine (auto-routing vs LLM fallback vs escalation)."""
    if app.state.decision_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DecisionEngine model is not loaded/available. Please check server logs.",
        )

    try:
        result = app.state.decision_engine.route_ticket(body.text)

        # Store predictions metadata in request.state for the observability middleware
        request.state.intent = result.get("intent")
        request.state.confidence = result.get("confidence")
        request.state.route = result.get("route")

        # Record metrics in Prometheus
        PREDICTION_CONFIDENCE.labels(intent=result.get("intent", "unknown")).observe(
            result.get("confidence", 0.0)
        )
        ROUTING_DECISIONS.labels(route=result.get("route", "unknown")).inc()

        return result
    except Exception as e:
        logger.exception("Inference error in /predict: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference processing failed: {e!s}",
        ) from e


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve_similar(request: RetrieveRequest):
    """Performs semantic search query over the support ticket database corpus."""
    if app.state.retriever is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SemanticRetriever database is not loaded/available. Please check server logs.",
        )

    try:
        results = app.state.retriever.retrieve(request.query, top_k=request.top_k)
        return {
            "query": request.query,
            "results": results,
        }
    except Exception as e:
        logger.exception("Retrieval error in /retrieve: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic retrieval failed: {e!s}",
        ) from e


@app.post("/explain", response_model=ExplainResponse)
def explain_ticket(request: ExplainRequest):
    """Generates LIME word attributions explaining the intent prediction."""
    if app.state.explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TicketExplainer is not loaded/available. Please check server logs.",
        )

    try:
        result = app.state.explainer.explain_ticket(
            text=request.text,
            num_features=request.num_features,
            num_samples=request.num_samples,
        )
        return {
            "predicted_class": result["predicted_class"],
            "predicted_probability": result["predicted_probability"],
            "attributions": result["attributions"],
            "explanation_html": result["explanation_html"],
        }
    except Exception as e:
        logger.exception("Explainability error in /explain: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explainability generation failed: {e!s}",
        ) from e


def run_api_main() -> None:
    """CLI entrypoint to start FastAPI uvicorn production server."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Start SupportAI REST API server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind.")  # noqa: S104
    parser.add_argument("--port", type=int, default=8000, help="Port to listen.")
    args = parser.parse_args()

    uvicorn.run("src.api.app:app", host=args.host, port=args.port, reload=False)
