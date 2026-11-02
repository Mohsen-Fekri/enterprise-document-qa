"""FastAPI backend for the enterprise document Q&A assistant."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from enterprise_qa.auth import USERS, authenticate, decode_token, issue_token, visible_classifications
from enterprise_qa.config import (
    CHUNK_SIZE_OPTIONS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_RETRIEVAL,
    RETRIEVAL_METHODS,
)
from enterprise_qa.evaluation import compare_configurations, evaluate_run, maybe_ragas
from enterprise_qa.ingest import infer_domain
from enterprise_qa.pipeline import get_pipeline


@asynccontextmanager
async def lifespan(_app: FastAPI):
    for size in CHUNK_SIZE_OPTIONS:
        get_pipeline(size)
    yield


app = FastAPI(
    title="Enterprise Document Q&A API",
    description="RAG assistant by JustinFE for policies, audits, risk reports, and cybersecurity documents.",
    version="1.0.0",
    contact={"name": "Mohsen-Fekri", "url": "https://github.com/Mohsen-Fekri"},
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoginRequest(BaseModel):
    username: str
    password: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    chunk_size: int = DEFAULT_CHUNK_SIZE
    retrieval_method: str = DEFAULT_RETRIEVAL
    top_k: int = 5


def current_user(authorization: Annotated[str | None, Header()] = None) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Sign in to continue.")
    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = decode_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Session expired. Sign in again.") from None
    user = USERS.get(payload.get("sub", ""))
    if user is None:
        raise HTTPException(status_code=401, detail="Unknown user.")
    return {"username": user.username, "full_name": user.full_name, "role": user.role}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/auth/login")
def login(body: LoginRequest) -> dict:
    user = authenticate(body.username, body.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = issue_token(user)
    return {
        "access_token": token,
        "token_type": "bearer",
        "username": user.username,
        "full_name": user.full_name,
        "role": user.role,
        "classifications": visible_classifications(user.role),
    }


@app.get("/auth/me")
def me(user: dict = Depends(current_user)) -> dict:
    return {**user, "classifications": visible_classifications(user["role"])}


@app.get("/documents")
def list_documents(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    user: dict = Depends(current_user),
) -> dict:
    pipeline = get_pipeline(chunk_size)
    docs = [doc.to_dict() for doc in pipeline.documents_for(user["role"])]
    return {"documents": docs, "stats": pipeline.stats(user["role"])}


@app.post("/ask")
def ask(body: AskRequest, user: dict = Depends(current_user)) -> dict:
    if body.chunk_size not in CHUNK_SIZE_OPTIONS:
        raise HTTPException(status_code=400, detail=f"chunk_size must be one of {list(CHUNK_SIZE_OPTIONS)}")
    if body.retrieval_method not in RETRIEVAL_METHODS:
        raise HTTPException(status_code=400, detail=f"retrieval_method must be one of {list(RETRIEVAL_METHODS)}")
    pipeline = get_pipeline(body.chunk_size)
    result = pipeline.ask(
        body.question.strip(),
        user["role"],
        method=body.retrieval_method,
        top_k=body.top_k,
    )
    payload = result.to_dict()
    payload["role"] = user["role"]
    return payload


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(""),
    classification: str = Form("internal"),
    domain: str = Form(""),
    chunk_size: int = Form(DEFAULT_CHUNK_SIZE),
    user: dict = Depends(current_user),
) -> dict:
    if user["role"] not in {"admin", "compliance"}:
        raise HTTPException(status_code=403, detail="Only admin and compliance officers may ingest documents.")
    filename = file.filename or "upload.bin"
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in {"pdf", "docx"}:
        raise HTTPException(status_code=400, detail="Only PDF and Word (.docx) files are supported.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    pipeline = get_pipeline(chunk_size)
    path = pipeline.save_upload(filename, data)
    meta = pipeline.ingest_file(
        path,
        title=title.strip() or path.stem.replace("_", " ").title(),
        classification=classification,
        domain=domain.strip() or infer_domain(filename),
        uploaded_by=user["username"],
    )
    return {"document": meta.to_dict(), "stats": pipeline.stats(user["role"])}


@app.post("/evaluate")
def evaluate(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    retrieval_method: str = DEFAULT_RETRIEVAL,
    include_ragas: bool = False,
    user: dict = Depends(current_user),
) -> dict:
    if user["role"] not in {"admin", "auditor", "compliance"}:
        raise HTTPException(status_code=403, detail="Evaluation is limited to admin, auditor, and compliance roles.")
    run = evaluate_run(chunk_size=chunk_size, method=retrieval_method)
    payload: dict = {"local": run}
    if include_ragas:
        payload["ragas"] = maybe_ragas(run)
    return payload


@app.post("/evaluate/compare")
def compare(user: dict = Depends(current_user)) -> dict:
    if user["role"] not in {"admin", "auditor", "compliance"}:
        raise HTTPException(status_code=403, detail="Evaluation is limited to admin, auditor, and compliance roles.")
    return compare_configurations()


@app.get("/options")
def options() -> dict:
    return {
        "chunk_sizes": list(CHUNK_SIZE_OPTIONS),
        "retrieval_methods": list(RETRIEVAL_METHODS),
        "classifications": ["public", "internal", "confidential", "restricted"],
        "domains": ["company_policy", "audit_procedure", "risk_report", "cybersecurity"],
        "demo_users": [
            {"username": u.username, "role": u.role, "password": u.password, "full_name": u.full_name}
            for u in USERS.values()
        ],
    }
