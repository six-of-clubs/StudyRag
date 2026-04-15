"""
API routes for StudyRAG.

Endpoints:
    Folders:  CRUD + document upload
    Chats:    CRUD + query + temp document upload
    Status:   health check
"""

from __future__ import annotations

import shutil
import tempfile
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from api.models import (
    FolderCreate, FolderInfo, DocumentInfo,
    ChatCreate, ChatInfo, ChatMessage, SourceInfo,
    QueryRequest, QueryResponse,
)
from api.state import state
from orchestrator.pipeline import ask as pipeline_ask

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@router.get("/api/status")
def status():
    folders = state.list_folders()
    chats = state.list_chats()
    return {
        "status": "ok",
        "folders": len(folders),
        "chats": len(chats),
    }


# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------

@router.get("/api/folders", response_model=list[FolderInfo])
def list_folders():
    return [
        FolderInfo(
            id=f.id, name=f.name,
            document_count=len(f.documents),
        )
        for f in state.list_folders()
    ]


@router.post("/api/folders", response_model=FolderInfo)
def create_folder(body: FolderCreate):
    f = state.create_folder(body.name)
    return FolderInfo(id=f.id, name=f.name, document_count=0)


@router.delete("/api/folders/{folder_id}")
def delete_folder(folder_id: str):
    if not state.delete_folder(folder_id):
        raise HTTPException(404, "Folder not found")
    return {"deleted": True}


@router.get("/api/folders/{folder_id}/documents", response_model=list[DocumentInfo])
def list_folder_documents(folder_id: str):
    folder = state.get_folder(folder_id)
    if not folder:
        raise HTTPException(404, "Folder not found")
    return [
        DocumentInfo(filename=name, chunk_count=count)
        for name, count in folder.documents.items()
    ]


@router.post("/api/folders/{folder_id}/upload", response_model=DocumentInfo)
async def upload_to_folder(folder_id: str, file: UploadFile = File(...)):
    folder = state.get_folder(folder_id)
    if not folder:
        raise HTTPException(404, "Folder not found")

    tmp = Path(tempfile.mkdtemp())
    try:
        dest = tmp / file.filename
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)

        count = state.ingest_to_folder(folder_id, dest)
        if count == 0:
            raise HTTPException(400, "No content could be extracted from this file")

        return DocumentInfo(filename=file.filename, chunk_count=count)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Chats
# ---------------------------------------------------------------------------

@router.get("/api/chats", response_model=list[ChatInfo])
def list_chats():
    return [
        ChatInfo(
            id=c.id, title=c.title,
            folder_id=c.folder_id,
            message_count=len(c.messages),
        )
        for c in state.list_chats()
    ]


@router.post("/api/chats", response_model=ChatInfo)
def create_chat(body: ChatCreate):
    c = state.create_chat(body.title)
    return ChatInfo(id=c.id, title=c.title, message_count=0)


@router.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str):
    if not state.delete_chat(chat_id):
        raise HTTPException(404, "Chat not found")
    return {"deleted": True}


@router.get("/api/chats/{chat_id}", response_model=dict)
def get_chat(chat_id: str):
    c = state.get_chat(chat_id)
    if not c:
        raise HTTPException(404, "Chat not found")
    return {
        "id": c.id,
        "title": c.title,
        "folder_id": c.folder_id,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "sources": m.sources,
                "declined": m.declined,
            }
            for m in c.messages
        ],
        "temp_docs": c.temp_docs,
    }


@router.patch("/api/chats/{chat_id}/folder")
def set_chat_folder(chat_id: str, folder_id: str | None = None):
    chat = state.get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")
    if folder_id and not state.get_folder(folder_id):
        raise HTTPException(404, "Folder not found")
    state.set_chat_folder(chat_id, folder_id)
    return {"folder_id": folder_id}


@router.patch("/api/chats/{chat_id}/rename")
def rename_chat(chat_id: str, title: str):
    chat = state.get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")
    state.rename_chat(chat_id, title)
    return {"title": title}


@router.post("/api/chats/{chat_id}/upload", response_model=DocumentInfo)
async def upload_to_chat(chat_id: str, file: UploadFile = File(...)):
    chat = state.get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")

    tmp = Path(tempfile.mkdtemp())
    try:
        dest = tmp / file.filename
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)

        count = state.ingest_to_chat(chat_id, dest)
        if count == 0:
            raise HTTPException(400, "No content could be extracted from this file")

        return DocumentInfo(filename=file.filename, chunk_count=count)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

@router.post("/api/query", response_model=QueryResponse)
def query(body: QueryRequest):
    chat = state.get_chat(body.chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")

    folder_id = body.folder_id or chat.folder_id

    # Save user message
    state.add_message(body.chat_id, "user", body.question)

    # Run the full pipeline through the orchestrator
    try:
        result = pipeline_ask(body.question, folder_id=folder_id, chat_id=body.chat_id)
    except ConnectionError as e:
        raise HTTPException(503, str(e))

    # Save assistant message
    sources_dicts = [
        {
            "source_number": s.source_number,
            "source_file": s.source_file,
            "page_number": s.page_number,
            "similarity": s.similarity,
        }
        for s in result.sources
    ]
    state.add_message(
        body.chat_id, "assistant", result.answer,
        sources=sources_dicts, declined=result.declined,
    )

    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceInfo(
                source_number=s.source_number,
                source_file=s.source_file,
                page_number=s.page_number,
                similarity=s.similarity,
            )
            for s in result.sources
        ],
        declined=result.declined,
    )
