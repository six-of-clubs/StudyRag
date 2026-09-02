"""
API routes

Endpoints:
    Folders:  CRUD + document upload + rename + pin + document delete + file serve
    Chats:    CRUD + query + temp document upload + rename + pin
    Models:   list available modes
    Status:   health check

FILENAME SAFETY
    Four endpoints build a filesystem path out of a name the client supplied:
    two from the multipart upload body, two from the URL. Every one of them
    goes through _safe_filename() first.
"""

from __future__ import annotations

import shutil
import tempfile
import logging
from pathlib import Path, PurePath

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from api.models import (
    FolderCreate, FolderInfo, DocumentInfo,
    ChatCreate, ChatInfo, ChatMessage, SourceInfo,
    QueryRequest, QueryResponse,
)
from api.state import state
from config import MODEL_PRESETS, DEFAULT_MODE, settings
from orchestrator.pipeline import ask as pipeline_ask

logger = logging.getLogger(__name__)

router = APIRouter()

FILES_DIR = Path(settings.chroma_persist_dir).parent / "files"

MAX_FILENAME_LENGTH = 200


def _safe_filename(raw: str | None) -> str:
    """
    Reducing a client-supplied filename to a single, safe path component:

    `UploadFile.filename` is fully attacker-controlled. Joining it to a 
    directory without checking is a possible path traversal: 
    a filename of "../../../.ssh/authorized_keys" escapes the
    upload folder and writes wherever the server process can reach.

    PurePath(...).name discards everything up to the last separator, so any
    directory component ["../", "/etc/", "C:\\Windows\\", ..] are dropped and only
    the final segment survives. The remaining checks reject the leftovers that
    are still a valid `.name`: "..", "." and empty strings.

    Leading dots are rejected too. They are not a traversal risk, but a file
    called ".env" or ".gitignore" sitting in the upload directory is also something
    we would like to prevent.
    """
    if not raw:
        raise HTTPException(400, "Filename is required")

    name = PurePath(raw).name.strip()

    if not name or name in (".", ".."):
        raise HTTPException(400, "Invalid filename")
    if name.startswith("."):
        raise HTTPException(400, "Filenames may not start with a dot")
    if len(name) > MAX_FILENAME_LENGTH:
        raise HTTPException(400, "Filename is too long")

    if name != raw:
        logger.warning("Sanitised filename %r → %r", raw, name)

    return name


def _folder_files_dir(folder_id: str) -> Path:
    d = FILES_DIR / folder_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@router.get("/api/status")
def status():
    folders = state.list_folders()
    chats = state.list_chats()
    return {"status": "ok", "folders": len(folders), "chats": len(chats)}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@router.get("/api/models")
def list_models():
    return {
        "modes": [
            {"id": mode, "model": model_name, "default": mode == DEFAULT_MODE}
            for mode, model_name in MODEL_PRESETS.items()
        ]
    }


# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------

@router.get("/api/folders", response_model=list[FolderInfo])
def list_folders():
    return [
        FolderInfo(id=f.id, name=f.name, document_count=len(f.documents), pinned=f.pinned)
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
    folder_dir = FILES_DIR / folder_id
    if folder_dir.exists():
        shutil.rmtree(folder_dir, ignore_errors=True)
    return {"deleted": True}


@router.patch("/api/folders/{folder_id}/rename")
def rename_folder(folder_id: str, name: str):
    folder = state.get_folder(folder_id)
    if not folder:
        raise HTTPException(404, "Folder not found")
    state.rename_folder(folder_id, name)
    return {"name": name}


@router.patch("/api/folders/{folder_id}/pin")
def toggle_pin_folder(folder_id: str):
    folder = state.get_folder(folder_id)
    if not folder:
        raise HTTPException(404, "Folder not found")
    pinned = state.toggle_pin_folder(folder_id)
    return {"pinned": pinned}


@router.delete("/api/folders/{folder_id}/documents/{filename}")
def delete_folder_document(folder_id: str, filename: str):
    folder = state.get_folder(folder_id)
    if not folder:
        raise HTTPException(404, "Folder not found")

    safe_name = _safe_filename(filename)
    if safe_name not in folder.documents:
        raise HTTPException(404, "Document not found")

    del folder.documents[safe_name]
    state._save()

    stored = _folder_files_dir(folder_id) / safe_name
    if stored.exists():
        stored.unlink()
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


@router.get("/api/folders/{folder_id}/files/{filename}")
def serve_file(folder_id: str, filename: str):
    """Serve an uploaded file for viewing in the browser."""
    folder = state.get_folder(folder_id)
    if not folder:
        raise HTTPException(404, "Folder not found")

    safe_name = _safe_filename(filename)

    # Only serve files this folder actually knows about. Without this check the
    # endpoint would happily read any file that happens to sit in the folder
    # directory, whether it was ingested or not.
    if safe_name not in folder.documents:
        raise HTTPException(404, "Document not found")

    file_path = _folder_files_dir(folder_id) / safe_name
    if not file_path.exists():
        raise HTTPException(404, "File not found on disk")

    media_types = {
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }
    media_type = media_types.get(file_path.suffix.lower(), "application/octet-stream")

    return FileResponse(path=file_path, filename=safe_name, media_type=media_type)


@router.post("/api/folders/{folder_id}/upload", response_model=DocumentInfo)
async def upload_to_folder(folder_id: str, file: UploadFile = File(...)):
    folder = state.get_folder(folder_id)
    if not folder:
        raise HTTPException(404, "Folder not found")

    safe_name = _safe_filename(file.filename)

    persistent_dir = _folder_files_dir(folder_id)
    persistent_path = persistent_dir / safe_name

    with open(persistent_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        count = state.ingest_to_folder(folder_id, persistent_path)
        if count == 0:
            persistent_path.unlink(missing_ok=True)
            raise HTTPException(400, "No content could be extracted from this file")
        return DocumentInfo(filename=safe_name, chunk_count=count)
    except HTTPException:
        raise
    except Exception as e:
        persistent_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Ingestion failed: {e}")


# ---------------------------------------------------------------------------
# Chats
# ---------------------------------------------------------------------------

@router.get("/api/chats", response_model=list[ChatInfo])
def list_chats():
    return [
        ChatInfo(id=c.id, title=c.title, folder_id=c.folder_id,
                 pinned=c.pinned, message_count=len(c.messages))
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
        "id": c.id, "title": c.title, "folder_id": c.folder_id,
        "messages": [
            {"role": m.role, "content": m.content,
             "sources": m.sources, "declined": m.declined}
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


@router.patch("/api/chats/{chat_id}/pin")
def toggle_pin_chat(chat_id: str):
    chat = state.get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")
    pinned = state.toggle_pin_chat(chat_id)
    return {"pinned": pinned}


@router.post("/api/chats/{chat_id}/upload", response_model=DocumentInfo)
async def upload_to_chat(chat_id: str, file: UploadFile = File(...)):
    chat = state.get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")

    safe_name = _safe_filename(file.filename)

    tmp = Path(tempfile.mkdtemp())
    try:
        dest = tmp / safe_name
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        count = state.ingest_to_chat(chat_id, dest)
        if count == 0:
            raise HTTPException(400, "No content could be extracted from this file")
        return DocumentInfo(filename=safe_name, chunk_count=count)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Topic extraction
# ---------------------------------------------------------------------------

@router.post("/api/extract-topic")
def extract_topic(question: str):
    from generation.llm import generate
    try:
        title = generate(
            system_prompt=(
                "Extract the academic topic from the user's question. "
                "Reply with ONLY a short title (2-5 words). No quotes, no punctuation, no explanation. "
                "Examples: 'Matrix Eigenvalues', 'Gradient Descent', 'Fourier Transform'."
            ),
            user_prompt=question,
        )
        title = title.strip().strip('"').strip("'")[:60]
        if not title:
            title = question[:40]
        return {"title": title}
    except Exception:
        words = question.split()[:5]
        return {"title": " ".join(words)[:40]}


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

@router.post("/api/query", response_model=QueryResponse)
def query(body: QueryRequest):
    chat = state.get_chat(body.chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")

    folder_id = body.folder_id or chat.folder_id
    state.add_message(body.chat_id, "user", body.question)

    try:
        result = pipeline_ask(
            body.question, folder_id=folder_id,
            chat_id=body.chat_id, mode=body.mode,
        )
    except ConnectionError as e:
        raise HTTPException(503, str(e))

    sources_dicts = [
        {"source_number": s.source_number, "source_file": s.source_file,
         "page_number": s.page_number, "similarity": s.similarity}
        for s in result.sources
    ]
    state.add_message(body.chat_id, "assistant", result.answer,
                      sources=sources_dicts, declined=result.declined)

    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceInfo(source_number=s.source_number, source_file=s.source_file,
                       page_number=s.page_number, similarity=s.similarity)
            for s in result.sources
        ],
        declined=result.declined,
    )
