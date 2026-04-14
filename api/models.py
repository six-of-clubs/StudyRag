"""
API data models for StudyRAG.

Pydantic schemas for request/response payloads and internal state.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Source Folders
# ---------------------------------------------------------------------------

class FolderCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, examples=["Linear Algebra"])


class FolderInfo(BaseModel):
    id: str
    name: str
    document_count: int = 0


class DocumentInfo(BaseModel):
    filename: str
    chunk_count: int


# ---------------------------------------------------------------------------
# Chats
# ---------------------------------------------------------------------------

class ChatCreate(BaseModel):
    title: str = Field(default="New Chat", max_length=200)


class ChatInfo(BaseModel):
    id: str
    title: str
    folder_id: str | None = None
    message_count: int = 0


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    sources: list[SourceInfo] = []
    declined: bool = False


class SourceInfo(BaseModel):
    source_number: int
    source_file: str
    page_number: int | str
    similarity: float


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    chat_id: str
    folder_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo] = []
    declined: bool = False
