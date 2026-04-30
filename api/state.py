"""
State manager:

Manages folders, chats, and the mapping between them and ChromaDB
collections. Each source folder gets its own collection. Each chat
can optionally have a temporary collection for chat-uploaded docs.

State is persisted as JSON so it survives server restarts.
"""

from __future__ import annotations

import json
import hashlib
import logging
import time
import re
from pathlib import Path
from dataclasses import dataclass, field

import chromadb
from sentence_transformers import SentenceTransformer

from config import settings
from ingestion.loader import load_file
from ingestion.chunker import chunk_documents

logger = logging.getLogger(__name__)

STATE_FILE = Path(settings.chroma_persist_dir).parent / "state.json"


def _slug(name: str) -> str:
    """Turn a folder name into a safe ChromaDB collection name."""
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return f"folder_{s}"[:60]


def _chat_collection_name(chat_id: str) -> str:
    return f"chat_{chat_id}"[:60]


def _chunk_id(source_file: str, page: int, chunk_idx: int, prefix: str = "") -> str:
    key = f"{prefix}{source_file}:p{page}:c{chunk_idx}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Embedding model singleton
# ---------------------------------------------------------------------------

_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        logger.info("Loading embedding model '%s' ...", settings.embedding_model)
        _embed_model = SentenceTransformer(settings.embedding_model)
    return _embed_model


# ---------------------------------------------------------------------------
# Persistent state
# ---------------------------------------------------------------------------

@dataclass
class FolderState:
    id: str
    name: str
    collection_name: str
    pinned: bool = False
    documents: dict[str, int] = field(default_factory=dict)  # filename → chunk count


@dataclass
class MessageState:
    role: str
    content: str
    sources: list[dict] = field(default_factory=list)
    declined: bool = False


@dataclass
class ChatState:
    id: str
    title: str
    folder_id: str | None = None
    pinned: bool = False
    messages: list[MessageState] = field(default_factory=list)
    temp_docs: dict[str, int] = field(default_factory=dict)  # filename → chunk count


class StateManager:
    """Central state for folders and chats, backed by JSON + ChromaDB."""

    def __init__(self):
        self.folders: dict[str, FolderState] = {}
        self.chats: dict[str, ChatState] = {}
        self._chroma = chromadb.PersistentClient(
            path=settings.chroma_persist_dir
        )
        self._load()

    # --- Persistence ---

    def _load(self):
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text())
                for fid, f in data.get("folders", {}).items():
                    self.folders[fid] = FolderState(**f)
                for cid, c in data.get("chats", {}).items():
                    msgs = [MessageState(**m) for m in c.pop("messages", [])]
                    self.chats[cid] = ChatState(**c, messages=msgs)
                logger.info(
                    "Loaded state: %d folder(s), %d chat(s)",
                    len(self.folders), len(self.chats),
                )
            except Exception as e:
                logger.error("Failed to load state: %s", e)

    def _save(self):
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "folders": {},
            "chats": {},
        }
        for fid, f in self.folders.items():
            data["folders"][fid] = {
                "id": f.id, "name": f.name,
                "collection_name": f.collection_name,
                "pinned": f.pinned,
                "documents": f.documents,
            }
        for cid, c in self.chats.items():
            data["chats"][cid] = {
                "id": c.id, "title": c.title,
                "folder_id": c.folder_id,
                "pinned": c.pinned,
                "messages": [
                    {"role": m.role, "content": m.content,
                     "sources": m.sources, "declined": m.declined}
                    for m in c.messages
                ],
                "temp_docs": c.temp_docs,
            }
        STATE_FILE.write_text(json.dumps(data, indent=2))

    # --- Folders ---

    def create_folder(self, name: str) -> FolderState:
        fid = hashlib.sha256(f"{name}{time.time()}".encode()).hexdigest()[:12]
        col_name = _slug(name)

        # Ensure unique collection name
        existing_cols = {f.collection_name for f in self.folders.values()}
        base = col_name
        counter = 2
        while col_name in existing_cols:
            col_name = f"{base}_{counter}"
            counter += 1

        folder = FolderState(id=fid, name=name, collection_name=col_name)
        self.folders[fid] = folder
        self._chroma.get_or_create_collection(
            name=col_name, metadata={"hnsw:space": "cosine"},
        )
        self._save()
        logger.info("Created folder '%s' (id=%s, col=%s)", name, fid, col_name)
        return folder

    def delete_folder(self, folder_id: str) -> bool:
        folder = self.folders.pop(folder_id, None)
        if folder is None:
            return False
        try:
            self._chroma.delete_collection(folder.collection_name)
        except Exception:
            pass
        self._save()
        return True

    def list_folders(self) -> list[FolderState]:
        return list(self.folders.values())

    def get_folder(self, folder_id: str) -> FolderState | None:
        return self.folders.get(folder_id)

    # --- Document ingestion into a folder ---

    def ingest_to_folder(self, folder_id: str, file_path: Path) -> int:
        folder = self.folders.get(folder_id)
        if folder is None:
            raise ValueError(f"Folder not found: {folder_id}")

        docs = load_file(file_path)
        if not docs:
            return 0

        chunks = chunk_documents(docs)
        if not chunks:
            return 0

        model = _get_embed_model()
        collection = self._chroma.get_or_create_collection(
            name=folder.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        ids = [_chunk_id(c.metadata["source_file"], c.metadata["page_number"],
                         c.metadata["chunk_index"]) for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.upsert(
            ids=ids, documents=texts,
            embeddings=embeddings, metadatas=metadatas,
        )

        folder.documents[file_path.name] = len(chunks)
        self._save()
        logger.info("Ingested %d chunks from %s into folder '%s'",
                     len(chunks), file_path.name, folder.name)
        return len(chunks)

    # --- Temporary chat documents ---

    def ingest_to_chat(self, chat_id: str, file_path: Path) -> int:
        chat = self.chats.get(chat_id)
        if chat is None:
            raise ValueError(f"Chat not found: {chat_id}")

        docs = load_file(file_path)
        if not docs:
            return 0

        chunks = chunk_documents(docs)
        if not chunks:
            return 0

        model = _get_embed_model()
        col_name = _chat_collection_name(chat_id)
        collection = self._chroma.get_or_create_collection(
            name=col_name, metadata={"hnsw:space": "cosine"},
        )

        ids = [_chunk_id(c.metadata["source_file"], c.metadata["page_number"],
                         c.metadata["chunk_index"], prefix=f"chat_{chat_id}_")
               for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.upsert(
            ids=ids, documents=texts,
            embeddings=embeddings, metadatas=metadatas,
        )

        chat.temp_docs[file_path.name] = len(chunks)
        self._save()
        return len(chunks)

    # --- Chats ---

    def create_chat(self, title: str = "New Chat") -> ChatState:
        cid = hashlib.sha256(f"chat{time.time()}".encode()).hexdigest()[:12]
        chat = ChatState(id=cid, title=title)
        self.chats[cid] = chat
        self._save()
        return chat

    def delete_chat(self, chat_id: str) -> bool:
        chat = self.chats.pop(chat_id, None)
        if chat is None:
            return False
        try:
            self._chroma.delete_collection(_chat_collection_name(chat_id))
        except Exception:
            pass
        self._save()
        return True

    def list_chats(self) -> list[ChatState]:
        return list(self.chats.values())

    def get_chat(self, chat_id: str) -> ChatState | None:
        return self.chats.get(chat_id)

    def set_chat_folder(self, chat_id: str, folder_id: str | None):
        chat = self.chats.get(chat_id)
        if chat:
            chat.folder_id = folder_id
            self._save()

    def add_message(self, chat_id: str, role: str, content: str,
                    sources: list[dict] | None = None, declined: bool = False):
        chat = self.chats.get(chat_id)
        if chat:
            chat.messages.append(MessageState(
                role=role, content=content,
                sources=sources or [], declined=declined,
            ))
            self._save()

    def rename_chat(self, chat_id: str, title: str):
        chat = self.chats.get(chat_id)
        if chat:
            chat.title = title
            self._save()

    def toggle_pin_chat(self, chat_id: str) -> bool:
        chat = self.chats.get(chat_id)
        if chat:
            chat.pinned = not chat.pinned
            self._save()
            return chat.pinned
        return False

    def toggle_pin_folder(self, folder_id: str) -> bool:
        folder = self.folders.get(folder_id)
        if folder:
            folder.pinned = not folder.pinned
            self._save()
            return folder.pinned
        return False

    def rename_folder(self, folder_id: str, name: str):
        folder = self.folders.get(folder_id)
        if folder:
            folder.name = name
            self._save()

    # --- Collection name resolution ---

    def get_collection_names(self, folder_id: str | None,
                             chat_id: str | None) -> list[str]:
        """
        Resolve a folder ID and/or chat ID into a list of ChromaDB
        collection names that the retriever should search.

        This is the ONLY place that maps user-facing IDs to storage.
        The retriever receives these names and searches nothing else,
        ensuring strict isolation between academic subjects.
        """
        names: list[str] = []

        if folder_id and folder_id in self.folders:
            names.append(self.folders[folder_id].collection_name)

        if chat_id:
            col_name = _chat_collection_name(chat_id)
            # Only include if the chat actually has temp docs
            chat = self.chats.get(chat_id)
            if chat and chat.temp_docs:
                names.append(col_name)

        return names


# Module-level singleton
state = StateManager()
