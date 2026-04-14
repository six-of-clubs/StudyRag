/* ==========================================================================
   StudyRAG — Frontend Application
   ========================================================================== */

const API = "";  // same origin

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let chats = [];
let folders = [];
let activeChatId = null;
let activeFolderId = null;   // folder being viewed in main area
let currentView = "empty";   // "empty" | "chat" | "folder"

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

async function api(method, path, body = null) {
  const opts = { method, headers: {} };
  if (body instanceof FormData) {
    opts.body = body;
  } else if (body) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`${res.status}: ${err}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

async function loadChats() {
  chats = await api("GET", "/api/chats");
  renderChatList();
}

async function loadFolders() {
  folders = await api("GET", "/api/folders");
  renderFolderList();
  renderFolderSelect();
}

// ---------------------------------------------------------------------------
// Rendering: Sidebar
// ---------------------------------------------------------------------------

function renderChatList() {
  const ul = document.getElementById("chat-list");
  ul.innerHTML = "";

  if (chats.length === 0) {
    ul.innerHTML = `<li class="muted" style="cursor:default;padding:8px 10px">No chats yet</li>`;
    return;
  }

  // Show newest first
  [...chats].reverse().forEach(chat => {
    const li = document.createElement("li");
    if (currentView === "chat" && activeChatId === chat.id) li.classList.add("active");

    li.innerHTML = `
      <span class="list-item-icon">💬</span>
      <span class="list-item-name">${esc(chat.title)}</span>
      <span class="list-item-badge">${chat.message_count}</span>
      <button class="list-item-delete" title="Delete chat">&times;</button>
    `;

    li.querySelector(".list-item-name").addEventListener("click", () => openChat(chat.id));
    li.querySelector(".list-item-delete").addEventListener("click", async (e) => {
      e.stopPropagation();
      if (confirm(`Delete "${chat.title}"?`)) {
        await api("DELETE", `/api/chats/${chat.id}`);
        if (activeChatId === chat.id) showEmpty();
        await loadChats();
      }
    });

    ul.appendChild(li);
  });
}

function renderFolderList() {
  const ul = document.getElementById("folder-list");
  ul.innerHTML = "";

  if (folders.length === 0) {
    ul.innerHTML = `<li class="muted" style="cursor:default;padding:8px 10px">No folders yet</li>`;
    return;
  }

  folders.forEach(folder => {
    const li = document.createElement("li");
    if (currentView === "folder" && activeFolderId === folder.id) li.classList.add("active");

    li.innerHTML = `
      <span class="list-item-icon">📁</span>
      <span class="list-item-name">${esc(folder.name)}</span>
      <span class="list-item-badge">${folder.document_count}</span>
      <button class="list-item-delete" title="Delete folder">&times;</button>
    `;

    li.querySelector(".list-item-name").addEventListener("click", () => openFolder(folder.id));
    li.querySelector(".list-item-delete").addEventListener("click", async (e) => {
      e.stopPropagation();
      if (confirm(`Delete folder "${folder.name}" and all its documents?`)) {
        await api("DELETE", `/api/folders/${folder.id}`);
        if (activeFolderId === folder.id) showEmpty();
        await loadFolders();
      }
    });

    ul.appendChild(li);
  });
}

function renderFolderSelect() {
  const sel = document.getElementById("folder-select");
  const current = sel.value;
  sel.innerHTML = `<option value="">No source folder</option>`;
  folders.forEach(f => {
    const opt = document.createElement("option");
    opt.value = f.id;
    opt.textContent = f.name;
    sel.appendChild(opt);
  });
  sel.value = current;
}

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

function showEmpty() {
  currentView = "empty";
  activeChatId = null;
  activeFolderId = null;
  document.getElementById("empty-state").classList.remove("hidden");
  document.getElementById("chat-view").classList.add("hidden");
  document.getElementById("folder-view").classList.add("hidden");
  renderChatList();
  renderFolderList();
}

async function openChat(chatId) {
  currentView = "chat";
  activeChatId = chatId;
  activeFolderId = null;

  document.getElementById("empty-state").classList.add("hidden");
  document.getElementById("folder-view").classList.add("hidden");
  document.getElementById("chat-view").classList.remove("hidden");

  renderChatList();
  renderFolderList();

  // Load full chat
  const chat = await api("GET", `/api/chats/${chatId}`);
  document.getElementById("chat-title").textContent = chat.title;

  // Set folder selector
  const sel = document.getElementById("folder-select");
  sel.value = chat.folder_id || "";

  // Render messages
  renderMessages(chat.messages);

  // Focus input
  document.getElementById("query-input").focus();
}

function renderMessages(messages) {
  const container = document.getElementById("messages");
  container.innerHTML = "";

  messages.forEach(msg => {
    container.appendChild(createMessageEl(msg));
  });

  container.scrollTop = container.scrollHeight;
}

function createMessageEl(msg) {
  const div = document.createElement("div");
  div.className = `message ${msg.role}`;
  if (msg.declined) div.classList.add("declined");

  let sourcesHtml = "";
  if (msg.sources && msg.sources.length > 0) {
    const tags = msg.sources.map(s =>
      `<span class="source-tag">[${s.source_number}] ${esc(s.source_file)} p.${s.page_number} <span class="sim">${(s.similarity * 100).toFixed(0)}%</span></span>`
    ).join("");
    sourcesHtml = `<div class="message-sources">${tags}</div>`;
  }

  div.innerHTML = `
    <div class="message-role">${msg.role === "user" ? "You" : "StudyRAG"}</div>
    <div class="message-content">${esc(msg.content)}</div>
    ${sourcesHtml}
  `;

  return div;
}

async function openFolder(folderId) {
  currentView = "folder";
  activeFolderId = folderId;
  activeChatId = null;

  document.getElementById("empty-state").classList.add("hidden");
  document.getElementById("chat-view").classList.add("hidden");
  document.getElementById("folder-view").classList.remove("hidden");

  renderChatList();
  renderFolderList();

  const folder = folders.find(f => f.id === folderId);
  document.getElementById("folder-view-title").textContent = folder ? folder.name : "Folder";

  // Load documents
  const docs = await api("GET", `/api/folders/${folderId}/documents`);
  const container = document.getElementById("folder-documents");

  if (docs.length === 0) {
    container.innerHTML = `<p class="muted">No documents yet. Upload files to add them as permanent sources.</p>`;
  } else {
    container.innerHTML = docs.map(d => `
      <div class="doc-item">
        <span class="doc-item-name">📄 ${esc(d.filename)}</span>
        <span class="doc-item-chunks">${d.chunk_count} chunks</span>
      </div>
    `).join("");
  }
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

// New chat
document.getElementById("btn-new-chat").addEventListener("click", async () => {
  const chat = await api("POST", "/api/chats", { title: "New Chat" });
  await loadChats();
  openChat(chat.id);
});

// New folder (modal)
document.getElementById("btn-new-folder").addEventListener("click", () => {
  showModal("New Source Folder", "Folder name", async (name) => {
    if (!name.trim()) return;
    await api("POST", "/api/folders", { name: name.trim() });
    await loadFolders();
  });
});

// Folder select change
document.getElementById("folder-select").addEventListener("change", async (e) => {
  if (!activeChatId) return;
  const folderId = e.target.value || null;
  await api("PATCH", `/api/chats/${activeChatId}/folder?folder_id=${folderId || ""}`);
});

// Chat file upload (temporary)
document.getElementById("btn-chat-upload").addEventListener("click", () => {
  document.getElementById("chat-file-input").click();
});

document.getElementById("chat-file-input").addEventListener("change", async (e) => {
  if (!activeChatId || !e.target.files.length) return;
  const file = e.target.files[0];
  e.target.value = "";

  const fd = new FormData();
  fd.append("file", file);

  // Show temporary notice
  const container = document.getElementById("messages");
  const notice = document.createElement("div");
  notice.className = "temp-notice";
  notice.textContent = `Uploading ${file.name}...`;
  container.appendChild(notice);
  container.scrollTop = container.scrollHeight;

  try {
    const result = await api("POST", `/api/chats/${activeChatId}/upload`, fd);
    notice.textContent = `📎 ${file.name} attached (${result.chunk_count} chunks) — available in this chat only`;
  } catch (err) {
    notice.textContent = `❌ Failed to upload ${file.name}: ${err.message}`;
  }
});

// Folder file upload (permanent)
document.getElementById("btn-folder-upload").addEventListener("click", () => {
  document.getElementById("folder-file-input").click();
});

document.getElementById("folder-file-input").addEventListener("change", async (e) => {
  if (!activeFolderId || !e.target.files.length) return;
  const files = [...e.target.files];
  e.target.value = "";

  for (const file of files) {
    const fd = new FormData();
    fd.append("file", file);
    try {
      await api("POST", `/api/folders/${activeFolderId}/upload`, fd);
    } catch (err) {
      alert(`Failed to upload ${file.name}: ${err.message}`);
    }
  }

  await loadFolders();
  openFolder(activeFolderId);
});

// Delete folder
document.getElementById("btn-delete-folder").addEventListener("click", async () => {
  if (!activeFolderId) return;
  const folder = folders.find(f => f.id === activeFolderId);
  if (confirm(`Delete folder "${folder?.name}" and all its documents?`)) {
    await api("DELETE", `/api/folders/${activeFolderId}`);
    showEmpty();
    await loadFolders();
  }
});

// ---------------------------------------------------------------------------
// Query / Send
// ---------------------------------------------------------------------------

const input = document.getElementById("query-input");
const btnSend = document.getElementById("btn-send");

// Auto-resize textarea
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 140) + "px";
});

// Send on Enter (Shift+Enter for newline)
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendQuery();
  }
});

btnSend.addEventListener("click", sendQuery);

async function sendQuery() {
  const question = input.value.trim();
  if (!question || !activeChatId) return;

  const folderId = document.getElementById("folder-select").value || null;

  input.value = "";
  input.style.height = "auto";
  btnSend.disabled = true;

  const container = document.getElementById("messages");

  // Render user message immediately
  container.appendChild(createMessageEl({ role: "user", content: question, sources: [] }));

  // Render loading indicator
  const loading = document.createElement("div");
  loading.className = "message assistant loading";
  loading.innerHTML = `
    <div class="message-role">StudyRAG</div>
    <div class="message-content">Thinking</div>
  `;
  container.appendChild(loading);
  container.scrollTop = container.scrollHeight;

  try {
    const result = await api("POST", "/api/query", {
      question,
      chat_id: activeChatId,
      folder_id: folderId,
    });

    loading.remove();
    container.appendChild(createMessageEl({
      role: "assistant",
      content: result.answer,
      sources: result.sources,
      declined: result.declined,
    }));

    // Update chat list message count
    await loadChats();

    // Auto-rename chat if it's the first message
    const chat = chats.find(c => c.id === activeChatId);
    if (chat && chat.message_count <= 2 && chat.title === "New Chat") {
      const title = question.length > 40 ? question.slice(0, 40) + "…" : question;
      await api("PATCH", `/api/chats/${activeChatId}/rename?title=${encodeURIComponent(title)}`);
      document.getElementById("chat-title").textContent = title;
      await loadChats();
    }

  } catch (err) {
    loading.remove();
    container.appendChild(createMessageEl({
      role: "assistant",
      content: `Error: ${err.message}`,
      sources: [],
      declined: true,
    }));
  }

  container.scrollTop = container.scrollHeight;
  btnSend.disabled = false;
  input.focus();
}

// ---------------------------------------------------------------------------
// Modal
// ---------------------------------------------------------------------------

function showModal(title, placeholder, onConfirm) {
  const overlay = document.createElement("div");
  overlay.className = "modal-overlay";

  overlay.innerHTML = `
    <div class="modal">
      <h3>${esc(title)}</h3>
      <input type="text" placeholder="${esc(placeholder)}" autofocus />
      <div class="modal-actions">
        <button class="btn-modal btn-modal-cancel">Cancel</button>
        <button class="btn-modal btn-modal-confirm">Create</button>
      </div>
    </div>
  `;

  const inputEl = overlay.querySelector("input");
  const cancelBtn = overlay.querySelector(".btn-modal-cancel");
  const confirmBtn = overlay.querySelector(".btn-modal-confirm");

  const close = () => overlay.remove();

  cancelBtn.addEventListener("click", close);
  overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });

  const submit = () => { onConfirm(inputEl.value); close(); };
  confirmBtn.addEventListener("click", submit);
  inputEl.addEventListener("keydown", (e) => { if (e.key === "Enter") submit(); });

  document.body.appendChild(overlay);
  inputEl.focus();
}

// ---------------------------------------------------------------------------
// Utils
// ---------------------------------------------------------------------------

function esc(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

(async () => {
  await Promise.all([loadChats(), loadFolders()]);
})();
