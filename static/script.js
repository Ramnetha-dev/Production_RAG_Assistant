const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chatForm");
const inputEl = document.getElementById("messageInput");
const statusEl = document.getElementById("status");
const reindexBtn = document.getElementById("reindexBtn");
const resetBtn = document.getElementById("resetBtn");
const healthStatusEl = document.getElementById("healthStatus");
const healthModeEl = document.getElementById("healthMode");
const healthChunksEl = document.getElementById("healthChunks");
const promptChipEls = document.querySelectorAll(".prompt-chip");
let typingEl = null;

const sessionId = `session_${Math.random().toString(36).slice(2)}`;
const welcomeText = "Ask a question grounded in docs.json. Use Reindex if you changed documents.";

function pushMessage(role, text, meta = "", citations = []) {
  const block = document.createElement("article");
  block.className = `msg ${role}`;
  const roleTag = document.createElement("div");
  roleTag.className = "role";
  roleTag.textContent = role === "user" ? "You" : "Assistant";
  block.appendChild(roleTag);

  const body = document.createElement("div");
  body.textContent = text;
  block.appendChild(body);

  if (citations.length) {
    const container = document.createElement("div");
    container.className = "citations";
    citations.forEach((c) => {
      const chip = document.createElement("span");
      chip.className = "citation";
      chip.textContent = `[${c.id}] ${c.title} (score=${c.score})`;
      container.appendChild(chip);
    });
    block.appendChild(container);
  }

  if (meta) {
    const m = document.createElement("div");
    m.className = "meta";
    m.textContent = meta;
    block.appendChild(m);
  }
  messagesEl.appendChild(block);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setStatus(msg, kind = "normal") {
  statusEl.textContent = msg;
  statusEl.classList.toggle("error", kind === "error");
}

function showTyping() {
  if (typingEl) return;
  typingEl = document.createElement("article");
  typingEl.className = "msg assistant";
  typingEl.innerHTML = `
    <div class="role">Assistant</div>
    <div class="typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>
  `;
  messagesEl.appendChild(typingEl);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function hideTyping() {
  if (!typingEl) return;
  typingEl.remove();
  typingEl = null;
}

async function postJson(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = data.error || `Request failed (${res.status})`;
    throw new Error(msg);
  }
  return data;
}

async function fetchHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    healthStatusEl.textContent = data.status || "unknown";
    healthChunksEl.textContent = `${data.indexed_chunks || 0} chunks`;
    healthModeEl.textContent = data.has_openai_key ? "OpenAI" : "Local fallback";
  } catch {
    healthStatusEl.textContent = "offline";
    healthModeEl.textContent = "unknown";
    healthChunksEl.textContent = "0 chunks";
  }
}

function clearMessages() {
  messagesEl.innerHTML = "";
}

function resetConversationView() {
  clearMessages();
  pushMessage("assistant", welcomeText);
}

formEl.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = inputEl.value.trim();
  if (!message) return;
  inputEl.value = "";
  pushMessage("user", message);
  setStatus("Thinking...");
  showTyping();

  try {
    const data = await postJson("/chat", { message, session_id: sessionId });
    hideTyping();
    if (!data.ok) {
      pushMessage("assistant", `Error: ${data.error || "Unknown error"}`);
      setStatus("Failed.", "error");
      return;
    }
    pushMessage("assistant", data.answer, `Latency: ${data.latency_ms} ms`, data.citations || []);
    setStatus("Ready.");
  } catch (err) {
    hideTyping();
    pushMessage("assistant", `Error: ${err.message}`);
    setStatus("Network error.", "error");
  }
});

reindexBtn.addEventListener("click", async () => {
  reindexBtn.disabled = true;
  reindexBtn.textContent = "Reindexing...";
  setStatus("Reindexing documents...");
  try {
    const data = await postJson("/reindex", {});
    const summary = `Reindexed ${data.stats.docs} docs into ${data.stats.chunks} chunks in ${data.elapsed_ms} ms.`;
    setStatus(
      summary
    );
    pushMessage("assistant", "Reindex completed.", summary);
    await fetchHealth();
  } catch (err) {
    setStatus(`Reindex error: ${err.message}`, "error");
    pushMessage("assistant", `Error: ${err.message}`);
  } finally {
    reindexBtn.disabled = false;
    reindexBtn.textContent = "Reindex Docs";
  }
});

resetBtn.addEventListener("click", async () => {
  resetBtn.disabled = true;
  resetBtn.textContent = "Resetting...";
  setStatus("Resetting memory...");
  try {
    const data = await postJson("/reset-memory", { session_id: sessionId });
    resetConversationView();
    setStatus("Conversation memory cleared.");
    pushMessage("assistant", `Memory reset for session: ${data.session_id}`);
  } catch (err) {
    setStatus(`Reset error: ${err.message}`, "error");
    pushMessage("assistant", `Error: ${err.message}`);
  } finally {
    resetBtn.disabled = false;
    resetBtn.textContent = "Reset Memory";
  }
});

promptChipEls.forEach((chip) => {
  chip.addEventListener("click", () => {
    inputEl.value = chip.dataset.prompt || "";
    inputEl.focus();
  });
});

resetConversationView();
fetchHealth();
