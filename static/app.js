"use strict";

const CAT_LABEL = { doc: "📄 文档配置", code: "🧩 源码实现", gen: "💻 代码需求" };
const state = { suggestions: [], loading: false, hasAsked: false };

const $ = (sel) => document.querySelector(sel);
const chatEl = $("#chat");
const inputEl = $("#input");
const suggestGroupsEl = $("#suggestGroups");
const suggestPanel = $("#suggestPanel");
const collapseBtn = $("#collapseBtn");
const emptyHintEl = $("#emptyHint");

// ---------- 转义 & 极简 markdown ----------
function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function inlineMd(s) {
  return escapeHtml(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
}

function renderMd(src) {
  const lines = String(src || "").split("\n");
  let html = "", inUl = false, inOl = false, inCode = false, codeBuf = [], codeLang = "";
  const flush = () => {
    if (inUl) { html += "</ul>"; inUl = false; }
    if (inOl) { html += "</ol>"; inOl = false; }
  };
  const flushCode = () => {
    html += `<pre><code class="lang-${codeLang}">${escapeHtml(codeBuf.join("\n"))}</code></pre>`;
    codeBuf = []; codeLang = ""; inCode = false;
  };
  for (const line of lines) {
    const fence = /^```(\w*)\s*$/.exec(line);
    if (fence) {
      if (inCode) flushCode();
      else { flush(); codeLang = fence[1]; inCode = true; }
      continue;
    }
    if (inCode) { codeBuf.push(line); continue; }
    if (!line.trim()) { flush(); continue; }
    let m;
    if ((m = /^(#{1,6})\s+(.*)$/.exec(line))) { flush(); html += `<h${m[1].length}>${inlineMd(m[2])}</h${m[1].length}>`; }
    else if ((m = /^\s*[-*+]\s+(.*)$/.exec(line))) { if (!inUl) { html += "<ul>"; inUl = true; } html += `<li>${inlineMd(m[1])}</li>`; }
    else if ((m = /^\s*(\d+)\.\s+(.*)$/.exec(line))) { if (!inOl) { html += "<ol>"; inOl = true; } html += `<li>${inlineMd(m[2])}</li>`; }
    else { flush(); html += `<p>${inlineMd(line)}</p>`; }
  }
  if (inCode) flushCode();
  flush();
  return html;
}

// ---------- 建议问题 ----------
async function loadSuggestions() {
  const btn = $("#refreshBtn");
  btn.disabled = true;
  try {
    const res = await fetch("/api/suggestions");
    const data = await res.json();
    state.suggestions = data.suggestions || [];
    renderSuggestions();
  } catch (e) {
    suggestGroupsEl.innerHTML = `<div class="empty-hint">加载建议失败：${escapeHtml(String(e))}</div>`;
  } finally {
    btn.disabled = false;
  }
}

function renderSuggestions() {
  const groups = { doc: [], code: [], gen: [] };
  state.suggestions.forEach((s, i) => {
    s._i = i;
    (groups[s.category] || groups.doc).push(s);
  });
  let html = "";
  for (const cat of ["doc", "code", "gen"]) {
    const items = groups[cat];
    if (!items.length) continue;
    html += `<div class="sug-group"><div class="sug-cat">${CAT_LABEL[cat]}</div>`;
    for (const it of items) {
      html += `<button class="sug-chip sug-${cat}" data-i="${it._i}">${escapeHtml(it.text)}</button>`;
    }
    html += "</div>";
  }
  suggestGroupsEl.innerHTML = html;
  suggestGroupsEl.querySelectorAll(".sug-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const it = state.suggestions[+chip.dataset.i];
      if (it) ask(it.text);
    });
  });
}

function setCollapsed(collapsed) {
  suggestPanel.classList.toggle("collapsed", collapsed);
  collapseBtn.textContent = collapsed ? "展开 ▾" : "收起 ▴";
  collapseBtn.title = collapsed ? "展开建议" : "收起建议";
}

// ---------- 对话 ----------
function addMessage(role, text, contexts) {
  if (emptyHintEl) emptyHintEl.remove();
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  wrap.innerHTML =
    `<div class="role">${role === "user" ? "你" : "Copilot"}</div>` +
    `<div class="bubble">${role === "user" ? escapeHtml(text) : renderMd(text)}</div>`;
  if (role === "assistant" && contexts && contexts.length) {
    const cites = contexts.map((c) => `<div class="ctx">${escapeHtml(c)}</div>`).join("");
    wrap.innerHTML += `<details class="citations"><summary>📎 检索依据（${contexts.length} 条）</summary>${cites}</details>`;
  }
  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;
  return wrap;
}

function addTyping() {
  const wrap = document.createElement("div");
  wrap.className = "msg assistant";
  wrap.innerHTML = `<div class="role">Copilot</div><div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>`;
  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;
  return wrap;
}

function ask(text) {
  inputEl.value = text;
  submit();
}

async function submit() {
  const q = inputEl.value.trim();
  if (!q || state.loading) return;
  if (!state.hasAsked) { state.hasAsked = true; setCollapsed(true); }
  inputEl.value = "";
  inputEl.style.height = "auto";
  addMessage("user", q);
  const typing = addTyping();
  state.loading = true;
  $("#sendBtn").disabled = true;
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q }),
    });
    const data = await res.json();
    typing.remove();
    if (data.error) addMessage("assistant", "⚠️ " + data.error);
    else addMessage("assistant", data.answer, data.contexts);
  } catch (e) {
    typing.remove();
    addMessage("assistant", "⚠️ 请求失败：" + e);
  } finally {
    state.loading = false;
    $("#sendBtn").disabled = false;
  }
}

// ---------- 事件 ----------
$("#refreshBtn").addEventListener("click", loadSuggestions);
$("#collapseBtn").addEventListener("click", () => setCollapsed(!suggestPanel.classList.contains("collapsed")));
$("#sendBtn").addEventListener("click", submit);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); submit(); }
});
inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px";
});

loadSuggestions();
