/* StreamVerse — frontend behaviour.
   The classifier returns six probabilities; the ladder shows all six in a fixed
   order so only the bars move between turns. */

const INTENT_ORDER = [
  "Information Query",
  "Content Search",
  "Music Recommendation",
  "Movie Recommendation",
  "TV Show Recommendation",
  "Podcast Recommendation",
];

/* A near-tie means the reading is unsettled. Shown as a caution, never hidden. */
const CLOSE_CALL_MARGIN = 0.05;

const el = (id) => document.getElementById(id);
const thread = el("thread");
const opening = el("opening");
const form = el("composer");
const input = el("composer-input");
const sendBtn = el("send");
const note = el("composer-note");
const ladder = el("ladder");
const suggestions = el("suggestions");

let busy = false;

/* ── ladder ─────────────────────────────────────────────────────────────── */

function buildLadder(intents) {
  ladder.replaceChildren();
  (intents && intents.length ? intents : INTENT_ORDER).forEach((name) => {
    const li = document.createElement("li");
    li.className = "rung";
    li.dataset.intent = name;

    const label = document.createElement("span");
    label.className = "rung-name";
    label.textContent = name;

    const value = document.createElement("span");
    value.className = "rung-value";
    value.textContent = "—";

    const track = document.createElement("span");
    track.className = "rung-track";
    const fill = document.createElement("span");
    fill.className = "rung-fill";
    track.append(fill);

    li.append(label, value, track);
    li.dataset.empty = "true";
    ladder.append(li);
  });
}

function resolveLadder(distribution, leadIntent) {
  const byIntent = new Map((distribution || []).map((d) => [d.intent, d.probability]));
  const max = Math.max(0.0001, ...byIntent.values());

  ladder.querySelectorAll(".rung").forEach((rung) => {
    const p = byIntent.get(rung.dataset.intent);
    const lead = rung.dataset.intent === leadIntent;
    rung.dataset.lead = String(lead);
    rung.dataset.empty = String(p === undefined);
    rung.querySelector(".rung-value").textContent =
      p === undefined ? "—" : p.toFixed(3);
    /* scaled against the top score so small spreads stay readable */
    rung.querySelector(".rung-fill").style.transform =
      `scaleX(${p === undefined ? 0 : Math.max(0.02, p / max)})`;
  });
}

function clearLadder() {
  ladder.querySelectorAll(".rung").forEach((rung) => {
    rung.dataset.lead = "false";
    rung.dataset.empty = "true";
    rung.querySelector(".rung-value").textContent = "—";
    rung.querySelector(".rung-fill").style.transform = "scaleX(0)";
  });
}

/* ── reading panel ──────────────────────────────────────────────────────── */

const LANGUAGE_NAME = {
  gu: "Gujarati",
  en: "English",
  unknown: "Not recognised",
};

function renderReading(data) {
  el("fact-language").textContent = LANGUAGE_NAME[data.language] || "—";

  const box = el("fact-tokens");
  box.replaceChildren();
  if (!data.tokens || !data.tokens.length) {
    box.textContent = "None";
  } else {
    const knownCount = data.known_tokens;
    data.tokens.forEach((t, i) => {
      const span = document.createElement("span");
      span.className = "token";
      span.textContent = t;
      /* Tokens beyond the known count aren't individually flagged by the API,
         so only mark state when every token is unseen. */
      if (knownCount === 0) span.dataset.known = "false";
      span.title = knownCount === 0 ? "Not in the training vocabulary" : "";
      box.append(span);
      if (i > 40) return;
    });
  }

  const unsure = data.margin !== null && data.margin !== undefined && data.margin < CLOSE_CALL_MARGIN;
  el("close-call").hidden = !unsure;

  if (data.distribution && data.distribution.length) {
    resolveLadder(data.distribution, data.intent);
  } else {
    clearLadder();
  }
  return unsure;
}

/* ── messages ───────────────────────────────────────────────────────────── */

function isGujarati(text) {
  return /[઀-૿]/.test(text);
}

function addYou(text) {
  if (opening && opening.isConnected) opening.remove();
  const div = document.createElement("div");
  div.className = "msg msg-you" + (isGujarati(text) ? " gu" : "");
  div.textContent = text;
  thread.append(div);
  scrollToEnd();
}

function addBot(data, unsure) {
  const wrap = document.createElement("div");
  wrap.className = "msg msg-bot";

  const p = document.createElement("p");
  p.className = "msg-bot-text gu";
  p.lang = "gu";
  p.textContent = data.response;
  wrap.append(p);

  const meta = document.createElement("div");
  meta.className = "msg-meta";

  if (data.intent) {
    const tag = document.createElement("span");
    tag.className = "intent-tag";
    tag.dataset.unsure = String(unsure);
    tag.textContent = unsure ? data.intent + " (unsettled)" : data.intent;
    meta.append(tag);

    if (data.confidence !== null && data.confidence !== undefined) {
      const score = document.createElement("span");
      score.textContent = "score " + data.confidence.toFixed(3);
      meta.append(score);
    }
  } else {
    const tag = document.createElement("span");
    tag.className = "intent-tag";
    tag.dataset.unsure = "true";
    tag.textContent = "No intent matched";
    meta.append(tag);
  }
  wrap.append(meta);

  if (data.note) {
    const n = document.createElement("p");
    n.className = "msg-note";
    n.textContent = data.note;
    wrap.append(n);
  }

  thread.append(wrap);
  scrollToEnd();
}

function addFailure(message) {
  const wrap = document.createElement("div");
  wrap.className = "msg msg-bot";
  const p = document.createElement("p");
  p.className = "msg-bot-text";
  p.textContent = message;
  wrap.append(p);
  thread.append(wrap);
  scrollToEnd();
}

function scrollToEnd() {
  thread.scrollTo({
    top: thread.scrollHeight,
    behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
  });
}

/* ── submit ─────────────────────────────────────────────────────────────── */

function setBusy(state) {
  busy = state;
  sendBtn.dataset.busy = String(state);
  sendBtn.disabled = state;
}

async function ask(text) {
  note.textContent = "";
  addYou(text);
  setBusy(true);

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      addFailure(
        body.detail ||
          "The classifier did not respond. Check the terminal running uvicorn, then send again."
      );
      clearLadder();
      return;
    }

    const data = await res.json();
    const unsure = renderReading(data);
    addBot(data, unsure);
  } catch (err) {
    addFailure("Could not reach the backend. Make sure uvicorn is running, then send again.");
    clearLadder();
  } finally {
    setBusy(false);
    input.focus();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) {
    note.textContent = "Type a question first.";
    input.focus();
    return;
  }
  if (busy) return;
  input.value = "";
  ask(text);
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    if (busy) return;
    const text = chip.dataset.fill;
    input.value = "";
    ask(text);
  });
});

/* ── theme ──────────────────────────────────────────────────────────────── */

const THEME_KEY = "streamverse-theme";
const THEME_COLOR = { light: "#F5F5FA", dark: "#0C0E1C" };

function applyTheme(theme, remember) {
  document.documentElement.dataset.theme = theme;

  const meta = el("theme-color");
  if (meta) meta.content = THEME_COLOR[theme];

  const btn = el("theme-toggle");
  if (btn) {
    btn.setAttribute(
      "aria-label",
      theme === "dark" ? "Switch to light theme" : "Switch to dark theme"
    );
  }

  if (remember) {
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch (e) {
      /* private browsing — the theme still applies for this session */
    }
  }
}

function initTheme() {
  let saved = null;
  try {
    saved = localStorage.getItem(THEME_KEY);
  } catch (e) {
    /* ignore */
  }

  const system = matchMedia("(prefers-color-scheme: dark)");
  applyTheme(saved || (system.matches ? "dark" : "light"), false);

  /* Follow the system only while the user has not made an explicit choice. */
  system.addEventListener("change", (e) => {
    let stored = null;
    try {
      stored = localStorage.getItem(THEME_KEY);
    } catch (err) {
      /* ignore */
    }
    if (!stored) applyTheme(e.matches ? "dark" : "light", false);
  });

  el("theme-toggle").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    applyTheme(next, true);
  });
}

/* ── startup ────────────────────────────────────────────────────────────── */

async function boot() {
  initTheme();
  buildLadder(INTENT_ORDER);

  const dot = el("status-dot");
  const text = el("status-text");

  try {
    const res = await fetch("/api/health");
    const h = await res.json();

    if (h.ready) {
      dot.dataset.state = "ready";
      const vocab = h.vocabulary || {};
      text.textContent = `Ready · ${vocab.gu ?? 0} Gujarati and ${vocab.en ?? 0} English terms`;
      if (h.intents && h.intents.length) buildLadder(h.intents);
      suggestions.hidden = false;
      if (matchMedia("(min-width: 901px)").matches) input.focus();
    } else {
      dot.dataset.state = "down";
      text.textContent = "Model not loaded";
      addFailure(
        h.error || "The model bundle could not be loaded. See the terminal for details."
      );
      input.disabled = true;
      sendBtn.disabled = true;
    }
  } catch (err) {
    dot.dataset.state = "down";
    text.textContent = "Backend unreachable";
    addFailure("Start the backend with: uvicorn app:app --reload --port 8000");
    input.disabled = true;
    sendBtn.disabled = true;
  }
}

boot();
