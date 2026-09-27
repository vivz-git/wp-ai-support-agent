"""Patient chat simulator: exercise the real agent without WhatsApp.

A demo-only WhatsApp-lookalike: typed messages run through the exact same
``process_incoming_message`` pipeline as a real webhook delivery (guardrails,
escalation policy, lead extraction, ``clinic_faq_lookup``, grounding) and
land on ``/staff`` as ordinary drafts. The only special-casing is on the
*sending* side (see ``app.staff.approve_draft``): a draft for a simulated
sender is never handed to ``WhatsAppClient`` — it is simply marked delivered
so it shows up back here instead.

Every simulated sender id starts with ``SIM-``; that prefix is the single
source of truth ``app.staff`` uses to skip the real WhatsApp call, so it must
never collide with a real WhatsApp number (which is digits only).

No authentication: this is a local demo, same as ``/staff``.
"""

import html
import json
import logging
import re
import uuid
from typing import Dict, List, Optional
from urllib.parse import parse_qs, quote

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.agent.orchestrator import AgentOrchestrator
from app.approval import Draft, DraftQueue, DraftStatus
from app.dependencies import get_draft_queue, get_memory, get_orchestrator
from app.memory import InMemoryConversationMemory
from app.turn_pipeline import process_incoming_message
from app.ui_style import BASE_STYLE

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Simulator"])

SIM_SENDER_PREFIX = "SIM-"
_MAX_SENDER_LEN = 64  # matches ConversationState.sender_id
_PREFIX_RE = re.compile(r"^sim[-_]?", re.IGNORECASE)
_UNSAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9_-]+")

# (sender, label, opening message) — five ready-made conversations covering
# the scenarios worth showing in a demo. Fixed ids so re-clicking the button
# continues the same five conversations instead of spawning new ones each time.
DEMO_SCENARIOS: List[Dict[str, str]] = [
    {
        "sender": "SIM-DEMO-PRICE",
        "label": "Hinglish price question",
        "message": "RCT ka kitna lagega?",
    },
    {
        "sender": "SIM-DEMO-BOOKING",
        "label": "Booking with details",
        "message": "Hi, I'm Ananya, I'd like to book a teeth cleaning for Saturday morning.",
    },
    {
        "sender": "SIM-DEMO-EMERGENCY",
        "label": "Emergency (Hindi)",
        "message": "मेरे दाँत में बहुत दर्द हो रहा है, रात भर सो नहीं पाया",
    },
    {
        "sender": "SIM-DEMO-ANGRY",
        "label": "Angry patient",
        "message": "This is ridiculous!!! I was charged twice for my last visit and no one replied. Worst service!",
    },
    {
        "sender": "SIM-DEMO-SPAM",
        "label": "Spam",
        "message": "Congratulations!!! You have won a free iPhone. Click http://win-prize.example.invalid to claim now",
    },
]

_NOTICES: Dict[str, str] = {
    "invalid_sender": "Enter a short patient id (letters, numbers, - or _).",
    "demo_loaded": "Loaded 5 demo conversations. Pick another one below to see the rest.",
    "empty_text": "Type a message first.",
}


def is_simulated_sender(sender: str) -> bool:
    """Whether ``sender`` is a simulator patient (never a real WhatsApp number)."""
    return sender.startswith(SIM_SENDER_PREFIX)


def normalize_sim_sender(raw: str) -> Optional[str]:
    """Turn free-typed input into a canonical ``SIM-...`` id, or ``None`` if empty.

    Accepts a bare label ("priya"), one already carrying the prefix in any
    case/separator ("sim_priya", "SIM-Priya"), or a full id — all normalize
    to the same ``SIM-PRIYA`` so re-typing a name reopens the same chat.
    """
    stripped = (raw or "").strip()
    if not stripped:
        return None
    without_prefix = _PREFIX_RE.sub("", stripped, count=1)
    safe = _UNSAFE_CHARS_RE.sub("-", without_prefix).strip("-").upper()
    if not safe:
        return None
    return f"{SIM_SENDER_PREFIX}{safe}"[:_MAX_SENDER_LEN]


async def _form(request: Request) -> Dict[str, str]:
    parsed = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
    return {key: values[0] for key, values in parsed.items()}


def _redirect(sender: Optional[str] = None, notice: Optional[str] = None) -> RedirectResponse:
    params = []
    if sender:
        params.append(f"sender={quote(sender)}")
    if notice:
        params.append(f"notice={notice}")
    url = "/simulate" + ("?" + "&".join(params) if params else "")
    return RedirectResponse(url=url, status_code=303)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/simulate/send")
async def send_message(
    request: Request,
    memory: InMemoryConversationMemory = Depends(get_memory),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    drafts: DraftQueue = Depends(get_draft_queue),
):
    form = await _form(request)
    sender = normalize_sim_sender(form.get("sender", ""))
    if sender is None:
        return _redirect(notice="invalid_sender")
    text = (form.get("text") or "").strip()
    if not text:
        return _redirect(sender=sender, notice="empty_text")

    await process_incoming_message(
        sender=sender,
        text=text,
        message_id=f"sim-{uuid.uuid4().hex}",
        memory=memory,
        orchestrator=orchestrator,
        drafts=drafts,
    )
    return _redirect(sender=sender)


@router.post("/simulate/demo")
async def load_demo_scenarios(
    memory: InMemoryConversationMemory = Depends(get_memory),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    drafts: DraftQueue = Depends(get_draft_queue),
):
    for scenario in DEMO_SCENARIOS:
        await process_incoming_message(
            sender=scenario["sender"],
            text=scenario["message"],
            message_id=f"sim-demo-{uuid.uuid4().hex}",
            memory=memory,
            orchestrator=orchestrator,
            drafts=drafts,
        )
    return _redirect(sender=DEMO_SCENARIOS[0]["sender"], notice="demo_loaded")


@router.get("/simulate/messages", response_class=HTMLResponse)
async def chat_fragment(sender: str, queue: DraftQueue = Depends(get_draft_queue)):
    """Just the chat-log HTML, for the page's own poll-and-replace script."""
    normalized = normalize_sim_sender(sender) or sender
    return HTMLResponse(_render_chat_log(queue.list_by_sender(normalized)))


@router.get("/simulate", response_class=HTMLResponse)
async def simulate_page(
    sender: Optional[str] = None,
    notice: Optional[str] = None,
    queue: DraftQueue = Depends(get_draft_queue),
):
    normalized = normalize_sim_sender(sender) if sender else None
    if sender and normalized is None:
        notice = "invalid_sender"
    notice_text = _NOTICES.get(notice)
    known_senders = queue.list_senders(prefix=SIM_SENDER_PREFIX)
    chat_log = _render_chat_log(queue.list_by_sender(normalized)) if normalized else ""
    return HTMLResponse(_render_page(normalized, known_senders, chat_log, notice_text))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_e = html.escape

_SIM_STYLE = """
#chat-log { display:flex; flex-direction:column; gap:10px; min-height:120px; max-height:55vh; overflow-y:auto;
            padding:12px; background:var(--card); border:1px solid var(--border); border-radius:10px; }
.bubble { max-width:80%; padding:8px 12px; border-radius:12px; }
.bubble.patient { align-self:flex-start; background:var(--bubble-patient); border:1px solid var(--border);
                  border-bottom-left-radius:2px; }
.bubble.agent { align-self:flex-end; background:var(--bubble-agent); border-bottom-right-radius:2px; }
.bubble.agent.pending { background:var(--bubble-pending); color:var(--muted); font-style:italic; }
.bubble-text { white-space:pre-wrap; overflow-wrap:anywhere; }
.bubble-meta { font-size:.72rem; color:var(--muted); margin-top:4px; }
.system-note { align-self:center; color:var(--muted); font-size:.78rem; font-style:italic; }
.sender-list { display:flex; flex-wrap:wrap; gap:8px; }
.compose { display:flex; gap:8px; margin-top:12px; }
.compose input[type=text] { flex:1; min-width:0; }
@media (max-width: 480px) {
  #chat-log { max-height:45vh; }
  .compose { flex-direction:column; }
}
"""


def _patient_bubble(draft: Draft) -> str:
    urgent = ' <span class="badge">URGENT</span>' if draft.is_urgent else ""
    return (
        f'<div class="bubble patient"><div class="bubble-text">{_e(draft.patient_message)}</div>'
        f'<div class="bubble-meta">{_e(draft.timestamp)} UTC{urgent}</div></div>'
    )


def _agent_bubble(draft: Draft) -> str:
    if draft.status == DraftStatus.SENT:
        return (
            f'<div class="bubble agent"><div class="bubble-text">{_e(draft.draft_text)}</div>'
            f'<div class="bubble-meta">{_e(draft.updated_at)} UTC</div></div>'
        )
    if draft.status == DraftStatus.REJECTED:
        return '<div class="system-note">Staff rejected this reply — nothing was sent.</div>'
    # pending or (briefly) sending: staff has not approved this yet.
    return '<div class="bubble agent pending">⏳ Waiting for staff approval on <a href="/staff">/staff</a>…</div>'


def _render_chat_log(drafts_for_sender: List[Draft]) -> str:
    if not drafts_for_sender:
        return '<p class="empty">No messages yet — say hello below.</p>'
    parts = []
    for draft in drafts_for_sender:
        parts.append(_patient_bubble(draft))
        parts.append(_agent_bubble(draft))
    return "\n".join(parts)


def _sender_links(known_senders: List[str], active: Optional[str]) -> str:
    if not known_senders:
        return '<span class="empty">None yet — start one below or load a demo scenario.</span>'
    links = []
    for sender in known_senders:
        css = "btn approve" if sender == active else "btn"
        links.append(f'<a class="{css}" href="/simulate?sender={quote(sender)}">{_e(sender)}</a>')
    return "\n".join(links)


def _chat_panel(sender: str, chat_log: str) -> str:
    return f"""
<h2>Chatting as {_e(sender)}</h2>
<div id="chat-log">{chat_log}</div>
<form method="post" action="/simulate/send" class="compose">
  <input type="hidden" name="sender" value="{_e(sender)}">
  <input type="text" name="text" placeholder="Type a message as the patient..." required autocomplete="off" maxlength="4096">
  <button class="approve" type="submit">Send</button>
</form>
<script>
(function() {{
  var sender = {json.dumps(sender)};
  function poll() {{
    fetch("/simulate/messages?sender=" + encodeURIComponent(sender))
      .then(function(r) {{ return r.text(); }})
      .then(function(fragment) {{
        var el = document.getElementById("chat-log");
        if (el && el.innerHTML !== fragment) {{
          el.innerHTML = fragment;
          el.scrollTop = el.scrollHeight;
        }}
      }})
      .catch(function() {{}});
  }}
  setInterval(poll, 2000);
  var log = document.getElementById("chat-log");
  if (log) {{ log.scrollTop = log.scrollHeight; }}
}})();
</script>"""


def _render_page(sender: Optional[str], known_senders: List[str], chat_log: str, notice: Optional[str]) -> str:
    notice_html = f'<div class="notice" role="status">{_e(notice)}</div>' if notice else ""
    chat_panel = _chat_panel(sender, chat_log) if sender else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SmileCare Patient Simulator</title>
<style>{BASE_STYLE}{_SIM_STYLE}</style>
</head>
<body>
<main>
  <nav class="top-links"><a href="/staff">Staff approvals</a><span>·</span><a href="/simulate">Simulator</a></nav>
  <h1>Simulate a patient chat</h1>
  <p class="sub">No real WhatsApp involved. Messages run through the real agent and land on
    <a href="/staff">/staff</a> as drafts; approving one for a <code>SIM-</code> patient shows the reply
    here instead of calling WhatsApp.</p>
  {notice_html}
  <form method="post" action="/simulate/demo">
    <button type="submit">Load demo scenario (5 conversations)</button>
  </form>
  <div class="card" style="margin-top:14px">
    <div class="label">Existing simulated patients</div>
    <div class="sender-list">{_sender_links(known_senders, sender)}</div>
    <div class="label" style="margin-top:14px">Start a new one</div>
    <form method="get" action="/simulate" class="compose">
      <input type="text" name="sender" placeholder="e.g. priya or 001" maxlength="40">
      <button type="submit">Start chat</button>
    </form>
  </div>
  {chat_panel}
</main>
</body>
</html>"""
