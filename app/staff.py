"""Staff approval page: review, edit, approve or reject reply drafts.

Mounted on the main app, so the webhook and ``/staff`` run in one process
and share the SQLite queue and WhatsApp client. Approving a draft is the only
path by which an agent reply reaches a patient.

No authentication: this is a local demo. Do not expose it publicly.
"""

import html
import logging
from typing import Dict, List, Optional
from urllib.parse import parse_qs

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.approval import Draft, DraftQueue
from app.config import mask_phone_number
from app.dependencies import get_draft_queue, get_whatsapp_client
from app.whatsapp.client import WhatsAppClient, WhatsAppClientError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Staff"])

# Meta error code for "Recipient phone number not in allowed list": sending to a
# number that isn't a verified tester on a WhatsApp Cloud API test number (e.g. the
# synthetic sender of Meta's dashboard test/simulator event).
META_RECIPIENT_NOT_ALLOWED_ERROR_CODE = 131030

# Fixed notice texts; the query string only ever selects one of these.
_NOTICES: Dict[str, str] = {
    "sent": "Draft #{id} approved and sent to the patient.",
    "saved": "Draft #{id} updated. It is still waiting for approval.",
    "rejected": "Draft #{id} rejected. Nothing was sent.",
    "already_handled": "Draft #{id} was already handled by someone else.",
    "not_found": "Draft #{id} does not exist.",
    "invalid_text": "Draft #{id} was not changed: the reply must be 1-4096 characters.",
    "send_failed": "Draft #{id} could not be sent (WhatsApp error). It is back in the queue; try again.",
    "recipient_not_allowed": (
        "Draft #{id} could not be sent: this number is not an allowed recipient on the WhatsApp test number. "
        "It is back in the queue."
    ),
}


async def _form(request: Request) -> Dict[str, str]:
    """Parse an ``application/x-www-form-urlencoded`` body without extra dependencies."""
    parsed = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
    return {key: values[0] for key, values in parsed.items()}


def _redirect(notice: str, draft_id: int) -> RedirectResponse:
    return RedirectResponse(url=f"/staff?notice={notice}&id={draft_id}", status_code=303)


@router.post("/staff/drafts/{draft_id}/approve")
async def approve_draft(
    draft_id: int,
    request: Request,
    queue: DraftQueue = Depends(get_draft_queue),
    wa_client: WhatsAppClient = Depends(get_whatsapp_client),
):
    text = (await _form(request)).get("text", "")
    if queue.get(draft_id) is None:
        return _redirect("not_found", draft_id)
    try:
        claimed = queue.claim_for_sending(draft_id, text)
    except ValueError:
        return _redirect("invalid_text", draft_id)
    if not claimed:
        return _redirect("already_handled", draft_id)

    draft = queue.get(draft_id)
    try:
        await wa_client.send_text(to=draft.sender, body=draft.draft_text)
    except Exception as exc:
        queue.release(draft_id)
        masked = mask_phone_number(draft.sender)
        if isinstance(exc, WhatsAppClientError) and exc.error_code == META_RECIPIENT_NOT_ALLOWED_ERROR_CODE:
            logger.info("Draft %d not sent: %s is not an allowed test recipient", draft_id, masked)
            return _redirect("recipient_not_allowed", draft_id)
        logger.error("Draft %d delivery to %s failed: %s", draft_id, masked, type(exc).__name__)
        logger.debug("Draft delivery failure detail", exc_info=exc)
        return _redirect("send_failed", draft_id)

    queue.mark_sent(draft_id)
    logger.info("Draft %d approved and sent to %s", draft_id, mask_phone_number(draft.sender))
    return _redirect("sent", draft_id)


@router.post("/staff/drafts/{draft_id}/edit")
async def edit_draft(draft_id: int, request: Request, queue: DraftQueue = Depends(get_draft_queue)):
    text = (await _form(request)).get("text", "")
    if queue.get(draft_id) is None:
        return _redirect("not_found", draft_id)
    try:
        updated = queue.update_text(draft_id, text)
    except ValueError:
        return _redirect("invalid_text", draft_id)
    return _redirect("saved" if updated else "already_handled", draft_id)


@router.post("/staff/drafts/{draft_id}/reject")
async def reject_draft(draft_id: int, queue: DraftQueue = Depends(get_draft_queue)):
    if queue.get(draft_id) is None:
        return _redirect("not_found", draft_id)
    return _redirect("rejected" if queue.reject(draft_id) else "already_handled", draft_id)


@router.get("/staff", response_class=HTMLResponse)
async def staff_page(
    notice: Optional[str] = None,
    id: Optional[int] = None,
    queue: DraftQueue = Depends(get_draft_queue),
):
    notice_text = _NOTICES[notice].format(id=id) if notice in _NOTICES and id is not None else None
    return HTMLResponse(_render_page(queue.list_pending(), queue.list_recent(10), notice_text))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_e = html.escape

_STYLE = """
:root { --bg:#f6f7f9; --card:#fff; --text:#1c1f24; --muted:#5f6670; --border:#dde1e6;
        --urgent:#c62828; --urgent-bg:#fdecea; --accent:#0b6e4f; --notice-bg:#e8f4ef; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#15171a; --card:#1f2226; --text:#e8eaed; --muted:#a0a6ad; --border:#33373d;
          --urgent:#ff6b6b; --urgent-bg:#3a1f1f; --accent:#4cc38a; --notice-bg:#1d3329; }
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text);
       font:15px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans Devanagari",sans-serif; }
main { max-width:860px; margin:0 auto; padding:16px; }
h1 { font-size:1.35rem; margin:8px 0 4px; }
h2 { font-size:1.05rem; margin:28px 0 10px; color:var(--muted); }
.sub { color:var(--muted); margin:0 0 16px; }
.notice { background:var(--notice-bg); border:1px solid var(--accent); border-radius:8px; padding:10px 12px; margin-bottom:16px; }
.card { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:14px; margin-bottom:14px; }
.card.urgent { border:2px solid var(--urgent); background:var(--urgent-bg); }
.meta { display:flex; flex-wrap:wrap; gap:8px 14px; color:var(--muted); font-size:.85rem; margin-bottom:8px; }
.badge { background:var(--urgent); color:#fff; font-weight:700; border-radius:4px; padding:1px 7px; letter-spacing:.03em; }
.label { font-size:.8rem; font-weight:600; color:var(--muted); margin:10px 0 4px; }
.patient { white-space:pre-wrap; overflow-wrap:anywhere; }
textarea { width:100%; min-height:140px; font:inherit; color:var(--text); background:var(--card);
           border:1px solid var(--border); border-radius:6px; padding:8px; }
.actions { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
button { font:inherit; border-radius:6px; padding:7px 14px; cursor:pointer; border:1px solid var(--border);
         background:var(--card); color:var(--text); }
button.approve { background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }
button.reject { color:var(--urgent); border-color:var(--urgent); }
.empty { color:var(--muted); }
.recent { font-size:.9rem; }
.recent .card { padding:10px 12px; }
a { color:var(--accent); }
"""


def _pending_card(draft: Draft) -> str:
    urgent_class = " urgent" if draft.is_urgent else ""
    badge = '<span class="badge">URGENT</span>' if draft.is_urgent else ""
    base = f"/staff/drafts/{draft.id}"
    return f"""
<section class="card{urgent_class}">
  <div class="meta">{badge}<span>Draft #{draft.id}</span><span>From +{_e(draft.sender)}</span><span>{_e(draft.timestamp)} UTC</span></div>
  <div class="label">Patient wrote</div>
  <div class="patient">{_e(draft.patient_message)}</div>
  <form method="post" action="{base}/approve">
    <div class="label"><label for="text-{draft.id}">Reply draft (edit before approving if needed)</label></div>
    <textarea id="text-{draft.id}" name="text" maxlength="4096" required>{_e(draft.draft_text)}</textarea>
    <div class="actions">
      <button class="approve" type="submit" formaction="{base}/approve">Approve &amp; send</button>
      <button type="submit" formaction="{base}/edit">Save edit</button>
      <button class="reject" type="submit" formaction="{base}/reject" formnovalidate>Reject</button>
    </div>
  </form>
</section>"""


def _recent_card(draft: Draft) -> str:
    return f"""
<section class="card">
  <div class="meta"><span>Draft #{draft.id}</span><span>+{_e(draft.sender)}</span><span>{_e(draft.status.value)}</span><span>{_e(draft.updated_at)} UTC</span></div>
  <div class="patient">{_e(draft.draft_text)}</div>
</section>"""


def _render_page(pending: List[Draft], recent: List[Draft], notice: Optional[str]) -> str:
    urgent_count = sum(1 for d in pending if d.is_urgent)
    notice_html = f'<div class="notice" role="status">{_e(notice)}</div>' if notice else ""
    pending_html = "".join(_pending_card(d) for d in pending) or '<p class="empty">No drafts waiting for approval.</p>'
    recent_html = "".join(_recent_card(d) for d in recent) or '<p class="empty">Nothing handled yet.</p>'
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SmileCare Reply Approvals</title>
<style>{_STYLE}</style>
</head>
<body>
<main>
  <h1>SmileCare Dental: reply approvals</h1>
  <p class="sub">{len(pending)} waiting, {urgent_count} urgent. Nothing reaches a patient until it is approved here. <a href="/staff">Refresh</a></p>
  {notice_html}
  {pending_html}
  <h2>Recently handled</h2>
  <div class="recent">{recent_html}</div>
</main>
</body>
</html>"""
