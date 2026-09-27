"""Tests for the SQLite approval queue (app/approval.py) and the /staff page (app/staff.py)."""

import threading

import pytest
from fastapi.testclient import TestClient

from app.approval import MAX_DRAFT_LENGTH, DraftQueue, DraftStatus
from tests.conftest import MockWhatsAppClient

SENDER = "919876543210"


@pytest.fixture
def queue() -> DraftQueue:
    q = DraftQueue(":memory:")
    yield q
    q.close()


# ---------------------------------------------------------------------------
# DraftQueue
# ---------------------------------------------------------------------------


def test_add_stores_a_pending_draft_with_all_columns(queue):
    draft = queue.add(SENDER, "cleaning price?", "Cleaning is ₹800–₹1,500.", is_urgent=False)
    assert draft.id == 1
    assert (draft.sender, draft.patient_message, draft.draft_text) == (SENDER, "cleaning price?", "Cleaning is ₹800–₹1,500.")
    assert draft.status == DraftStatus.PENDING
    assert draft.is_urgent is False
    assert draft.timestamp and draft.updated_at


def test_table_is_named_drafts_with_the_requested_columns(queue):
    columns = {row[1] for row in queue._conn.execute("PRAGMA table_info(drafts)")}
    assert {"sender", "patient_message", "draft_text", "status", "is_urgent", "timestamp"} <= columns


def test_pending_lists_urgent_first_then_oldest_first(queue):
    a = queue.add("1", "a", "reply a", is_urgent=False)
    b = queue.add("2", "b", "reply b", is_urgent=True)
    c = queue.add("3", "c", "reply c", is_urgent=False)
    d = queue.add("4", "d", "reply d", is_urgent=True)
    assert [x.id for x in queue.list_pending()] == [b.id, d.id, a.id, c.id]


def test_claim_is_atomic_so_a_draft_is_sent_at_most_once(queue):
    draft = queue.add(SENDER, "hi", "hello", is_urgent=False)
    assert queue.claim_for_sending(draft.id, "hello!") is True
    assert queue.claim_for_sending(draft.id, "hello again") is False
    assert queue.get(draft.id).status == DraftStatus.SENDING
    assert queue.get(draft.id).draft_text == "hello!"
    assert queue.mark_sent(draft.id) is True
    assert queue.get(draft.id).status == DraftStatus.SENT
    assert queue.list_pending() == []


def test_concurrent_claims_only_one_wins(queue):
    draft = queue.add(SENDER, "hi", "hello", is_urgent=False)
    wins = []
    barrier = threading.Barrier(8)

    def claim():
        barrier.wait()
        wins.append(queue.claim_for_sending(draft.id, "hello"))

    threads = [threading.Thread(target=claim) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert wins.count(True) == 1


def test_release_returns_a_failed_send_to_pending(queue):
    draft = queue.add(SENDER, "hi", "hello", is_urgent=True)
    queue.claim_for_sending(draft.id, "hello")
    assert queue.release(draft.id) is True
    assert queue.get(draft.id).status == DraftStatus.PENDING
    assert [d.id for d in queue.list_pending()] == [draft.id]


def test_edit_and_reject_only_apply_to_pending_drafts(queue):
    draft = queue.add(SENDER, "hi", "hello", is_urgent=False)
    assert queue.update_text(draft.id, "  edited  ") is True
    assert queue.get(draft.id).draft_text == "edited"
    assert queue.reject(draft.id) is True
    assert queue.get(draft.id).status == DraftStatus.REJECTED
    assert queue.update_text(draft.id, "too late") is False
    assert queue.reject(draft.id) is False
    assert queue.claim_for_sending(draft.id, "too late") is False


@pytest.mark.parametrize("bad", ["", "   ", "x" * (MAX_DRAFT_LENGTH + 1)])
def test_invalid_text_is_rejected(queue, bad):
    with pytest.raises(ValueError):
        queue.add(SENDER, "hi", bad, is_urgent=False)
    draft = queue.add(SENDER, "hi", "ok", is_urgent=False)
    with pytest.raises(ValueError):
        queue.update_text(draft.id, bad)
    with pytest.raises(ValueError):
        queue.claim_for_sending(draft.id, bad)
    assert queue.get(draft.id).status == DraftStatus.PENDING


def test_recent_lists_sent_and_rejected_newest_first(queue):
    a = queue.add("1", "a", "a", is_urgent=False)
    b = queue.add("2", "b", "b", is_urgent=False)
    queue.add("3", "c", "c", is_urgent=False)  # stays pending
    queue.reject(a.id)
    queue.claim_for_sending(b.id, "b")
    queue.mark_sent(b.id)
    assert {d.id for d in queue.list_recent()} == {a.id, b.id}


def test_drafts_persist_across_connections_to_the_same_file(tmp_path):
    path = str(tmp_path / "drafts.db")
    first = DraftQueue(path)
    draft = first.add(SENDER, "hi", "hello", is_urgent=True)
    first.close()
    second = DraftQueue(path)
    assert [(d.id, d.is_urgent) for d in second.list_pending()] == [(draft.id, True)]
    second.close()


# ---------------------------------------------------------------------------
# /staff page
# ---------------------------------------------------------------------------


def test_staff_page_lists_pending_drafts_and_escapes_html(client: TestClient, test_drafts):
    test_drafts.add(SENDER, "<script>alert(1)</script>", "Reply with <b>bold</b>", is_urgent=False)
    page = client.get("/staff")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert "<script>alert(1)</script>" not in page.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page.text
    assert "Reply with &lt;b&gt;bold&lt;/b&gt;" in page.text
    assert "1 waiting, 0 urgent" in page.text


def test_staff_page_highlights_urgent_drafts_first(client: TestClient, test_drafts):
    test_drafts.add("919800000001", "cleaning price?", "Cleaning is ₹800–₹1,500.", is_urgent=False)
    test_drafts.add("919800000002", "tooth broke", "The clinic will call you back shortly.", is_urgent=True)
    page = client.get("/staff").text
    assert page.index("919800000002") < page.index("919800000001")
    assert 'class="card urgent"' in page
    assert "2 waiting, 1 urgent" in page


def test_empty_queue_renders(client: TestClient):
    assert "No drafts waiting for approval." in client.get("/staff").text


def test_notice_query_only_selects_fixed_messages(client: TestClient):
    page = client.get("/staff", params={"notice": "<script>x</script>", "id": 1}).text
    assert "<script>x</script>" not in page and "&lt;script&gt;" not in page
    assert "approved and sent" in client.get("/staff", params={"notice": "sent", "id": 7}).text


def test_approve_sends_edited_text_and_marks_sent(client: TestClient, test_drafts, mock_wa):
    draft = test_drafts.add(SENDER, "timings?", "We are open 10-8.", is_urgent=False)
    response = client.post(
        f"/staff/drafts/{draft.id}/approve",
        data={"text": "We're open 10am-8pm, Monday to Saturday."},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == f"/staff?notice=sent&id={draft.id}"
    assert mock_wa.sent_messages == [{"to": SENDER, "body": "We're open 10am-8pm, Monday to Saturday."}]
    stored = test_drafts.get(draft.id)
    assert stored.status == DraftStatus.SENT
    assert stored.draft_text == "We're open 10am-8pm, Monday to Saturday."


def test_double_approve_sends_once(client: TestClient, test_drafts, mock_wa):
    draft = test_drafts.add(SENDER, "hi", "hello", is_urgent=False)
    client.post(f"/staff/drafts/{draft.id}/approve", data={"text": "hello"}, follow_redirects=False)
    second = client.post(f"/staff/drafts/{draft.id}/approve", data={"text": "hello"}, follow_redirects=False)
    assert "notice=already_handled" in second.headers["location"]
    assert len(mock_wa.sent_messages) == 1


def test_edit_saves_text_without_sending(client: TestClient, test_drafts, mock_wa):
    draft = test_drafts.add(SENDER, "hi", "hello", is_urgent=False)
    response = client.post(f"/staff/drafts/{draft.id}/edit", data={"text": "Hello! How can we help?"}, follow_redirects=False)
    assert "notice=saved" in response.headers["location"]
    assert test_drafts.get(draft.id).draft_text == "Hello! How can we help?"
    assert test_drafts.get(draft.id).status == DraftStatus.PENDING
    assert mock_wa.sent_messages == []


def test_reject_sends_nothing(client: TestClient, test_drafts, mock_wa):
    draft = test_drafts.add(SENDER, "spam", "Thanks for your message.", is_urgent=False)
    response = client.post(f"/staff/drafts/{draft.id}/reject", follow_redirects=False)
    assert "notice=rejected" in response.headers["location"]
    assert test_drafts.get(draft.id).status == DraftStatus.REJECTED
    assert mock_wa.sent_messages == []
    assert test_drafts.list_pending() == []


def test_blank_text_is_refused_and_draft_stays_pending(client: TestClient, test_drafts, mock_wa):
    draft = test_drafts.add(SENDER, "hi", "hello", is_urgent=False)
    response = client.post(f"/staff/drafts/{draft.id}/approve", data={"text": "   "}, follow_redirects=False)
    assert "notice=invalid_text" in response.headers["location"]
    assert test_drafts.get(draft.id).status == DraftStatus.PENDING
    assert mock_wa.sent_messages == []


@pytest.mark.parametrize("action", ["approve", "edit", "reject"])
def test_unknown_draft_is_reported(client: TestClient, action):
    response = client.post(f"/staff/drafts/999/{action}", data={"text": "x"}, follow_redirects=False)
    assert response.status_code == 303
    assert "notice=not_found" in response.headers["location"]


def test_send_failure_puts_the_draft_back_in_the_queue(client: TestClient, test_drafts):
    from app.main import app, get_whatsapp_client
    from app.whatsapp.client import WhatsAppClientError

    class FailingClient(MockWhatsAppClient):
        async def send_text(self, to, body):
            raise WhatsAppClientError("Meta 500", error_code=1)

    app.dependency_overrides[get_whatsapp_client] = lambda: FailingClient()
    draft = test_drafts.add(SENDER, "hi", "hello", is_urgent=True)
    response = client.post(f"/staff/drafts/{draft.id}/approve", data={"text": "hello"}, follow_redirects=False)
    assert "notice=send_failed" in response.headers["location"]
    assert test_drafts.get(draft.id).status == DraftStatus.PENDING


def test_unicode_form_text_round_trips(client: TestClient, test_drafts, mock_wa):
    draft = test_drafts.add(SENDER, "दर्द", "draft", is_urgent=True)
    reply = "क्लिनिक आपको जल्द ही कॉल करेगा। ₹800–₹1,500 & more"
    client.post(f"/staff/drafts/{draft.id}/approve", data={"text": reply}, follow_redirects=False)
    assert mock_wa.sent_messages == [{"to": SENDER, "body": reply}]
