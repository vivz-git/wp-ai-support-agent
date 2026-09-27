"""Tests for the patient chat simulator (app/simulate.py).

Every message sent through /simulate must run through the exact same
pipeline as a real webhook message (app.turn_pipeline.process_incoming_message)
and land on /staff as an ordinary draft. The one thing that must always
differ is delivery: approving a draft for a ``SIM-`` sender must never call
the WhatsApp client, in contrast to a real phone number sender, which must.
"""

from fastapi.testclient import TestClient

from app.approval import DraftStatus
from app.simulate import DEMO_SCENARIOS, is_simulated_sender, normalize_sim_sender

REAL_SENDER = "919876543210"


# ---------------------------------------------------------------------------
# normalize_sim_sender / is_simulated_sender
# ---------------------------------------------------------------------------


def test_is_simulated_sender():
    assert is_simulated_sender("SIM-PRIYA") is True
    assert is_simulated_sender("919876543210") is False
    assert is_simulated_sender("SIMPLETON") is False  # must be the exact "SIM-" prefix


def test_normalize_bare_label_gets_prefixed():
    assert normalize_sim_sender("priya") == "SIM-PRIYA"
    assert normalize_sim_sender("  001  ") == "SIM-001"


def test_normalize_is_idempotent_across_separators_and_case():
    for raw in ("priya", "SIM-priya", "sim_priya", "simpriya", "SIM-PRIYA"):
        assert normalize_sim_sender(raw) == "SIM-PRIYA"


def test_normalize_rejects_blank_input():
    assert normalize_sim_sender("") is None
    assert normalize_sim_sender("   ") is None
    assert normalize_sim_sender("SIM-") is None  # prefix with nothing after it


def test_normalize_strips_unsafe_characters():
    assert normalize_sim_sender("priya sharma!!") == "SIM-PRIYA-SHARMA"


def test_normalize_is_bounded_to_sender_id_max_length():
    result = normalize_sim_sender("x" * 200)
    assert result is not None
    assert len(result) <= 64


# ---------------------------------------------------------------------------
# GET /simulate
# ---------------------------------------------------------------------------


def test_simulate_page_loads_empty(client: TestClient):
    response = client.get("/simulate")
    assert response.status_code == 200
    assert "Simulate a patient chat" in response.text
    assert "None yet" in response.text


def test_simulate_page_normalizes_query_param_sender(client: TestClient):
    response = client.get("/simulate", params={"sender": "priya"})
    assert response.status_code == 200
    assert "Chatting as SIM-PRIYA" in response.text
    assert "No messages yet" in response.text


def test_simulate_page_rejects_blank_sender_with_notice(client: TestClient):
    response = client.get("/simulate", params={"sender": "   "})
    assert response.status_code == 200
    assert "Enter a short patient id" in response.text
    assert "Chatting as" not in response.text


# ---------------------------------------------------------------------------
# POST /simulate/send runs the real pipeline
# ---------------------------------------------------------------------------


def test_send_message_runs_the_real_orchestrator_and_queues_a_draft(
    client: TestClient, mock_llm, test_drafts
):
    response = client.post(
        "/simulate/send", data={"sender": "priya", "text": "RCT ka kitna lagega?"}, follow_redirects=False
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/simulate?sender=SIM-PRIYA"

    # Same pipeline as the real webhook: the LLM was actually called.
    assert len(mock_llm.calls) >= 1
    [draft] = test_drafts.list_pending()
    assert draft.sender == "SIM-PRIYA"
    assert draft.patient_message == "RCT ka kitna lagega?"
    assert draft.draft_text == mock_llm.response_text
    assert draft.status == DraftStatus.PENDING


def test_send_message_normalizes_sender_consistently_across_calls(client: TestClient, test_drafts):
    client.post("/simulate/send", data={"sender": "priya", "text": "hello"}, follow_redirects=False)
    client.post("/simulate/send", data={"sender": "SIM-Priya", "text": "second message"}, follow_redirects=False)
    drafts = test_drafts.list_by_sender("SIM-PRIYA")
    assert [d.patient_message for d in drafts] == ["hello", "second message"]


def test_send_rejects_blank_text(client: TestClient, test_drafts):
    response = client.post("/simulate/send", data={"sender": "priya", "text": "   "}, follow_redirects=False)
    assert response.status_code == 303
    assert "notice=empty_text" in response.headers["location"]
    assert test_drafts.list_pending() == []


def test_send_rejects_blank_sender(client: TestClient, test_drafts):
    response = client.post("/simulate/send", data={"sender": "   ", "text": "hi"}, follow_redirects=False)
    assert "notice=invalid_sender" in response.headers["location"]
    assert test_drafts.list_pending() == []


def test_emergency_message_escalates_without_calling_the_llm(client: TestClient, mock_llm, test_drafts):
    response = client.post(
        "/simulate/send",
        data={"sender": "priya", "text": "मेरे दाँत में बहुत दर्द हो रहा है"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert mock_llm.calls == []
    [draft] = test_drafts.list_pending()
    assert draft.is_urgent is True


# ---------------------------------------------------------------------------
# The core guarantee: SIM- approvals never touch WhatsApp; real ones do
# ---------------------------------------------------------------------------


def test_approving_a_simulated_draft_never_calls_whatsapp_client(client: TestClient, mock_wa, test_drafts):
    client.post("/simulate/send", data={"sender": "priya", "text": "hi there"}, follow_redirects=False)
    [draft] = test_drafts.list_pending()

    response = client.post(f"/staff/drafts/{draft.id}/approve", data={"text": "Hello Priya!"}, follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == f"/staff?notice=sent&id={draft.id}"
    assert mock_wa.sent_messages == []  # the whole point: no real WhatsApp call
    assert test_drafts.get(draft.id).status == DraftStatus.SENT
    assert test_drafts.get(draft.id).draft_text == "Hello Priya!"


def test_approving_a_real_sender_draft_still_calls_whatsapp_client(client: TestClient, mock_wa, test_drafts):
    """Contrast case: proves the SIM- skip is sender-specific, not a general regression."""
    test_drafts.add(sender=REAL_SENDER, patient_message="hi", draft_text="Hello!", is_urgent=False)
    [draft] = test_drafts.list_pending()

    client.post(f"/staff/drafts/{draft.id}/approve", data={"text": "Hello!"}, follow_redirects=False)

    assert mock_wa.sent_messages == [{"to": REAL_SENDER, "body": "Hello!"}]
    assert test_drafts.get(draft.id).status == DraftStatus.SENT


def test_approved_simulated_reply_appears_in_the_chat_fragment(client: TestClient, test_drafts):
    client.post("/simulate/send", data={"sender": "priya", "text": "hi there"}, follow_redirects=False)
    [draft] = test_drafts.list_pending()

    before = client.get("/simulate/messages", params={"sender": "SIM-PRIYA"})
    assert "Waiting for staff approval" in before.text
    assert "Hello Priya, final answer!" not in before.text

    client.post(f"/staff/drafts/{draft.id}/approve", data={"text": "Hello Priya, final answer!"})

    after = client.get("/simulate/messages", params={"sender": "SIM-PRIYA"})
    assert "Hello Priya, final answer!" in after.text
    assert "Waiting for staff approval" not in after.text


def test_rejected_simulated_draft_shows_a_system_note_not_a_reply(client: TestClient, test_drafts, mock_wa):
    client.post("/simulate/send", data={"sender": "priya", "text": "spam-ish message"}, follow_redirects=False)
    [draft] = test_drafts.list_pending()

    client.post(f"/staff/drafts/{draft.id}/reject", follow_redirects=False)

    fragment = client.get("/simulate/messages", params={"sender": "SIM-PRIYA"}).text
    assert "Staff rejected this reply" in fragment
    assert mock_wa.sent_messages == []


# ---------------------------------------------------------------------------
# Load demo scenario
# ---------------------------------------------------------------------------


def test_load_demo_seeds_five_conversations(client: TestClient, test_drafts):
    response = client.post("/simulate/demo", follow_redirects=False)
    assert response.status_code == 303
    assert "notice=demo_loaded" in response.headers["location"]

    senders = {d.sender for d in test_drafts.list_pending()}
    assert senders == {scenario["sender"] for scenario in DEMO_SCENARIOS}
    assert len(DEMO_SCENARIOS) == 5


def test_demo_scenario_covers_hinglish_booking_emergency_angry_and_spam(client: TestClient, test_drafts):
    client.post("/simulate/demo")
    pending = {d.sender: d for d in test_drafts.list_pending()}

    price = pending["SIM-DEMO-PRICE"]
    assert "RCT" in price.patient_message

    booking = pending["SIM-DEMO-BOOKING"]
    assert "book" in booking.patient_message.lower()

    emergency = pending["SIM-DEMO-EMERGENCY"]
    assert emergency.is_urgent is True
    assert any(c in emergency.patient_message for c in "दर्द")

    angry = pending["SIM-DEMO-ANGRY"]
    assert angry.is_urgent is True

    spam = pending["SIM-DEMO-SPAM"]
    assert "http://" in spam.patient_message

    # Only the two safety-triggering scenarios are urgent.
    assert {s: d.is_urgent for s, d in pending.items()} == {
        "SIM-DEMO-PRICE": False,
        "SIM-DEMO-BOOKING": False,
        "SIM-DEMO-EMERGENCY": True,
        "SIM-DEMO-ANGRY": True,
        "SIM-DEMO-SPAM": False,
    }


def test_demo_scenarios_never_call_whatsapp_even_when_urgent(client: TestClient, mock_wa, test_drafts):
    client.post("/simulate/demo")
    assert mock_wa.sent_messages == []

    for draft in test_drafts.list_pending():
        client.post(f"/staff/drafts/{draft.id}/approve", data={"text": draft.draft_text})

    assert mock_wa.sent_messages == []
    assert all(d.status == DraftStatus.SENT for d in test_drafts.list_recent(20))


def test_reloading_demo_continues_the_same_five_conversations(client: TestClient, test_drafts):
    client.post("/simulate/demo")
    client.post("/simulate/demo")

    senders = {d.sender for d in test_drafts.list_by_sender("SIM-DEMO-PRICE")}
    assert senders == {"SIM-DEMO-PRICE"}
    assert len(test_drafts.list_by_sender("SIM-DEMO-PRICE")) == 2  # two turns, same conversation
    assert len(test_drafts.list_senders(prefix="SIM-")) == 5  # still exactly five, not ten


# ---------------------------------------------------------------------------
# The staff page also lists SIM- drafts normally (no special-casing on read)
# ---------------------------------------------------------------------------


def test_simulated_drafts_appear_on_staff_page_like_any_other(client: TestClient):
    client.post("/simulate/send", data={"sender": "priya", "text": "hi"}, follow_redirects=False)
    page = client.get("/staff").text
    assert "SIM-PRIYA" in page
    assert "1 waiting" in page
