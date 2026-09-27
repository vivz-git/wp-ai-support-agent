# AI WhatsApp Support + Lead Qualification Agent (Milestone 1)

This project contains **Portfolio Demo #2**, an AI-driven WhatsApp support agent built with FastAPI, Meta WhatsApp Cloud API, and Groq LLM.

---

## Dental Clinic Demo (SmileCare Dental)

The agent now runs as the WhatsApp front desk of **SmileCare Dental**, a fictional clinic in Pune. It reuses the existing pipeline rather than adding parallel systems:

```
WhatsApp -----\
                >- process_incoming_message() -> AgentOrchestrator (guardrails -> policy ->
/simulate ----/    extraction -> LLM + clinic_faq_lookup -> grounding)
         -> SQLite draft queue -> /staff (human approves / edits / rejects)
                                     -> WhatsApp Cloud API (real sender)
                                     -> shown in /simulate (SIM- sender, no API call)
```

### What changed

| Area | Change |
| :--- | :--- |
| Knowledge | `data/clinic_info.json` replaces the coffee `business.json`/`catalog.json`: address, timings (Mon–Sat 10am–8pm, Sunday closed), phone, three dentists, services with rough price **ranges** (consultation, cleaning, root canal, braces, whitening), FAQs. |
| Tool | `clinic_faq_lookup` (`app/tools/clinic_tool.py`) replaces `product_lookup` in the same registry. It matches English, Hindi (Devanagari) and Hinglish queries, e.g. `"RCT ka kitna lagega?"` → root canal, ₹3,500–₹8,000. |
| Emergencies | `EmergencyDetector` (`app/agent/guardrails.py`) flags pain, bleeding, swelling and dental trauma in English, Hinglish and Devanagari. `EscalationPolicy` ranks it above every other rule: the patient gets a fixed "the clinic will call you back shortly" reply in their own language, an **urgent** handoff is created, and the LLM, lead extraction and booking questions never run on that turn. Questions *about* procedures ("Is RCT painful?", "RCT mein dard hota hai kya?") are not flagged. |
| Booking details | `LeadExtractor` now collects patient name, phone, concern and preferred day/time across turns. The WhatsApp number is the default callback phone, so the bot never asks for it. Each turn the orchestrator authorizes **one** question for the next missing field. |
| Prompt | New persona and clinic facts, plus strict rules: no medical advice, diagnoses or medicine suggestions, never confirm a slot, and reply in the patient's language **and script** (English / Hindi / Hinglish). |
| Grounding | `GroundingValidator` checks every rupee amount against the service ranges, rejects unknown "Dr. …" names, and treats any medicine name or dose as unverified medical advice. |
| Human approval | The webhook no longer sends replies. Every reply is written to the `drafts` table in SQLite (`sender`, `patient_message`, `draft_text`, `status`, `is_urgent`, `timestamp`). Escalated turns (emergency, angry complaint, human request) are stored with `is_urgent=1`. |
| Staff page | `app/staff.py` serves `/staff`: pending drafts with urgent ones highlighted at the top, each with **Approve & send**, **Save edit** and **Reject**. Approve sends the (possibly edited) text through the existing `WhatsAppClient.send_text` and marks the draft `sent`. Approvals are claimed atomically, so a double click cannot send twice; a failed send puts the draft back in the queue. |
| Patient simulator | `app/simulate.py` serves `/simulate`: a WhatsApp-lookalike chat for demos with no real WhatsApp number needed — see below. |

### Running the webhook and `/staff`

`/staff` is mounted on the main FastAPI app, so one process serves both:

```bash
cp .env.example .env            # fill in the WhatsApp + Groq values
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- Webhook: `http://localhost:8000/webhook/whatsapp` (expose it with ngrok as below)
- Staff approval page: open `http://localhost:8000/staff` in a browser

Drafts are stored in `drafts.db` in the working directory; set `DRAFTS_DB_PATH` to put it elsewhere. The file is git-ignored.

> [!WARNING]
> `/staff` has no authentication: it is a local demo. It shows patients' phone numbers and can send WhatsApp messages, so never expose it on a public URL. If you tunnel the webhook with ngrok, block `/staff` at the tunnel or run a separate tunnel for the webhook path only.

### Try it without WhatsApp

```bash
curl -X POST http://localhost:8000/webhook/whatsapp -H "Content-Type: application/json" -d '{
  "object": "whatsapp_business_account",
  "entry": [{"id": "1", "changes": [{"field": "messages", "value": {"messages": [{
    "from": "919876543210", "id": "wamid.demo1", "type": "text",
    "text": {"body": "मेरे दाँत में बहुत दर्द हो रहा है"}}]}}]}]}'
```

The response is `{"status":"pending_approval", ..., "is_urgent":true}`. Refresh `/staff` to see the urgent draft at the top. Approving it calls the real WhatsApp API, which only delivers to numbers on your test allow-list.

### Demo mode: `/simulate` (no WhatsApp, no allow-list)

The recipient-allow-list limit above is exactly why `/simulate` exists: a WhatsApp-lookalike chat page for running a full demo — including approving replies — without a real phone number or Meta test recipient.

```
http://localhost:8000/simulate
```

How it works:

- Every simulated patient's id starts with `SIM-`. Type any label (e.g. `priya`) and it's normalized to `SIM-PRIYA`; typing it again (any case, with or without the prefix) reopens the same conversation.
- A message you send there runs through **the exact same pipeline** as a real WhatsApp message — `app/turn_pipeline.py`'s `process_incoming_message()` is the one function both the webhook and `/simulate` call — so it hits the same guardrails, escalation policy, lead extraction, `clinic_faq_lookup` tool, and grounding checks. The reply lands on `/staff` as an ordinary pending draft, indistinguishable from a real one except for the `SIM-` sender.
- The one deliberate difference is on approval: `/staff` recognizes a `SIM-` sender and **skips the WhatsApp API call entirely** (`app/staff.py`'s `approve_draft` checks `is_simulated_sender()`). The approved (or edited) text is simply marked delivered, and the `/simulate` chat — which polls `/simulate/messages` every 2 seconds — shows it appear as a reply. A rejected draft shows a muted "Staff rejected this reply" note instead, matching what a real patient would experience (nothing).
- **Load demo scenario** seeds five ready-made conversations in one click: a Hinglish price question (`RCT ka kitna lagega?`), a booking with name/concern/day given up front, a Hindi emergency (escalates instantly, urgent, no LLM call), an angry patient (escalates to a human), and spam. Clicking it again continues the same five conversations rather than creating new ones.

Walkthrough for a live demo:
1. Open `/simulate`, click **Load demo scenario**.
2. Open `/staff` in a second tab — five drafts are waiting, the emergency and angry ones pinned at the top under **URGENT**.
3. Approve a couple, reject one, edit one before approving.
4. Switch back to `/simulate` (or wait 2 seconds) — approved replies appear as green bubbles; the rejected one shows the muted note. Nothing ever touched WhatsApp.
5. Type a follow-up in any conversation (e.g. reply to the booking one with a phone number) to show the one-question-at-a-time flow continuing live.

> [!NOTE]
> `/simulate` has the same lack of authentication as `/staff` — it's a local demo tool. It doesn't expose anything `/staff` doesn't already (draft text, masked-free `SIM-` ids), but keep both off the public internet.

### Tests

`tests/test_dental_conversations.py` runs 24 realistic patient messages through the real webhook and orchestrator with a fake LLM:

- booking requests
- Hinglish and Hindi price questions
- FAQs
- procedure questions that mention pain
- spam and prompt injection
- an angry patient
- eight emergencies, including Devanagari ones

It asserts that emergencies escalate urgently and never reach the booking flow, and that normal messages end up as pending drafts, not immediate sends. `tests/test_staff.py` covers the queue and the staff page. `tests/test_simulate.py` covers `/simulate` itself, including the core guarantee: approving a `SIM-` draft never calls the WhatsApp client, while approving a real-number draft still does (a direct contrast test, not just an absence check). Run everything with `pytest`.

### Known limitations

- A draft is written to the conversation history when it is created. If staff edit or reject it, the model's memory of the conversation still holds the original draft.
- Staff should read every draft; the approval step is what makes this safe to demo with real patients.
- Once a conversation is escalated (for example after an emergency), it stays with the humans. Later messages get the fixed handoff reply until the conversation is reset.
- The emergency and language detectors are keyword-based. They are biased towards escalating and will miss unusual phrasings.

---

## Milestone 1 Implementation Scope

The initial milestone implements a direct, closed-loop messaging cycle:

```
WhatsApp test number -> Meta Cloud API webhook -> FastAPI backend -> Groq LLM -> WhatsApp Cloud API reply
```

- **FastAPI application** with health check and webhook endpoints.
- **GET `/webhook/whatsapp`**: Meta webhook verification handshake.
- **POST `/webhook/whatsapp`**: Safe WhatsApp payload parsing, event filtering, Groq LLM generation, and Meta Cloud API reply.
- **WhatsApp Client module**: Authenticated message delivery with safe logging.
- **Groq Provider**: Decoupled LLM abstraction (`get_agent_reply`).
- **In-memory memory**: Bounded conversation history buffer per sender.
- **Security**: Strict credential isolation, PII phone masking, and zero secret logging.

---

## Environment Variables

Copy `.env.example` to create your local `.env`:

```bash
cp .env.example .env
```

### Required Variables (Names Only)

| Variable Name | Description |
| :--- | :--- |
| `WHATSAPP_ACCESS_TOKEN` | Meta WhatsApp Cloud API access token (temporary or system user token). |
| `WHATSAPP_PHONE_NUMBER_ID` | WhatsApp Business Phone Number ID from the Meta Developer Dashboard. |
| `WHATSAPP_VERIFY_TOKEN` | A secret verification string you define and configure in the Meta App Dashboard. |
| `GROQ_API_KEY` | Your Groq Cloud API key. |

### Optional Variables (Names Only)

| Variable Name | Default | Description |
| :--- | :--- | :--- |
| `WHATSAPP_API_VERSION` | `v22.0` | Meta Graph API version. |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model identifier used for replies (Groq periodically retires older models; check console.groq.com/docs/models if replies start failing). |
| `SYSTEM_PROMPT` | *(Included in config)* | System instructions guiding the assistant's behavior. |
| `MAX_MEMORY_MESSAGES` | `10` | Maximum recent turns retained per sender in memory. |
| `DRAFTS_DB_PATH` | `drafts.db` | SQLite file for reply drafts awaiting staff approval (`/staff`). |
| `APP_PORT` | `8000` | Port for the FastAPI server. |
| `APP_HOST` | `0.0.0.0` | Host interface to bind. |

> [!WARNING]
> Never commit `.env` or place actual API tokens or secret values into `.env.example` or documentation.

---

## Local Setup & Installation

### 1. Prerequisites
- Python 3.10 or higher.
- A Meta Developer Account with WhatsApp Cloud API configured.
- A Groq Cloud API key.

### 2. Install Dependencies

From the project root:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the development server using `uvicorn`:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Or run the module directly:

```bash
python -m app.main
```

Verify the service is running:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status":"ok","service":"demo-whatsapp-support"}
```

---

## Meta Webhook Configuration

To receive events from Meta on your local machine:

### 1. Expose Local Server to the Internet
Use a tunneling tool such as `ngrok` or `cloudflared`:

```bash
ngrok http 8000
```

Copy the HTTPS forwarding URL (e.g. `https://your-domain.ngrok-free.app`).

### 2. Configure Meta Developer Dashboard
1. Go to the [Meta App Dashboard](https://developers.facebook.com/).
2. Select your WhatsApp App.
3. In the left sidebar, navigate to **WhatsApp** > **Configuration**.
4. In the **Webhook** section, click **Edit**:
   - **Callback URL**: `https://your-domain.ngrok-free.app/webhook/whatsapp`
   - **Verify token**: The value you set in `WHATSAPP_VERIFY_TOKEN`.
5. Click **Verify and Save**. Meta will send a GET request to `/webhook/whatsapp` to validate the handshake.
6. Under **Webhook fields**, click **Manage** and subscribe to:
   - `messages`

---

## Testing Milestone 1

### Automated Test Suite

Run the full test suite with `pytest`:

```bash
pytest -v
```

The Milestone 1 tests cover the items below; the suite has since grown to cover the full agent and the dental demo (see above):
- Webhook verification success (`GET` with correct token & mode).
- Webhook verification failure (`GET` with invalid token, wrong mode, or missing params).
- Valid incoming WhatsApp text message payload (`POST` handling, LLM invocation, draft queued for staff approval).
- Malformed payloads (missing fields, invalid JSON returning 400).
- Unsupported events (ignoring status delivery receipts and media messages with 200).
- Basic LLM abstraction with mock Groq client.
- WhatsApp Cloud API client with mock HTTP transport.

### Manual Verification via Curl

#### Test Webhook Handshake (GET):
```bash
curl "http://localhost:8000/webhook/whatsapp?hub.mode=subscribe&hub.challenge=12345&hub.verify_token=YOUR_CONFIGURED_VERIFY_TOKEN"
```
Expected response: `12345` with HTTP 200.

#### Test Incoming Message (POST):
```bash
curl -X POST http://localhost:8000/webhook/whatsapp \
  -H "Content-Type: application/json" \
  -d '{
    "object": "whatsapp_business_account",
    "entry": [{
      "id": "12345",
      "changes": [{
        "value": {
          "messaging_product": "whatsapp",
          "messages": [{
            "from": "919876543210",
            "id": "wamid.test12345",
            "type": "text",
            "text": {"body": "Hello"}
          }]
        },
        "field": "messages"
      }]
    }]
  }'
```
Expected response: `{"status":"pending_approval","message_id":"wamid.test12345","draft_id":1,"is_urgent":false}` with HTTP 200. The reply is waiting on `/staff` for approval.
