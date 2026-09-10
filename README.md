# AI WhatsApp Support + Lead Qualification Agent (Milestone 1)

This project contains **Portfolio Demo #2**, an AI-driven WhatsApp support agent built with FastAPI, Meta WhatsApp Cloud API, and Groq LLM.

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
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model identifier used for replies. |
| `SYSTEM_PROMPT` | *(Included in config)* | System instructions guiding the assistant's behavior. |
| `MAX_MEMORY_MESSAGES` | `10` | Maximum recent turns retained per sender in memory. |
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

All 22 tests cover:
- Webhook verification success (`GET` with correct token & mode).
- Webhook verification failure (`GET` with invalid token, wrong mode, or missing params).
- Valid incoming WhatsApp text message payload (`POST` handling, memory storage, LLM invocation, WhatsApp client dispatch).
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
Expected response: `{"status":"ok","message_id":"wamid.test12345"}` with HTTP 200.
