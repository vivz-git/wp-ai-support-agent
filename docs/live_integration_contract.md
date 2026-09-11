# Live Integration Contract (Milestone 2, Slice 14 → Slice 15)

This document fixes the contract that Slice 15 must implement when the agent
core (`app/agent/*`) is wired into the WhatsApp webhook. Nothing here is
implemented in `app/main.py` yet: Slice 14 is a policy review and hardening
slice, and the Milestone 1 request path is untouched.

Verified by: `tests/test_orchestrator.py` (`test_slice14_*`),
`tests/test_escalation.py` (`test_slice14_*`), `tests/test_agent_evaluation.py`
(scenario 24 and the resolved policy edge case), `tests/test_webhook.py`
(Milestone 1 duplicate-delivery test).

## 1. Layering and the WAMID boundary

```
Meta webhook (POST /webhook/whatsapp)
    │  raw JSON
    ▼
Webhook adapter (app/main.py + app/whatsapp/models.py)
    │  parse/validate, event-type filter, sender, text, WAMID
    ▼
WAMID duplicate check  (InMemoryConversationMemory.has_processed / mark_processed)
    │  first delivery only
    ▼
AgentOrchestrator.handle_turn(sender_id, text, message_id=WAMID)
    │  state load → guardrails → policy → extraction → LLM/tools → grounding → handoff → save
    ▼
AgentTurnResult (reply_text, state_snapshot, tool_calls, diagnostics)
    │  reply_text only
    ▼
WhatsApp client (app/whatsapp/client.py)  →  HTTP 200 acknowledgement to Meta
```

**Idempotency is an adapter responsibility.** The Meta message ID
(`wamid.…`) is a transport-level retry token. The agent core:

- does not know what a WAMID is (`app/agent/*` never imports `app.memory`
  or `app.whatsapp`; the words `wamid`, `has_processed`, `mark_processed`
  do not appear in the agent package — pinned by
  `test_slice14_orchestrator_has_no_wamid_or_duplicate_ledger_logic`);
- accepts `message_id` as an *optional, opaque* string and only echoes it
  into `TurnDiagnostics.message_id` for correlation; it is never stored in
  `ConversationState`;
- **does not deduplicate**: calling `handle_turn` twice with the same
  `message_id` processes two turns
  (`test_slice14_orchestrator_does_not_depend_on_message_id`, evaluation
  scenario 24). The check must therefore run *before* `handle_turn`.

The existing Milestone 1 ledger (`InMemoryConversationMemory`, bounded FIFO
of 500 WAMIDs) already implements the check on the current webhook path and
is the component Slice 15 keeps in front of the orchestrator
(`test_slice14_wamid_duplicate_check_belongs_to_the_adapter_in_front_of_handle_turn`,
`test_slice14_milestone_one_webhook_keeps_the_ledger_in_front_of_the_model`).
No second ledger may be added inside `AgentOrchestrator` or
`ConversationStore`.

Known limitation (unchanged from Milestone 1, out of scope here): the ledger
is per-process and in-memory. A multi-instance deployment or a restart loses
it; that is a deployment concern for a later slice, not a reason to move the
check into the agent core.

## 2. The ten steps Slice 15 implements

| # | Step | Owner | Contract |
|---|------|-------|----------|
| 1 | Inbound Meta event arrives | FastAPI route | `POST /webhook/whatsapp`; non-JSON → 400 (unchanged). |
| 2 | Parse / validate webhook | `parse_incoming_webhook` | Malformed structure → 400. Status callbacks, media, reactions and other non-text events → 200 `{"status": "ignored"}` **without** touching the ledger or the agent. |
| 3 | Extract WAMID | adapter | `msg.message_id`. Also extract `msg.sender` and `msg.text`. |
| 4 | Duplicate check | adapter + `InMemoryConversationMemory` | `if memory.has_processed(wamid): return 200 {"status": "duplicate_ignored"}` — before any agent work. Then `memory.mark_processed(wamid)` **before** calling the agent (Milestone 1 semantics: a retry that lands mid-turn must not start a second turn). |
| 5 | Create / load `ConversationState` | `AgentOrchestrator` via `ConversationStore` | The adapter does **not** load state itself; it passes `sender_id`. A process-wide `ConversationStore` singleton replaces the per-sender history in `InMemoryConversationMemory` for the agent path (the memory object keeps only its ledger role). |
| 6 | Call `AgentOrchestrator.handle_turn(sender, text, message_id=wamid)` | adapter | Exactly once per accepted event. The adapter must guard blank text: `handle_turn` raises `ValueError` for empty/whitespace text *before* touching state — treat that as "ignored", 200. Any other exception from `handle_turn` is a bug (it contains guardrail, policy, extraction, LLM, tool, grounding and sink failures itself), but the route must still catch it, log the exception **type only**, and return 200. |
| 7 | Receive `AgentTurnResult` | adapter | `reply_text` is always non-empty and already safe: it is a grounded model reply, a deterministic policy reply (`SAFE_REFUSAL_REPLY`, `HUMAN_HANDOFF_REPLY`, `CLARIFICATION_REPLY`, `UNVERIFIED_RECOVERY_REPLY`) or `SAFE_FALLBACK_REPLY`. The adapter must not rewrite, filter, or append to it, and must not read `state_snapshot` or `tool_calls` to decide anything. |
| 8 | Send reply through WhatsApp client | adapter + `WhatsAppClient.send_text` | Send `reply_text` to `sender`. Meta error 131030 (test recipient not allowed) stays informational; any other `WhatsAppClientError` is logged by type/code and does not trigger a retry of the agent turn. |
| 9 | Mark / retain processing state | adapter | Conversation state was already persisted inside `handle_turn` (step 6) *before* the reply is sent, so a send failure never loses the turn. The WAMID was marked in step 4. Nothing else to persist. Diagnostics (`result.diagnostics`) may be logged as structured fields — see §4. |
| 10 | Acknowledge webhook | adapter | 200 for every accepted, duplicate, ignored, agent-failed or send-failed event; 400 only for unparsable or malformed payloads (steps 1–2). Never 5xx for a downstream failure: Meta would redeliver and the ledger would then correctly drop the retry, but the customer gains nothing. |

Failure semantics the adapter inherits from the core (all pinned by
`test_slice14_failure_boundary_is_contained_deterministic_and_leaks_nothing`):

| Boundary | Behaviour of `handle_turn` | Customer sees |
|----------|----------------------------|---------------|
| Guardrail detector raises | fail closed, no model call, no handoff | `SAFE_FALLBACK_REPLY` |
| Escalation policy raises | fail closed: `escalate`/`policy_error`, escalation handoff | `HUMAN_HANDOFF_REPLY` |
| Grounding validator raises | reply treated as ungrounded, no rewrite | `UNVERIFIED_RECOVERY_REPLY` |
| LLM provider raises | safe fallback, lead data kept | `SAFE_FALLBACK_REPLY` |
| Tool handler raises | `unavailable` fed back, tools closed for the turn | model's grounded reply |
| Lead extractor raises | lead untouched, turn continues | model's grounded reply |
| Handoff sink raises | `handoff_outcome=unavailable`, state still escalated | `HUMAN_HANDOFF_REPLY` |

In every case the exception **class name only** reaches diagnostics and
INFO-level logs; the message (which for provider errors can carry headers or
response bodies) is logged at DEBUG only. Production must run at INFO.

## 3. Policy precedence the adapter relies on

The adapter never inspects the decision; it only sends `reply_text`. For
reference, the reviewed order (Slice 14, `app/agent/escalation.py`):

```
 1 grounding violation            suppress
 2 already escalated (sticky)     escalate
 3 explicit human request         escalate
 4 high anger + complaint         escalate
 5 repeated unresolved ask        escalate
 6 repeated/aggressive injection  escalate
 7 injection + secrets/internal   refuse      ← moved above 9 in Slice 14
 8 basic injection attempt        refuse      ← moved above 9 in Slice 14
 9 qualified, complete lead       handoff_ready  (was 7)
10 high anger, no complaint ctx   clarify
11 single repetition              clarify
12 nothing fired                  continue
```

Consequence for live traffic: a qualified lead who sends an injection gets
the deterministic refusal with no model call and no new handoff, exactly like
any other sender; the qualified-lead ticket already raised stays as it is,
and the next ordinary message is `handoff_ready` again.

## 4. What the adapter may log

Allowed: `result.diagnostics` (all fields), the masked sender
(`mask_phone_number`), the WAMID, HTTP status codes, exception class names.
`TurnDiagnostics` contains no free text by construction — every string
value is a code, a class name, a masked sender, a message ID or a sink ID
(`test_slice14_diagnostics_never_carry_secrets_phone_numbers_prompt_or_free_text`).

Not allowed in logs: `reply_text`, `state_snapshot` (history, lead profile),
`result.tool_calls` (model-authored arguments; the model can echo customer
text, including phone numbers, into a tool query), the prompt bundle,
raw Meta payloads, exception messages, and anything from `Settings`.

## 5. Out of scope for Slice 15's first cut

- Persisting `ConversationStore` or the WAMID ledger beyond process memory.
- Signature verification of Meta payloads (`X-Hub-Signature-256`) — worth
  adding at the adapter, never inside the agent.
- Delivering handoffs anywhere but `InMemoryHandoffSink`.
