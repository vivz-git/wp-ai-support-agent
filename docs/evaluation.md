# Agent Core Evaluation (Milestone 2, Slice 13)

Suite: `tests/test_agent_evaluation.py` (35 tests, all passing; full suite 690/690 after
Slice 14 — 34 tests / 655 total at the end of Slice 13).

Every scenario drives the real `AgentOrchestrator` end to end:

```
customer message → incoming guardrails → escalation policy → lead extraction
→ state update → qualification → PromptBuilder → LLM → tool calls → tool results
→ grounding validation → outgoing policy → handoff → final response → persistence
```

Collaborators are real except the two models: a `ScriptedLLM` plays the agent
model, a second one plays the extractor model. The real `product_lookup` tool,
`KnowledgeBase`, detectors, `EscalationPolicy`, `GroundingValidator`,
`LeadExtractor`, `ConversationStore` and `InMemoryHandoffSink` are used
unchanged. No network, no Groq, no Meta/WhatsApp, no sleeps, no randomness.
No production code was modified.

## Scenario matrix

| # | Scenario | Observed behaviour | Result |
|---|----------|--------------------|--------|
| 1 | Normal consumer ("light roast for pour-over") | tool round → grounded reply, `continue`, only stated fields extracted, `collecting`, no handoff | pass |
| 2 | Product lookup ("How much is Yirgacheffe Light?") | `product_lookup` executed, result (₹780) fed back in transcript + system prompt, reply accepted | pass |
| 3 | Product + lead extraction | name/city/brew method merged with provenance turn 1; `name=…`/`city=…` visible in every prompt of the turn | pass |
| 4 | Wholesale lead | track/type/volume/city set; `collecting` (missing name, business name, timeline) despite model claiming "approved" | pass |
| 5 | Multi-turn wholesale qualification | values accumulate, null extraction erases nothing, `qualified` on the completing turn → `handoff_ready` (outgoing) → `qualified_lead` handoff | pass |
| 6 | Human request | `escalate/human_requested`, no model/extractor call, deterministic reply, `escalation` handoff (HIGH), state `pending` | pass |
| 7a | High anger, fresh conversation | anger 0.85 but no complaint context → `clarify`, no handoff | pass |
| 7b | High anger after an unresolved repeat | repetition context → `escalate/high_anger_complaint`, URGENT handoff | pass |
| 8 | First repetition | `clarify`, no model call, `repeated_question_count == 1`, no handoff | pass |
| 9 | Repeated unresolved question | clarify → escalate (created) → `already_escalated` (deduplicated, same ID, `submission_count == 2`) | pass |
| 10 | Prompt injection | `refuse`, no model call, `SAFE_REFUSAL_REPLY`, flags recorded, no handoff | pass |
| 11 | Secret request | `refuse/injection_secrets_requested`; second attempt → `escalate/injection_repeated` + handoff | pass |
| 12 | Hallucinated price (₹500 vs ₹780) | suppressed, ONE tool-free corrective call, second bad draft rejected, `UNVERIFIED_RECOVERY_REPLY`, `grounding_violations == 2` | pass |
| 12b | Corrective rewrite drops the claim | accepted as `model_corrected`; rewrite prompt carries the rejected draft + reason codes only | pass |
| 13 | Grounded price claim | accepted from catalog facts without a tool call | pass |
| 14/15 | Wrong origin / wrong tasting notes | suppressed, never stored in history, recovery reply | pass |
| 16 | Tool failure (catalog unavailable) | real tool returns `unavailable`, tools closed for the turn, invented price suppressed, corrected safe reply | pass |
| 17 | LLM failure | `SAFE_FALLBACK_REPLY`, extracted lead kept, state persisted, no handoff | pass |
| 18 | Extraction failure (provider error / extractor crash) | model reply as normal, existing lead preserved, no fallback | pass |
| 19 | Grounding validator exception | fail closed: no rewrite attempted, recovery reply, one model call | pass |
| 20 | Handoff sink failure | deterministic handoff reply, state `pending`, `handoff_outcome == unavailable`, error type only | pass |
| 21 | Qualified lead handoff | disallowed `qualification`/`escalated` keys dropped by extractor; `qualified_lead` LOW handoff; state stays `qualified` | pass |
| 22 | Handoff deduplication | same handoff ID, one record, `submission_count == 2` | pass |
| 23 | Sender isolation | lead, history, escalation and tickets never cross senders | pass |
| 24 | Duplicate message ID | orchestrator does **not** dedupe (see observations); the existing `InMemoryConversationMemory` ledger gates duplicates | pass (documented) |
| 25 | Tool + extraction interaction | exactly one extraction across two tool rounds; lead visible in every prompt | pass |
| 26 | Tool round limit | `AGENT_MAX_TOOL_ROUNDS` rounds then one call with no `tools`/`tool_choice` | pass |
| 27 | Mixed anger + "asked three times" + injection | `refuse/injection_internal_data_requested` wins (anger 0.3 < 0.6, no prior repeat); flags accumulate; no handoff | pass |
| 28 | > 4096-char message with injection and `!!!` runs | bounded to 4096 in history, refused, no model call | pass |
| 29 | Greeting → product → preference → business → human | state and provenance accumulate; escalation carries lead + 10-entry transcript | pass |
| 30 | 8-turn realistic conversation (with correction and mild frustration) | zero grounding violations, city correction reflected in handoff, exactly one handoff, no network | pass |
| — | Policy edge case: qualified lead + injection | `refuse`, no model call, no new handoff (Slice 14; see below) | pass |
| — | Diagnostics sanitation across refusal/escalation/grounding turns | no customer text, raw number, secrets or model output in diagnostics | pass |

## Policy edge case: qualified lead + injection (resolved in Slice 14)

**Before Slice 14 (observed in Slice 13, pinned then):** for a lead whose
profile was already `qualified` and complete, an injection message produced
`handoff_ready` (priority 7) rather than `refuse` (priority 8/9). The turn
proceeded to the model with the injected text delimited as untrusted
customer content; the deterministic refusal was not used. Only a second
attempt (crossing `INJECTION_ESCALATION_HITS`) escalated.

**After Slice 14:** the two injection refusals were moved above the
qualified-lead rule — priorities 7/8 for `injection_secrets_requested` /
`injection_internal_data_requested` / `injection_attempt`, 9 for
`lead_qualified` / `handoff_ready`. Ranks 1–6 and 10–12 are unchanged. The
principle: *security restrictions must not be bypassed by qualification
state.* Observed end to end
(`test_policy_edge_case_qualified_lead_plus_injection_is_refused_since_slice_14`,
`test_policy_edge_case_qualified_lead_recovers_to_handoff_ready_after_a_single_refusal`):

| Qualified lead sends… | Decision | Reply | Model call | Handoff |
|---|---|---|---|---|
| ordinary message | `handoff_ready` (9) | grounded model reply | yes | `qualified_lead` (deduplicated) |
| human request | `escalate` (3) | `HUMAN_HANDOFF_REPLY` | no | `escalation` |
| basic injection | `refuse` (8) `["injection_attempt", "lead_qualified"]` | `SAFE_REFUSAL_REPLY` | no | none |
| secret / system-prompt request | `refuse` (7) | `SAFE_REFUSAL_REPLY` | no | none |
| repeated or multi-pattern injection | `escalate` (6) `injection_repeated` | `HUMAN_HANDOFF_REPLY` | no | `escalation` |
| ordinary message after one refusal | `handoff_ready` (9) | grounded model reply | yes | deduplicated |

The `lead_qualified` code is still appended to `reason_codes` on a refused
turn so the outcome stays explainable; qualification is neither demoted nor
transitioned by a refusal (state stays `qualified`, `escalation.status`
stays `none`). An incomplete lead plus injection is refused without any
`lead_qualified`/`handoff_ready` code. The exhaustive policy-level check
(`tests/test_escalation.py::test_slice14_security_sensitive_decision_cannot_become_handoff_ready`)
covers every qualification state × injection variant × prior-hit count.

Tests deliberately updated for the swap (the only "old" assertions touched):
`test_escalation.py::test_escalation_priority_is_deterministic` (7/8/9 →
9/7/8), `test_orchestrator.py::test_qualified_complete_lead_creates_qualified_lead_handoff`
(`policy_priority` 7 → 9), and the Slice 13 pin above, rewritten to pin the
reviewed behaviour.

## Slice 14 hardening review (no production defects found)

Reviewed and pinned by `test_slice14_*` in `tests/test_escalation.py` (11
tests) and `tests/test_orchestrator.py` (23 tests):

- **Determinism.** Identical `(state, signals)` → identical
  `EscalationDecision`, across fresh policy instances and structurally-equal
  fresh inputs; `reason_codes` is always in priority order with no
  duplicates; the winning priority equals the rank of the first code. The
  policy module imports no LLM, clock, randomness or network.
- **Python control.** Neither the agent model nor the extractor model can set
  qualification, escalation or a handoff: `LeadDelta` and `GuardrailSignals`
  have no such fields, the orchestrator constructs exactly one
  `EscalationDecision` itself (the fail-closed `policy_error` constant) and
  never calls `mark_handoff_ready`.
- **WAMID boundary.** The agent package contains no ledger, no
  `wamid`/`has_processed`/`mark_processed`, no `app.memory` or
  `app.whatsapp` import; `message_id` is optional and only echoed into
  diagnostics; `app/main.py` still checks the ledger before its model call.
  Contract for Slice 15: `docs/live_integration_contract.md`.
- **Failure boundaries.** Guardrail, policy, grounding, LLM, tool,
  extraction and sink failures (each raising an error whose message carries a
  fake `Authorization: Bearer …` header) are contained, produce identical
  replies/diagnostics/state on two identical runs, persist state, and leak
  nothing into the reply, diagnostics, state, tool records or INFO-level
  logs (no stack traces at INFO).
- **Diagnostics.** Every string leaf of `TurnDiagnostics` matches
  `[A-Za-z0-9_.*:-]+` — no free text, so no customer text, model output,
  system prompt (checked line by line against the real built prompt), API
  keys, headers or unmasked numbers can appear. `AgentTurnResult.tool_calls`
  is model-authored and *can* echo customer text (a phone number copied into
  a tool query); it is not part of diagnostics and must not be logged as such.

## Architectural observations

1. **Idempotency is not the orchestrator's job — by contract.** `handle_turn`
   records `message_id` in diagnostics only. Calling it twice with the same
   ID processes two turns. Message-level deduplication lives in
   `InMemoryConversationMemory.has_processed/mark_processed` on the
   Milestone 1 webhook path; Slice 14 fixed this as the architectural
   boundary (`docs/live_integration_contract.md`) and the slice that wires
   the agent core into the webhook must keep that ledger in front of
   `handle_turn`.
2. **Anger needs complaint context to escalate.** "This is ridiculous!!! I've
   asked three times! Fix this now!" scores 0.85 but "asked three times" is
   not in the complaint lexicon, so a fresh conversation gets `clarify`. With
   a prior repeat (or complaint wording such as "third time") it escalates.
   This is by design (rule 4 vs 10), but the lexicon gap is worth noting.
3. **Diagnostics report the decisive (last) policy decision.** On a
   `handoff_ready` turn the incoming decision is non-blocking, so
   `escalation_stage` reads `outgoing` and the reason codes reflect the
   outgoing (grounding-only) evaluation. Tests that need the incoming
   decision must observe the policy directly (`RecordingPolicy`).
4. **`handoff_ready` is recorded, not transitioned.** After a qualified-lead
   handoff the state stays `qualified` and every later turn re-submits
   (deduplicated by the sink). The explicit `mark_handoff_ready` consent step
   remains owned by the qualification-policy slice.
5. **Short questions never count as repeats.** The repetition detector needs
   two content tokens; "Where is my order?" reduces to one and is treated as
   too short. Realistic repeats ("check the status of my order number 4521")
   are detected.

## Not covered here

- Live provider behaviour (Groq tool-call formatting, latency, retries).
- Webhook wiring, WhatsApp delivery, Meta payload handling (`app/main.py`
  and `app/whatsapp/*` are untouched and out of scope for this slice).
- Extraction *quality*: the extractor model is scripted, so these tests prove
  the merge/qualification path, not what a real model would extract.
