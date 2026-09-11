"""Tests for the deterministic agent orchestrator (app/agent/orchestrator.py).

All dependencies are fakes or scripted: a scripted ``LLMProvider`` that
returns canned ``LLMResponse`` objects (or raises), a real ``ToolRegistry``
holding either the real ``product_lookup`` tool or small scripted tools, the
real (fictional) knowledge base, and an in-memory ``ConversationStore``.
No Groq or WhatsApp network calls are made anywhere in this module.

Slice 8 tests (bottom of the file) add a real ``LeadExtractor`` over its own
scripted provider and verify it enriches state before the prompt is built,
never runs inside the tool loop, never decides qualification/escalation,
and never turns the turn into a safe fallback when it fails.
"""

import asyncio
import inspect
import json
import os
import re
import socket
from typing import Any, Dict, List, Optional, Union
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, ConfigDict

from app.agent.extraction import ALLOWED_EXTRACTION_FIELDS, LeadExtractor
from app.agent.lead import LeadDelta, LeadTrack, QualificationState, evaluate_qualification
from app.agent.orchestrator import (
    AGENT_MAX_TOOL_ROUNDS,
    SAFE_FALLBACK_REPLY,
    AgentOrchestrator,
    AgentTurnResult,
    ToolCallRecord,
    TurnDiagnostics,
)
from app.agent.prompts import CUSTOMER_MESSAGE_OPEN
from app.agent.state import MAX_CHAT_HISTORY, ConversationState, EscalationStatus
from app.agent.store import ConversationStore
from app.knowledge import get_knowledge_base
from app.llm.base import ChatMessage, LLMProviderError, LLMResponse, TokenUsage, ToolCall
from app.llm.groq_provider import GroqProvider
from app.tools import build_default_registry
from app.tools.registry import ToolRegistry, ToolSpec

SENDER = "919876543210"
OTHER_SENDER = "918765432109"

# conftest.py sets these mock credential values before app.config loads.
_SECRET_ENV_VARS = ("WHATSAPP_ACCESS_TOKEN", "WHATSAPP_VERIFY_TOKEN", "GROQ_API_KEY")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class ScriptedLLM:
    """``LLMProvider`` that replays a script of responses/exceptions and records every call."""

    def __init__(self, script: List[Union[LLMResponse, Exception]]):
        self._script = list(script)
        self.calls: List[Dict[str, Any]] = []

    async def complete(self, messages: List[ChatMessage], **kwargs) -> LLMResponse:
        self.calls.append({"messages": list(messages), "kwargs": dict(kwargs)})
        if not self._script:
            raise AssertionError("ScriptedLLM received more calls than scripted")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def get_agent_reply(self, messages: List[ChatMessage]) -> str:
        raise AssertionError("get_agent_reply must not be used by the orchestrator")

    @property
    def exhausted(self) -> bool:
        return not self._script


def text(content: str, usage: Optional[TokenUsage] = None) -> LLMResponse:
    return LLMResponse(content=content, finish_reason="stop", usage=usage)


def tool_calls(*calls: ToolCall, content: Optional[str] = None) -> LLMResponse:
    return LLMResponse(content=content, tool_calls=list(calls), finish_reason="tool_calls")


def call(call_id: str, name: str = "product_lookup", raw_arguments: str = '{"query": "ethiopia"}') -> ToolCall:
    return ToolCall.from_raw_arguments(id=call_id, name=name, raw_arguments=raw_arguments)


class _AnyInput(BaseModel):
    model_config = ConfigDict(extra="allow")


def scripted_tool(name: str, results: List[Union[dict, Exception]]) -> ToolSpec:
    """A registry tool whose handler replays ``results`` (a dict or an exception per call)."""
    queue = list(results)

    def handler(arguments: dict) -> dict:
        if not queue:
            raise AssertionError(f"tool {name} called more times than scripted")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return ToolSpec(name=name, description=f"scripted {name}", input_model=_AnyInput, handler=handler)


def registry_with(*specs: ToolSpec) -> ToolRegistry:
    registry = ToolRegistry()
    for spec in specs:
        registry.register(spec)
    return registry


OK_RESULT = {"status": "ok", "result_count": 1, "results": [{"sku": "X", "name": "Scripted Bean"}]}
NO_MATCH_RESULT = {"status": "no_match", "result_count": 0, "results": [], "suggestions": []}
INVALID_INPUT_RESULT = {
    "status": "invalid_input",
    "result_count": 0,
    "results": [],
    "suggestions": [],
    "error": {"code": "validation_failed", "message": "bad args", "fields": ["query"]},
}
UNAVAILABLE_RESULT = {
    "status": "unavailable",
    "result_count": 0,
    "error": {"code": "catalog_unavailable", "message": "The product catalog is temporarily unavailable."},
}


@pytest.fixture(scope="module")
def knowledge():
    return get_knowledge_base()


@pytest.fixture
def store() -> ConversationStore:
    return ConversationStore()


def make(llm, knowledge, store, tools: Optional[ToolRegistry] = None, **kwargs) -> AgentOrchestrator:
    return AgentOrchestrator(
        llm=llm,
        knowledge=knowledge,
        tools=tools if tools is not None else build_default_registry(),
        store=store,
        **kwargs,
    )


def run(orchestrator: AgentOrchestrator, message: str, sender: str = SENDER, **kwargs) -> AgentTurnResult:
    return asyncio.run(orchestrator.handle_turn(sender, message, **kwargs))


def _tool_messages(messages: List[ChatMessage]) -> List[ChatMessage]:
    return [m for m in messages if m.role == "tool"]


# ---------------------------------------------------------------------------
# 1. Simple text response
# ---------------------------------------------------------------------------


def test_simple_text_response(knowledge, store):
    llm = ScriptedLLM([text("Hi there! How can I help?")])
    result = run(make(llm, knowledge, store), "hello")

    assert result.reply_text == "Hi there! How can I help?"
    assert len(llm.calls) == 1
    assert result.tool_calls == []
    assert result.diagnostics.llm_calls == 1
    assert result.diagnostics.tool_rounds == 0
    assert result.diagnostics.fallback_used is False
    assert result.diagnostics.loop_exit == "text_reply"


def test_first_call_offers_registry_tools_to_the_model(knowledge, store):
    llm = ScriptedLLM([text("ok")])
    run(make(llm, knowledge, store), "hello")

    tools = llm.calls[0]["kwargs"]["tools"]
    assert [t.name for t in tools] == ["product_lookup"]
    assert tools[0].to_openai_tool() == build_default_registry().list_specs()[0]


# ---------------------------------------------------------------------------
# 2 & 3. One successful tool call -> tool result -> final text
# ---------------------------------------------------------------------------


def test_one_successful_tool_call_with_real_product_lookup(knowledge, store):
    llm = ScriptedLLM([tool_calls(call("call_1")), text("Yes! Our Ethiopia roast is in stock.")])
    result = run(make(llm, knowledge, store), "Do you have Ethiopia?")

    assert result.reply_text == "Yes! Our Ethiopia roast is in stock."
    assert len(result.tool_calls) == 1
    record = result.tool_calls[0]
    assert record.tool_name == "product_lookup"
    assert record.disposition == "executed"
    assert record.status == "ok"
    assert record.ok is True
    assert record.call_id == "call_1"
    assert result.diagnostics.llm_calls == 2
    assert result.diagnostics.tool_rounds == 1
    assert result.diagnostics.tool_calls_executed == 1


def test_tool_result_is_fed_back_before_final_text(knowledge, store):
    llm = ScriptedLLM([tool_calls(call("call_1")), text("done")])
    run(make(llm, knowledge, store), "Do you have Ethiopia?")

    second_call = llm.calls[1]["messages"]
    # ... system, history, <current user>, assistant(tool_calls), tool(result)
    assert second_call[-3].role == "user"
    assistant_msg = second_call[-2]
    assert assistant_msg.role == "assistant"
    assert assistant_msg.tool_calls == [
        {"id": "call_1", "type": "function", "function": {"name": "product_lookup", "arguments": '{"query": "ethiopia"}'}}
    ]
    tool_msg = second_call[-1]
    assert tool_msg.role == "tool"
    assert tool_msg.tool_call_id == "call_1"  # ID preserved
    payload = json.loads(tool_msg.content)
    assert payload["status"] == "ok"
    assert payload["result_count"] >= 1
    # The system prompt for the second call also carries the current-turn tool results.
    assert "Tool results for this turn" in second_call[0].content
    assert "Tool results for this turn" not in llm.calls[0]["messages"][0].content


# ---------------------------------------------------------------------------
# 4. Multiple sequential tool calls
# ---------------------------------------------------------------------------


def test_multiple_tool_calls_in_one_response_execute_in_order(knowledge, store):
    tool = scripted_tool("lookup", [OK_RESULT, NO_MATCH_RESULT])
    llm = ScriptedLLM(
        [
            tool_calls(call("call_a", "lookup", '{"q": "first"}'), call("call_b", "lookup", '{"q": "second"}')),
            text("Here you go"),
        ]
    )
    result = run(make(llm, knowledge, store, tools=registry_with(tool)), "two things")

    assert [r.call_id for r in result.tool_calls] == ["call_a", "call_b"]
    assert [r.status for r in result.tool_calls] == ["ok", "no_match"]
    assert [r.arguments for r in result.tool_calls] == [{"q": "first"}, {"q": "second"}]
    tool_msgs = _tool_messages(llm.calls[1]["messages"])
    assert [m.tool_call_id for m in tool_msgs] == ["call_a", "call_b"]


def test_multiple_tool_rounds_execute_sequentially(knowledge, store):
    tool = scripted_tool("lookup", [OK_RESULT, OK_RESULT])
    llm = ScriptedLLM(
        [
            tool_calls(call("call_1", "lookup", '{"q": "a"}')),
            tool_calls(call("call_2", "lookup", '{"q": "b"}')),
            text("final"),
        ]
    )
    result = run(make(llm, knowledge, store, tools=registry_with(tool)), "hi")

    assert result.reply_text == "final"
    assert [r.round for r in result.tool_calls] == [1, 2]
    assert result.diagnostics.tool_rounds == 2
    assert result.diagnostics.llm_calls == 3
    # The third call's transcript replays both rounds in order.
    roles = [m.role for m in llm.calls[2]["messages"][-4:]]
    assert roles == ["assistant", "tool", "assistant", "tool"]


# ---------------------------------------------------------------------------
# 5. Malformed tool arguments
# ---------------------------------------------------------------------------


def test_malformed_tool_arguments_are_rejected_without_execution(knowledge, store):
    tool = scripted_tool("lookup", [])  # would raise if executed
    llm = ScriptedLLM([tool_calls(call("call_1", "lookup", "{not json")), text("Could you rephrase?")])
    result = run(make(llm, knowledge, store, tools=registry_with(tool)), "hi")

    assert result.reply_text == "Could you rephrase?"
    record = result.tool_calls[0]
    assert record.disposition == "rejected_malformed_arguments"
    assert record.status == "invalid_input"
    assert record.error_code == "malformed_arguments_json"
    assert record.ok is False
    assert record.arguments == {}
    fed_back = json.loads(_tool_messages(llm.calls[1]["messages"])[0].content)
    assert fed_back["status"] == "invalid_input"
    assert fed_back["error"]["code"] == "malformed_arguments_json"
    assert result.diagnostics.tool_calls_rejected == 1


# ---------------------------------------------------------------------------
# 6. Unknown tool
# ---------------------------------------------------------------------------


def test_unknown_tool_is_rejected_safely(knowledge, store):
    llm = ScriptedLLM([tool_calls(call("call_1", "delete_everything", "{}")), text("I can't do that.")])
    result = run(make(llm, knowledge, store), "hi")

    assert result.reply_text == "I can't do that."
    record = result.tool_calls[0]
    assert record.disposition == "rejected_unknown_tool"
    assert record.status == "invalid_input"
    assert record.error_code == "unknown_tool"
    fed_back = json.loads(_tool_messages(llm.calls[1]["messages"])[0].content)
    assert fed_back["error"]["code"] == "unknown_tool"
    # Still recorded in state (bounded name) so failure counters stay honest.
    assert result.state_snapshot.tool_history[-1].tool_name == "delete_everything"
    assert result.state_snapshot.flags.tool_failures_this_turn == 1


def test_overlong_unknown_tool_name_does_not_break_state(knowledge, store):
    llm = ScriptedLLM([tool_calls(call("call_1", "x" * 200, "{}")), text("ok")])
    result = run(make(llm, knowledge, store), "hi")

    assert result.reply_text == "ok"
    assert len(result.state_snapshot.tool_history[-1].tool_name) == 64


# ---------------------------------------------------------------------------
# 7 & 8. invalid_input -> one corrective retry; second failure stops retry
# ---------------------------------------------------------------------------


def test_invalid_input_allows_one_corrective_retry(knowledge, store):
    tool = scripted_tool("lookup", [INVALID_INPUT_RESULT, OK_RESULT])
    llm = ScriptedLLM(
        [
            tool_calls(call("call_1", "lookup", '{"q": ""}')),
            tool_calls(call("call_2", "lookup", '{"q": "fixed"}')),
            text("Found it!"),
        ]
    )
    # Headroom beyond the retry so the round limit is not what ends tool use.
    result = run(make(llm, knowledge, store, tools=registry_with(tool), max_tool_rounds=5), "hi")

    assert result.reply_text == "Found it!"
    assert [r.status for r in result.tool_calls] == ["invalid_input", "ok"]
    assert [r.disposition for r in result.tool_calls] == ["executed", "executed"]
    assert result.diagnostics.loop_exit == "text_reply"
    assert result.diagnostics.final_call_tools_omitted is False
    assert "tools" in llm.calls[2]["kwargs"]  # the retry was still a tool-enabled round


def test_second_invalid_input_stops_further_tool_calls(knowledge, store):
    # Give the loop room for a third round so the retry rule, not the round
    # limit, is what stops tool use.
    tool = scripted_tool("lookup", [INVALID_INPUT_RESULT, INVALID_INPUT_RESULT])
    llm = ScriptedLLM(
        [
            tool_calls(call("call_1", "lookup", '{"q": ""}')),
            tool_calls(call("call_2", "lookup", '{"q": ""}')),
            text("Sorry, I couldn't look that up."),
        ]
    )
    result = run(make(llm, knowledge, store, tools=registry_with(tool), max_tool_rounds=5), "hi")

    assert result.reply_text == "Sorry, I couldn't look that up."
    assert [r.status for r in result.tool_calls] == ["invalid_input", "invalid_input"]
    assert result.diagnostics.loop_exit == "invalid_input_retries_exhausted"
    assert result.diagnostics.final_call_tools_omitted is True
    assert "tools" not in llm.calls[2]["kwargs"]
    assert llm.exhausted


def test_tool_calls_after_retry_budget_are_rejected_not_executed(knowledge, store):
    tool = scripted_tool("lookup", [INVALID_INPUT_RESULT, INVALID_INPUT_RESULT])  # third would raise
    llm = ScriptedLLM(
        [
            tool_calls(
                call("c1", "lookup", '{"q": ""}'),
                call("c2", "lookup", '{"q": ""}'),
                call("c3", "lookup", '{"q": "third"}'),
            ),
            text("done"),
        ]
    )
    result = run(make(llm, knowledge, store, tools=registry_with(tool)), "hi")

    assert [r.disposition for r in result.tool_calls] == ["executed", "executed", "rejected_retry_budget"]
    assert result.tool_calls[2].error_code == "retry_limit_reached"
    # Every tool-call ID still gets an answer in the transcript.
    assert [m.tool_call_id for m in _tool_messages(llm.calls[1]["messages"])] == ["c1", "c2", "c3"]


# ---------------------------------------------------------------------------
# 9. unavailable tool result
# ---------------------------------------------------------------------------


def test_unavailable_result_is_fed_back_and_not_retried(knowledge, store):
    tool = scripted_tool("lookup", [UNAVAILABLE_RESULT])  # a second call would raise
    llm = ScriptedLLM([tool_calls(call("call_1", "lookup", '{"q": "x"}')), text("Catalog is down, sorry.")])
    result = run(make(llm, knowledge, store, tools=registry_with(tool), max_tool_rounds=5), "hi")

    assert result.reply_text == "Catalog is down, sorry."
    assert result.tool_calls[0].status == "unavailable"
    assert result.tool_calls[0].ok is False
    assert result.diagnostics.loop_exit == "tool_unavailable"
    assert result.diagnostics.final_call_tools_omitted is True
    assert "tools" not in llm.calls[1]["kwargs"]
    fed_back = json.loads(_tool_messages(llm.calls[1]["messages"])[0].content)
    assert fed_back["status"] == "unavailable"


def test_tool_calls_after_unavailable_in_same_round_are_rejected(knowledge, store):
    tool = scripted_tool("lookup", [UNAVAILABLE_RESULT])
    llm = ScriptedLLM(
        [tool_calls(call("c1", "lookup", '{"q": "x"}'), call("c2", "lookup", '{"q": "y"}')), text("done")]
    )
    result = run(make(llm, knowledge, store, tools=registry_with(tool)), "hi")

    assert [r.disposition for r in result.tool_calls] == ["executed", "rejected_tool_unavailable"]
    assert result.tool_calls[1].status == "unavailable"


# ---------------------------------------------------------------------------
# 10, 11, 12. Tool round limit; final call has NO tools; tool_choice="none" never sent
# ---------------------------------------------------------------------------


def test_tool_round_limit_then_final_text_only_call(knowledge, store):
    tool = scripted_tool("lookup", [OK_RESULT] * 5)
    llm = ScriptedLLM(
        [
            tool_calls(call("c1", "lookup", '{"q": "a"}')),
            tool_calls(call("c2", "lookup", '{"q": "b"}')),
            text("Final answer without more tools."),
        ]
    )
    result = run(make(llm, knowledge, store, tools=registry_with(tool)), "hi")

    assert AGENT_MAX_TOOL_ROUNDS == 2
    assert result.reply_text == "Final answer without more tools."
    assert result.diagnostics.tool_rounds == 2
    assert result.diagnostics.tool_round_limit_reached is True
    assert result.diagnostics.final_call_tools_omitted is True
    assert result.diagnostics.loop_exit == "tool_round_limit"
    assert len(llm.calls) == AGENT_MAX_TOOL_ROUNDS + 1
    assert llm.exhausted


def test_final_text_only_call_has_no_tools_parameter(knowledge, store):
    tool = scripted_tool("lookup", [OK_RESULT] * 2)
    llm = ScriptedLLM(
        [tool_calls(call("c1", "lookup", "{}")), tool_calls(call("c2", "lookup", "{}")), text("final")]
    )
    run(make(llm, knowledge, store, tools=registry_with(tool)), "hi")

    assert "tools" in llm.calls[0]["kwargs"]
    assert "tools" in llm.calls[1]["kwargs"]
    assert "tools" not in llm.calls[2]["kwargs"]


def test_tool_choice_none_is_never_sent(knowledge, store):
    tool = scripted_tool("lookup", [OK_RESULT] * 2)
    llm = ScriptedLLM(
        [tool_calls(call("c1", "lookup", "{}")), tool_calls(call("c2", "lookup", "{}")), text("final")]
    )
    run(make(llm, knowledge, store, tools=registry_with(tool)), "hi")

    for recorded in llm.calls:
        assert "tool_choice" not in recorded["kwargs"]
        assert recorded["kwargs"].get("tool_choice") != "none"


def test_round_limit_respects_custom_max_tool_rounds(knowledge, store):
    tool = scripted_tool("lookup", [OK_RESULT])
    llm = ScriptedLLM([tool_calls(call("c1", "lookup", "{}")), text("final")])
    result = run(make(llm, knowledge, store, tools=registry_with(tool), max_tool_rounds=1), "hi")

    assert result.diagnostics.tool_round_limit_reached is True
    assert "tools" not in llm.calls[1]["kwargs"]


# ---------------------------------------------------------------------------
# 13, 14, 15. LLM failure, tool failure, safe fallback
# ---------------------------------------------------------------------------


def test_llm_failure_returns_safe_fallback_and_persists_state(knowledge, store):
    llm = ScriptedLLM([LLMProviderError("Groq API error (AuthenticationError): key sk-secret-123 rejected")])
    result = run(make(llm, knowledge, store), "hello")

    assert result.reply_text == SAFE_FALLBACK_REPLY
    assert result.diagnostics.fallback_used is True
    assert result.diagnostics.fallback_reason == "llm_error"
    assert result.diagnostics.error_type == "LLMProviderError"
    assert "sk-secret-123" not in json.dumps(result.diagnostics.model_dump())
    saved = store.get(SENDER)
    assert saved is not None
    assert [m.role for m in saved.history] == ["user", "assistant"]
    assert saved.history[-1].content == SAFE_FALLBACK_REPLY
    assert saved.turn_count == 1


def test_unexpected_provider_exception_is_also_contained(knowledge, store):
    llm = ScriptedLLM([RuntimeError("boom")])
    result = run(make(llm, knowledge, store), "hello")

    assert result.reply_text == SAFE_FALLBACK_REPLY
    assert result.diagnostics.error_type == "RuntimeError"


def test_final_text_only_call_failure_returns_fallback_not_raise(knowledge, store):
    tool = scripted_tool("lookup", [OK_RESULT] * 2)
    llm = ScriptedLLM(
        [
            tool_calls(call("c1", "lookup", "{}")),
            tool_calls(call("c2", "lookup", "{}")),
            LLMProviderError("timeout"),
        ]
    )
    result = run(make(llm, knowledge, store, tools=registry_with(tool)), "hi")

    assert result.reply_text == SAFE_FALLBACK_REPLY
    assert result.diagnostics.final_call_tools_omitted is True
    assert result.diagnostics.loop_exit == "llm_error"
    # Tool work done before the failure is still persisted.
    assert len(store.get(SENDER).tool_history) == 2


def test_tool_handler_exception_is_contained_as_unavailable(knowledge, store):
    tool = scripted_tool("lookup", [RuntimeError("catalog exploded")])
    llm = ScriptedLLM([tool_calls(call("c1", "lookup", '{"q": "x"}')), text("Let me get the team to check.")])
    result = run(make(llm, knowledge, store, tools=registry_with(tool)), "hi")

    assert result.reply_text == "Let me get the team to check."
    record = result.tool_calls[0]
    assert record.disposition == "executed"
    assert record.status == "unavailable"
    assert record.error_code == "tool_execution_failed"
    fed_back = json.loads(_tool_messages(llm.calls[1]["messages"])[0].content)
    assert "catalog exploded" not in json.dumps(fed_back)


def test_empty_model_reply_falls_back_safely(knowledge, store):
    llm = ScriptedLLM([LLMResponse(content="", finish_reason="stop")])
    result = run(make(llm, knowledge, store), "hello")

    assert result.reply_text == SAFE_FALLBACK_REPLY
    assert result.diagnostics.fallback_reason == "empty_reply"
    assert result.diagnostics.loop_exit == "empty_reply"


def test_safe_fallback_is_deterministic_and_exposes_nothing_internal():
    assert SAFE_FALLBACK_REPLY == "Sorry — I’m having trouble checking that right now. Let me get the team to help."
    for word in ("Groq", "LLM", "API", "token", "exception", "Traceback"):
        assert word.lower() not in SAFE_FALLBACK_REPLY.lower()


# ---------------------------------------------------------------------------
# 16, 17, 18, 19, 20. State load / save / message stored / tool recorded / reset
# ---------------------------------------------------------------------------


def test_state_is_loaded_from_store(knowledge, store):
    existing = ConversationState.new(SENDER)
    existing.begin_turn()
    existing.add_user_message("earlier question")
    existing.add_assistant_message("earlier answer")
    existing.remember_fact("preferred_roast", "dark")
    store.save(existing)

    llm = ScriptedLLM([text("welcome back")])
    result = run(make(llm, knowledge, store), "I'm back")

    assert result.state_snapshot.turn_count == 2
    assert result.state_snapshot.known_facts == {"preferred_roast": "dark"}
    sent = llm.calls[0]["messages"]
    assert [m.content for m in sent[1:3]] == ["earlier question", "earlier answer"]
    assert "preferred_roast=dark" in sent[0].content


def test_new_sender_gets_fresh_state(knowledge, store):
    assert SENDER not in store
    result = run(make(ScriptedLLM([text("hi")]), knowledge, store), "hello")

    assert SENDER in store
    assert result.state_snapshot.turn_count == 1
    assert result.state_snapshot.lead.whatsapp_number == SENDER


def test_state_is_saved_after_turn(knowledge, store):
    result = run(make(ScriptedLLM([text("reply")]), knowledge, store), "hello")

    saved = store.get(SENDER)
    assert saved is not None
    assert saved.model_dump(mode="json") == result.state_snapshot.model_dump(mode="json")
    assert saved.turn_count == 1


def test_incoming_message_is_stored(knowledge, store):
    run(make(ScriptedLLM([text("reply")]), knowledge, store), "  What are your hours?  ")

    saved = store.get(SENDER)
    assert saved.history[0].role == "user"
    assert saved.history[0].content == "What are your hours?"
    assert saved.history[0].turn == 1
    assert saved.history[1].role == "assistant"
    assert saved.history[1].content == "reply"


def test_blank_message_is_rejected_before_touching_state(knowledge, store):
    orchestrator = make(ScriptedLLM([]), knowledge, store)
    with pytest.raises(ValueError):
        run(orchestrator, "   ")
    assert SENDER not in store


def test_tool_invocation_is_recorded_in_state(knowledge, store):
    llm = ScriptedLLM([tool_calls(call("call_1")), text("done")])
    result = run(make(llm, knowledge, store), "Do you have Ethiopia?")

    saved = store.get(SENDER)
    assert len(saved.tool_history) == 1
    invocation = saved.tool_history[0]
    assert invocation.tool_name == "product_lookup"
    assert invocation.turn == 1
    assert invocation.status == "ok"
    assert invocation.ok is True
    assert invocation.arguments == {"query": "ethiopia"}
    assert invocation.result is None  # history entries drop the payload
    # The in-turn record keeps the result for grounding.
    assert result.state_snapshot.current_turn_tool_results[0].result["status"] == "ok"


def test_current_turn_tool_results_reset_each_turn(knowledge, store):
    llm = ScriptedLLM([tool_calls(call("call_1")), text("done"), text("plain reply")])
    orchestrator = make(llm, knowledge, store)

    first = run(orchestrator, "Do you have Ethiopia?")
    assert len(first.state_snapshot.current_turn_tool_results) == 1

    second = run(orchestrator, "thanks")
    assert second.state_snapshot.current_turn_tool_results == []
    assert len(second.state_snapshot.tool_history) == 1  # history survives
    # The second turn's prompt does not carry the first turn's tool results.
    assert "Tool results for this turn" not in llm.calls[2]["messages"][0].content


# ---------------------------------------------------------------------------
# 21 & 22. Sender isolation; bounded history preserved
# ---------------------------------------------------------------------------


def test_sender_isolation(knowledge, store):
    llm = ScriptedLLM([text("reply A"), text("reply B")])
    orchestrator = make(llm, knowledge, store)

    run(orchestrator, "message from A", sender=SENDER)
    run(orchestrator, "message from B", sender=OTHER_SENDER)

    state_a, state_b = store.get(SENDER), store.get(OTHER_SENDER)
    assert [m.content for m in state_a.history] == ["message from A", "reply A"]
    assert [m.content for m in state_b.history] == ["message from B", "reply B"]
    sent_for_b = " ".join(m.content for m in llm.calls[1]["messages"])
    assert "message from A" not in sent_for_b
    assert "reply A" not in sent_for_b


def test_bounded_history_is_preserved_across_turns(knowledge, store):
    turns = MAX_CHAT_HISTORY  # 2 messages per turn -> exceeds the cap
    llm = ScriptedLLM([text(f"reply {i}") for i in range(turns)])
    orchestrator = make(llm, knowledge, store)
    for i in range(turns):
        run(orchestrator, f"message {i}")

    saved = store.get(SENDER)
    assert saved.turn_count == turns
    assert len(saved.history) == MAX_CHAT_HISTORY
    assert saved.history[0].content == f"message {turns - MAX_CHAT_HISTORY // 2}"
    assert saved.history[-1].content == f"reply {turns - 1}"
    # The last prompt carried the bounded prior history verbatim, oldest
    # first, followed by the current message exactly once (delimited).
    last_prompt = llm.calls[-1]["messages"]
    history_sent = [m.content for m in last_prompt[1:-1]]
    assert history_sent[-1] == f"reply {turns - 2}"
    assert len(history_sent) == MAX_CHAT_HISTORY
    assert f"message {turns - 1}" not in history_sent
    assert f"message {turns - 1}" in last_prompt[-1].content


def test_current_message_is_sent_once_and_delimited(knowledge, store):
    llm = ScriptedLLM([text("reply")])
    run(make(llm, knowledge, store), "unique-phrase-42")

    sent = llm.calls[0]["messages"]
    occurrences = sum(m.content.count("unique-phrase-42") for m in sent)
    assert occurrences == 1
    assert sent[-1].role == "user"
    assert sent[-1].content.startswith("<customer_message>")


# ---------------------------------------------------------------------------
# 23 & 24. AgentTurnResult structure; diagnostics contain no secrets
# ---------------------------------------------------------------------------


def test_agent_turn_result_structure(knowledge, store):
    usage = TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120)
    llm = ScriptedLLM([tool_calls(call("call_1")), text("done", usage=usage)])
    result = run(make(llm, knowledge, store), "Do you have Ethiopia?", message_id="wamid.ABC")

    assert isinstance(result, AgentTurnResult)
    assert isinstance(result.reply_text, str) and result.reply_text
    assert isinstance(result.state_snapshot, ConversationState)
    assert all(isinstance(r, ToolCallRecord) for r in result.tool_calls)
    assert isinstance(result.diagnostics, TurnDiagnostics)
    assert result.diagnostics.message_id == "wamid.ABC"
    assert result.diagnostics.turn == 1
    assert result.diagnostics.usage_prompt_tokens == 100
    assert result.diagnostics.usage_completion_tokens == 20
    # Frozen result models.
    with pytest.raises(Exception):
        result.reply_text = "changed"
    # Round-trips to JSON.
    json.dumps(result.model_dump(mode="json"))


def test_snapshot_is_detached_from_store(knowledge, store):
    result = run(make(ScriptedLLM([text("hi")]), knowledge, store), "hello")
    result.state_snapshot.remember_fact("tampered", "yes")
    assert "tampered" not in store.get(SENDER).known_facts


def test_diagnostics_contain_no_secrets_or_full_phone_number(knowledge, store):
    llm = ScriptedLLM([tool_calls(call("call_1")), LLMProviderError("Bearer mock_groq_api_key_67890 rejected")])
    result = run(make(llm, knowledge, store), "Do you have Ethiopia?", message_id="wamid.X")

    safe_blob = json.dumps(
        {"diagnostics": result.diagnostics.model_dump(), "tool_calls": [r.model_dump() for r in result.tool_calls]}
    )
    assert SENDER not in safe_blob
    assert result.diagnostics.sender == "********3210"
    for var in _SECRET_ENV_VARS:
        assert os.environ[var] not in safe_blob
    assert "Bearer" not in safe_blob
    assert "Authorization" not in safe_blob
    assert result.diagnostics.error_type == "LLMProviderError"


# ---------------------------------------------------------------------------
# 25. No network activity
# ---------------------------------------------------------------------------


def test_no_network_activity_during_a_full_tool_turn(knowledge, store, monkeypatch):
    # asyncio's event loop uses a loopback socketpair internally (notably on
    # Windows), so only non-loopback destinations count as network access.
    loopback = {"127.0.0.1", "::1", "localhost"}
    original_connect = socket.socket.connect
    original_getaddrinfo = socket.getaddrinfo

    def _guarded_connect(sock, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if host not in loopback:
            raise AssertionError(f"network access attempted during orchestrator turn: {host!r}")
        return original_connect(sock, address, *args, **kwargs)

    def _guarded_getaddrinfo(host, *args, **kwargs):
        if host not in loopback:
            raise AssertionError(f"DNS lookup attempted during orchestrator turn: {host!r}")
        return original_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", _guarded_getaddrinfo)

    llm = ScriptedLLM([tool_calls(call("call_1")), text("done")])
    result = run(make(llm, knowledge, store), "Do you have Ethiopia?")
    assert result.reply_text == "done"
    assert result.tool_calls[0].status == "ok"


def test_orchestrator_never_uses_get_agent_reply(knowledge, store):
    llm = ScriptedLLM([tool_calls(call("call_1")), text("done")])
    llm.get_agent_reply = AsyncMock(side_effect=AssertionError("legacy path used"))
    run(make(llm, knowledge, store), "Do you have Ethiopia?")
    llm.get_agent_reply.assert_not_called()


# ---------------------------------------------------------------------------
# Tool transcript reaches GroqProvider in Groq/OpenAI wire format (mocked client)
# ---------------------------------------------------------------------------


def _completion(content=None, tool_calls=None, finish_reason="stop"):
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    message.reasoning = None
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason
    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = None
    return completion


def test_tool_transcript_is_wire_compatible_with_groq_provider(knowledge, store):
    tool_call = MagicMock()
    tool_call.id = "call_groq_1"
    tool_call.function = MagicMock()
    tool_call.function.name = "product_lookup"
    tool_call.function.arguments = '{"query": "ethiopia"}'

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=[
            _completion(content=None, tool_calls=[tool_call], finish_reason="tool_calls"),
            _completion(content="Grounded reply", finish_reason="stop"),
        ]
    )
    provider = GroqProvider(api_key="test_groq_api_key", model="openai/gpt-oss-120b", client=mock_client)

    result = run(make(provider, knowledge, store), "Do you have Ethiopia?")

    assert result.reply_text == "Grounded reply"
    first_kwargs = mock_client.chat.completions.create.call_args_list[0].kwargs
    assert first_kwargs["tools"] == build_default_registry().list_specs()
    assert "tool_choice" not in first_kwargs

    second_kwargs = mock_client.chat.completions.create.call_args_list[1].kwargs
    sent = second_kwargs["messages"]
    assistant_msg, tool_msg = sent[-2], sent[-1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["tool_calls"][0]["id"] == "call_groq_1"
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "product_lookup"
    assert tool_msg == {"role": "tool", "content": tool_msg["content"], "tool_call_id": "call_groq_1"}
    assert json.loads(tool_msg["content"])["status"] == "ok"
    # Plain messages keep their exact two-key shape.
    assert set(sent[0].keys()) == {"role", "content"}
    assert all(set(m.keys()) == {"role", "content"} for m in sent[:-2])


def test_plain_chat_message_has_no_tool_fields_by_default():
    message = ChatMessage(role="user", content="hi")
    assert message.tool_calls is None
    assert message.tool_call_id is None
    assert message.model_dump(exclude_none=True) == {"role": "user", "content": "hi"}


# ---------------------------------------------------------------------------
# 26. Orchestrator does not touch app/main.py behaviour
# ---------------------------------------------------------------------------


def test_orchestrator_is_not_wired_into_app_main():
    import app.main as main_module
    import app.agent.orchestrator as orchestrator_module

    main_source = inspect.getsource(main_module)
    assert "orchestrator" not in main_source
    assert "AgentOrchestrator" not in main_source
    assert "ConversationStore" not in main_source
    assert "PromptBuilder" not in main_source

    orchestrator_source = inspect.getsource(orchestrator_module)
    import_pattern = re.compile(r"^\s*(from|import)\s+app\.(main|whatsapp)", re.MULTILINE)
    assert import_pattern.search(orchestrator_source) is None


def test_milestone_one_webhook_path_still_uses_legacy_provider(client, mock_llm, mock_wa, valid_text_payload):
    response = client.post("/webhook/whatsapp", json=valid_text_payload)

    assert response.status_code == 200
    assert len(mock_llm.calls) == 1  # get_agent_reply, the Milestone 1 path
    assert len(mock_wa.sent_messages) == 1


# ===========================================================================
# Slice 8: lead extraction integrated into the orchestrator
# ===========================================================================
#
# Every test below wires a real ``LeadExtractor`` over its own ``ScriptedLLM``
# (so the extraction call never competes with the agent script), or a small
# duck-typed extractor that raises. No Groq or network calls anywhere.


class LoggingLLM(ScriptedLLM):
    """``ScriptedLLM`` that also appends a label to a shared log on every call, to prove ordering."""

    def __init__(self, script, log: List[str], label: str):
        super().__init__(script)
        self._log = log
        self._label = label

    async def complete(self, messages: List[ChatMessage], **kwargs) -> LLMResponse:
        self._log.append(self._label)
        return await super().complete(messages, **kwargs)


class RaisingExtractor:
    """Extractor whose implementation crashes; must never fail the customer turn."""

    def __init__(self, exc: Exception):
        self._exc = exc
        self.calls = 0

    async def extract(self, message: str, state=None, profile=None):
        self.calls += 1
        raise self._exc


def extraction_payload(**overrides) -> LLMResponse:
    """A complete extraction JSON object (all fields null) with ``overrides`` applied."""
    payload = {name: None for name in ALLOWED_EXTRACTION_FIELDS}
    payload.update(overrides)
    return text(json.dumps(payload))


WHOLESALE_FIELDS_RAHUL = dict(
    track="wholesale",
    contact_name="Rahul",
    business_name="Bean House",
    business_type="cafe",
    monthly_volume_kg=25,
    city="Bengaluru",
    timeline="within_1_month",
)

CONSUMER_FIELDS_PRIYA = dict(
    track="consumer",
    contact_name="Priya",
    brew_method="pourover",
    taste_preference="fruity",
    budget_band="500_1000",
    subscription_interest=True,
)


def make_with_extraction(
    agent_llm,
    extractor_llm,
    knowledge,
    store,
    tools: Optional[ToolRegistry] = None,
    **kwargs,
) -> AgentOrchestrator:
    return make(agent_llm, knowledge, store, tools=tools, extractor=LeadExtractor(extractor_llm), **kwargs)


def _state_with_lead(**fields) -> ConversationState:
    """A fresh sender state with ``fields`` already merged (provenance turn 0)."""
    state = ConversationState.new(SENDER)
    state.apply_lead_delta(LeadDelta(**fields))
    return state


def _system_prompt(llm_call: Dict[str, Any]) -> str:
    return llm_call["messages"][0].content


def _delimited_user_messages(messages: List[ChatMessage]) -> List[ChatMessage]:
    return [m for m in messages if m.role == "user" and m.content.startswith(CUSTOMER_MESSAGE_OPEN)]


# ---------------------------------------------------------------------------
# 1-3. Extraction runs before prompt generation and is visible to the model
# ---------------------------------------------------------------------------


def test_successful_extraction_updates_state_before_prompt_generation(knowledge, store):
    log: List[str] = []
    agent_llm = LoggingLLM([text("Nice to meet you, Rahul!")], log, "agent")
    extractor_llm = LoggingLLM([extraction_payload(contact_name="Rahul", city="Bengaluru")], log, "extract")
    result = run(make_with_extraction(agent_llm, extractor_llm, knowledge, store), "Hi, I'm Rahul from Bengaluru")

    assert log == ["extract", "agent"]
    assert result.reply_text == "Nice to meet you, Rahul!"
    # The first (and only) agent prompt already reflects the merged lead state.
    assert "name=Rahul" in _system_prompt(agent_llm.calls[0])
    assert "city=Bengaluru" in _system_prompt(agent_llm.calls[0])


def test_extracted_name_appears_in_returned_state_snapshot(knowledge, store):
    result = run(
        make_with_extraction(ScriptedLLM([text("ok")]), ScriptedLLM([extraction_payload(contact_name="Rahul")]), knowledge, store),
        "I'm Rahul",
    )
    assert result.state_snapshot.lead.contact_name == "Rahul"
    assert result.state_snapshot.lead.field_provenance == {"contact_name": 1}
    assert result.state_snapshot.lead.whatsapp_number == SENDER  # webhook metadata untouched


def test_extracted_lead_track_appears_before_first_llm_call(knowledge, store):
    agent_llm = ScriptedLLM([text("ok")])
    run(
        make_with_extraction(agent_llm, ScriptedLLM([extraction_payload(track="wholesale")]), knowledge, store),
        "I run a cafe and want bulk beans",
    )
    assert "track=wholesale" in _system_prompt(agent_llm.calls[0])
    assert "qualification: collecting" in _system_prompt(agent_llm.calls[0])


# ---------------------------------------------------------------------------
# 4-6. Deterministic qualification; consumer and wholesale fields
# ---------------------------------------------------------------------------


def test_extraction_updates_qualification_deterministically(knowledge, store):
    orchestrator = make_with_extraction(
        ScriptedLLM([text("one"), text("two"), text("three")]),
        ScriptedLLM(
            [
                extraction_payload(),  # nothing said -> browsing
                extraction_payload(track="wholesale", contact_name="Rahul"),  # partial -> collecting
                extraction_payload(**{k: v for k, v in WHOLESALE_FIELDS_RAHUL.items() if k != "contact_name"}),
            ]
        ),
        knowledge,
        store,
    )
    first = run(orchestrator, "hello")
    assert first.state_snapshot.qualification == QualificationState.BROWSING

    second = run(orchestrator, "I'm Rahul, I run a cafe")
    assert second.state_snapshot.qualification == QualificationState.COLLECTING

    third = run(orchestrator, "Bean House in Bengaluru, about 25kg a month, starting next month")
    lead = third.state_snapshot.lead
    assert lead.track == LeadTrack.WHOLESALE
    assert lead.contact_name == "Rahul"
    assert lead.business_name == "Bean House"
    assert lead.business_type.value == "cafe"
    assert lead.monthly_volume_kg == 25
    assert lead.city == "Bengaluru"
    assert lead.timeline.value == "within_1_month"
    assert lead.missing_required_fields() == []
    # Computed by Python from the merged profile, identical to calling the rule directly.
    assert third.state_snapshot.qualification == QualificationState.QUALIFIED
    assert third.state_snapshot.qualification == evaluate_qualification(lead, QualificationState.COLLECTING, 3)
    assert lead.field_provenance["contact_name"] == 2
    assert lead.field_provenance["business_name"] == 3


def test_consumer_extraction_updates_consumer_fields(knowledge, store):
    result = run(
        make_with_extraction(ScriptedLLM([text("ok")]), ScriptedLLM([extraction_payload(**CONSUMER_FIELDS_PRIYA)]), knowledge, store),
        "I'm Priya, pourover, fruity, 500-1000 per order, monthly subscription please",
    )
    lead = result.state_snapshot.lead
    assert lead.track == LeadTrack.CONSUMER
    assert lead.contact_name == "Priya"
    assert lead.brew_method.value == "pourover"
    assert lead.taste_preference == "fruity"
    assert lead.budget_band.value == "500_1000"
    assert lead.subscription_interest is True
    assert lead.business_name is None
    assert result.state_snapshot.qualification == QualificationState.QUALIFIED
    assert result.diagnostics.extracted_field_names == list(CONSUMER_FIELDS_PRIYA.keys())


def test_wholesale_extraction_updates_wholesale_fields(knowledge, store):
    result = run(
        make_with_extraction(
            ScriptedLLM([text("ok")]),
            ScriptedLLM([extraction_payload(**WHOLESALE_FIELDS_RAHUL, current_supplier="Local roaster")]),
            knowledge,
            store,
        ),
        "Rahul from Bean House cafe, Bengaluru, 25kg/month, this month, currently with a local roaster",
    )
    lead = result.state_snapshot.lead
    assert lead.track == LeadTrack.WHOLESALE
    assert lead.business_name == "Bean House"
    assert lead.business_type.value == "cafe"
    assert lead.monthly_volume_kg == 25
    assert lead.timeline.value == "within_1_month"
    assert lead.current_supplier == "Local roaster"
    assert lead.brew_method is None
    assert result.state_snapshot.qualification == QualificationState.QUALIFIED


# ---------------------------------------------------------------------------
# 7. Null extraction keeps existing values; corrections replace them
# ---------------------------------------------------------------------------


def test_null_extraction_does_not_erase_existing_lead_values(knowledge, store):
    orchestrator = make_with_extraction(
        ScriptedLLM([text("one"), text("two")]),
        ScriptedLLM([extraction_payload(contact_name="Rahul", city="Bengaluru"), extraction_payload()]),
        knowledge,
        store,
    )
    run(orchestrator, "I'm Rahul from Bengaluru")
    result = run(orchestrator, "what are your hours?")

    lead = result.state_snapshot.lead
    assert lead.contact_name == "Rahul"
    assert lead.city == "Bengaluru"
    assert lead.field_provenance == {"contact_name": 1, "city": 1}
    assert result.diagnostics.extraction_success is True
    assert result.diagnostics.extracted_field_names == []
    assert store.get(SENDER).lead.city == "Bengaluru"


def test_corrected_extraction_value_replaces_previous_with_new_provenance(knowledge, store):
    orchestrator = make_with_extraction(
        ScriptedLLM([text("one"), text("two")]),
        ScriptedLLM([extraction_payload(city="Bengaluru"), extraction_payload(city="Mysuru")]),
        knowledge,
        store,
    )
    run(orchestrator, "I'm in Bengaluru")
    result = run(orchestrator, "sorry, I meant Mysuru")

    assert result.state_snapshot.lead.city == "Mysuru"
    assert result.state_snapshot.lead.field_provenance == {"city": 2}


# ---------------------------------------------------------------------------
# 8-10, 26. Extraction failure is non-fatal and never triggers the safe fallback
# ---------------------------------------------------------------------------


def test_extraction_failure_is_non_fatal(knowledge, store):
    store.save(_state_with_lead(contact_name="Rahul"))
    # Non-JSON twice: initial attempt + one repair -> fallback result.
    extractor_llm = ScriptedLLM([text("I cannot help with that"), text("still not json")])
    result = run(make_with_extraction(ScriptedLLM([text("Normal reply")]), extractor_llm, knowledge, store), "hello")

    assert result.reply_text == "Normal reply"
    assert result.diagnostics.fallback_used is False
    assert result.diagnostics.extraction_attempted is True
    assert result.diagnostics.extraction_success is False
    assert result.diagnostics.extraction_source == "fallback"
    assert result.diagnostics.extraction_errors_count >= 1
    assert result.diagnostics.extracted_field_names == []
    assert len(extractor_llm.calls) == 2  # exactly one repair attempt, no more
    # Lead untouched, turn still persisted normally.
    assert result.state_snapshot.lead.contact_name == "Rahul"
    assert result.state_snapshot.lead.field_provenance == {"contact_name": 0}
    assert [m.role for m in store.get(SENDER).history] == ["user", "assistant"]


def test_extraction_provider_exception_is_non_fatal(knowledge, store):
    store.save(_state_with_lead(contact_name="Rahul"))
    extractor_llm = ScriptedLLM([LLMProviderError("Bearer mock_groq_api_key_67890 rejected")])
    result = run(make_with_extraction(ScriptedLLM([text("Normal reply")]), extractor_llm, knowledge, store), "hello")

    assert result.reply_text == "Normal reply"
    assert result.diagnostics.fallback_used is False
    assert result.diagnostics.extraction_success is False
    assert result.diagnostics.extraction_source == "fallback"
    assert len(extractor_llm.calls) == 1  # provider errors are not repaired
    assert result.state_snapshot.lead.contact_name == "Rahul"
    assert "mock_groq_api_key_67890" not in json.dumps(result.diagnostics.model_dump())


def test_malformed_extraction_output_is_non_fatal(knowledge, store):
    # Valid JSON but the wrong shape, then a repair that is also invalid.
    extractor_llm = ScriptedLLM([text('["not", "an", "object"]'), text('{"monthly_volume_kg": "lots"}')])
    result = run(make_with_extraction(ScriptedLLM([text("Normal reply")]), extractor_llm, knowledge, store), "hello")

    assert result.reply_text == "Normal reply"
    assert result.diagnostics.fallback_used is False
    assert result.diagnostics.extraction_success is False
    assert result.state_snapshot.lead.has_any_lead_data() is False


def test_repaired_extraction_output_is_applied(knowledge, store):
    extractor_llm = ScriptedLLM([text("not json"), extraction_payload(contact_name="Rahul")])
    result = run(make_with_extraction(ScriptedLLM([text("ok")]), extractor_llm, knowledge, store), "I'm Rahul")

    assert result.diagnostics.extraction_success is True
    assert result.diagnostics.extraction_source == "model_repaired"
    assert result.state_snapshot.lead.contact_name == "Rahul"


def test_unexpected_extractor_exception_is_contained(knowledge, store):
    store.save(_state_with_lead(contact_name="Rahul"))
    extractor = RaisingExtractor(RuntimeError("secret detail: Bearer mock_groq_api_key_67890"))
    result = run(make(ScriptedLLM([text("Normal reply")]), knowledge, store, extractor=extractor), "hello")

    assert extractor.calls == 1
    assert result.reply_text == "Normal reply"
    assert result.diagnostics.fallback_used is False
    assert result.diagnostics.extraction_attempted is True
    assert result.diagnostics.extraction_success is False
    assert result.diagnostics.extraction_error_type == "RuntimeError"
    assert result.state_snapshot.lead.contact_name == "Rahul"
    assert "secret detail" not in json.dumps(result.diagnostics.model_dump())
    assert store.get(SENDER).turn_count == 1


def test_extraction_failure_does_not_trigger_safe_fallback_by_itself(knowledge, store):
    extractor_llm = ScriptedLLM([LLMProviderError("extractor down")])
    result = run(make_with_extraction(ScriptedLLM([text("Still fine")]), extractor_llm, knowledge, store), "hello")

    assert result.reply_text == "Still fine"
    assert result.reply_text != SAFE_FALLBACK_REPLY
    assert result.diagnostics.fallback_used is False
    assert result.diagnostics.fallback_reason is None
    assert result.diagnostics.error_type is None
    assert result.diagnostics.loop_exit == "text_reply"


# ---------------------------------------------------------------------------
# 11, 12, 14. Exactly one extraction per turn; tool loop unaffected
# ---------------------------------------------------------------------------


def test_extraction_occurs_exactly_once_per_turn(knowledge, store):
    extractor_llm = ScriptedLLM([extraction_payload(contact_name="Rahul"), extraction_payload()])
    orchestrator = make_with_extraction(ScriptedLLM([text("one"), text("two")]), extractor_llm, knowledge, store)

    run(orchestrator, "I'm Rahul")
    assert len(extractor_llm.calls) == 1
    run(orchestrator, "hello again")
    assert len(extractor_llm.calls) == 2


def test_extraction_does_not_occur_once_per_tool_round(knowledge, store):
    tool = scripted_tool("lookup", [OK_RESULT, OK_RESULT])
    agent_llm = ScriptedLLM(
        [
            tool_calls(call("call_1", "lookup", '{"q": "a"}')),
            tool_calls(call("call_2", "lookup", '{"q": "b"}')),
            text("final"),
        ]
    )
    extractor_llm = ScriptedLLM([extraction_payload(contact_name="Rahul")])
    result = run(
        make_with_extraction(agent_llm, extractor_llm, knowledge, store, tools=registry_with(tool)),
        "I'm Rahul, two things",
    )

    assert result.reply_text == "final"
    assert result.diagnostics.tool_rounds == AGENT_MAX_TOOL_ROUNDS
    assert result.diagnostics.llm_calls == 3
    assert len(extractor_llm.calls) == 1
    assert extractor_llm.exhausted
    # Tool calls never reach the extractor's provider.
    assert all("tools" not in c["kwargs"] for c in extractor_llm.calls)


def test_tool_loop_still_works_after_extraction(knowledge, store):
    agent_llm = ScriptedLLM([tool_calls(call("call_1")), text("Yes! Ethiopia is in stock.")])
    extractor_llm = ScriptedLLM([extraction_payload(contact_name="Rahul")])
    result = run(make_with_extraction(agent_llm, extractor_llm, knowledge, store), "I'm Rahul, do you have Ethiopia?")

    assert result.reply_text == "Yes! Ethiopia is in stock."
    assert [r.disposition for r in result.tool_calls] == ["executed"]
    assert result.tool_calls[0].status == "ok"
    assert result.state_snapshot.lead.contact_name == "Rahul"
    assert result.state_snapshot.current_turn_tool_results[0].tool_name == "product_lookup"
    # Every agent call in the loop sees the updated lead state.
    assert all("name=Rahul" in _system_prompt(c) for c in agent_llm.calls)
    second_call = agent_llm.calls[1]["messages"]
    assert second_call[-2].role == "assistant" and second_call[-1].role == "tool"


# ---------------------------------------------------------------------------
# 13, 21. Prompt receives updated state; current message still sent once
# ---------------------------------------------------------------------------


def test_prompt_receives_updated_state(knowledge, store):
    agent_llm = ScriptedLLM([text("ok")])
    run(
        make_with_extraction(agent_llm, ScriptedLLM([extraction_payload(**WHOLESALE_FIELDS_RAHUL)]), knowledge, store),
        "Rahul from Bean House",
    )
    system_prompt = _system_prompt(agent_llm.calls[0])
    assert "qualification: qualified" in system_prompt
    assert "name=Rahul" in system_prompt
    assert "city=Bengaluru" in system_prompt
    assert "track=wholesale" in system_prompt


def test_duplicate_current_message_prevention_still_works_with_extraction(knowledge, store):
    agent_llm = ScriptedLLM([tool_calls(call("call_1")), text("done")])
    extractor_llm = ScriptedLLM([extraction_payload(contact_name="Rahul")])
    orchestrator = make_with_extraction(agent_llm, extractor_llm, knowledge, store)
    run(orchestrator, "I'm Rahul, do you have Ethiopia?")

    for llm_call in agent_llm.calls:
        messages = llm_call["messages"]
        delimited = _delimited_user_messages(messages)
        assert len(delimited) == 1
        assert "I'm Rahul, do you have Ethiopia?" in delimited[0].content
        assert sum("I'm Rahul, do you have Ethiopia?" in m.content for m in messages) == 1
    # The extractor sees the message once too, delimited.
    extractor_messages = extractor_llm.calls[0]["messages"]
    assert len(_delimited_user_messages(extractor_messages)) == 1
    # History is written once, after the loop.
    assert [m.role for m in store.get(SENDER).history] == ["user", "assistant"]


# ---------------------------------------------------------------------------
# 15, 22. Persistence and sender isolation
# ---------------------------------------------------------------------------


def test_extracted_state_persists_to_conversation_store(knowledge, store):
    run(
        make_with_extraction(ScriptedLLM([text("ok")]), ScriptedLLM([extraction_payload(**WHOLESALE_FIELDS_RAHUL)]), knowledge, store),
        "Rahul from Bean House",
    )
    stored = store.get(SENDER)
    assert stored.lead.contact_name == "Rahul"
    assert stored.lead.business_name == "Bean House"
    assert stored.qualification == QualificationState.QUALIFIED
    assert stored.lead.field_provenance["business_name"] == 1
    # Round-trips through the store's JSON snapshot.
    restored = ConversationStore.from_snapshot(store.snapshot()).get(SENDER)
    assert restored.lead.business_name == "Bean House"


def test_sender_isolation_with_extraction(knowledge, store):
    orchestrator = make_with_extraction(
        ScriptedLLM([text("one"), text("two")]),
        ScriptedLLM([extraction_payload(contact_name="Rahul"), extraction_payload(contact_name="Priya")]),
        knowledge,
        store,
    )
    run(orchestrator, "I'm Rahul", sender=SENDER)
    run(orchestrator, "I'm Priya", sender=OTHER_SENDER)

    assert store.get(SENDER).lead.contact_name == "Rahul"
    assert store.get(OTHER_SENDER).lead.contact_name == "Priya"
    assert store.get(SENDER).lead.whatsapp_number == SENDER
    assert store.get(OTHER_SENDER).lead.whatsapp_number == OTHER_SENDER


# ---------------------------------------------------------------------------
# 16, 17. Diagnostics and extraction prompt are safe
# ---------------------------------------------------------------------------


def test_extraction_diagnostics_are_safe(knowledge, store):
    extractor_llm = ScriptedLLM([extraction_payload(**WHOLESALE_FIELDS_RAHUL, email="rahul@example.com")])
    result = run(
        make_with_extraction(ScriptedLLM([text("ok")]), extractor_llm, knowledge, store),
        "Rahul from Bean House, rahul@example.com",
        message_id="wamid.X",
    )
    diagnostics = result.diagnostics
    assert diagnostics.extraction_attempted is True
    assert diagnostics.extraction_success is True
    assert diagnostics.extraction_source == "model"
    assert set(diagnostics.extracted_field_names) == set(WHOLESALE_FIELDS_RAHUL) | {"email"}
    assert diagnostics.extraction_errors_count == 0
    assert diagnostics.extraction_error_type is None

    safe_blob = json.dumps(diagnostics.model_dump())
    # Field names only: no extracted values, no phone number, no secrets.
    for value in ("Rahul", "Bean House", "Bengaluru", "rahul@example.com"):
        assert value not in safe_blob
    assert SENDER not in safe_blob
    assert "raw_response" not in safe_blob
    for var in _SECRET_ENV_VARS:
        assert os.environ[var] not in safe_blob
    assert "Bearer" not in safe_blob
    assert "Authorization" not in safe_blob
    json.dumps(result.model_dump(mode="json"))


def test_no_credentials_enter_extraction_prompt(knowledge, store):
    extractor_llm = ScriptedLLM([extraction_payload(contact_name="Rahul")])
    run(make_with_extraction(ScriptedLLM([text("ok")]), extractor_llm, knowledge, store), "I'm Rahul")

    assert len(extractor_llm.calls) == 1
    prompt_blob = json.dumps([m.model_dump() for m in extractor_llm.calls[0]["messages"]])
    for var in _SECRET_ENV_VARS:
        assert os.environ[var] not in prompt_blob
    assert "Bearer" not in prompt_blob
    assert "Authorization" not in prompt_blob
    assert SENDER not in prompt_blob
    assert extractor_llm.calls[0]["kwargs"] == {}  # plain completion: no tools offered


# ---------------------------------------------------------------------------
# 18, 19. Model output cannot inject qualification or escalation
# ---------------------------------------------------------------------------


def test_qualification_cannot_be_injected_by_model_output(knowledge, store):
    injected = {
        "contact_name": "Rahul",
        "qualification": "qualified",
        "qualified": True,
        "handoff_ready": True,
        "declined": True,
    }
    extractor_llm = ScriptedLLM([text(json.dumps(injected))])
    result = run(make_with_extraction(ScriptedLLM([text("ok")]), extractor_llm, knowledge, store), "I'm Rahul")

    state = result.state_snapshot
    assert state.lead.contact_name == "Rahul"
    # Only a name is known: Python computes ``collecting``, whatever the model claimed.
    assert state.qualification == QualificationState.COLLECTING
    assert state.qualification == evaluate_qualification(state.lead, QualificationState.UNKNOWN, 1)
    assert state.flags.declines == 0
    assert result.diagnostics.extraction_success is True
    assert result.diagnostics.extraction_errors_count == 4  # the dropped fields
    assert result.diagnostics.extracted_field_names == ["contact_name"]
    assert not any(name in result.diagnostics.extracted_field_names for name in injected if name != "contact_name")


def test_escalation_cannot_be_injected_by_model_output(knowledge, store):
    injected = {
        "contact_name": "Rahul",
        "escalated": True,
        "escalation": {"status": "pending", "reason": "angry"},
        "escalation_status": "handed_off",
    }
    extractor_llm = ScriptedLLM([text(json.dumps(injected))])
    agent_llm = ScriptedLLM([text("ok")])
    result = run(make_with_extraction(agent_llm, extractor_llm, knowledge, store), "I'm Rahul")

    state = result.state_snapshot
    assert state.escalation.status == EscalationStatus.NONE
    assert state.escalation.reason is None
    assert state.qualification == QualificationState.COLLECTING
    assert state.qualification != QualificationState.ESCALATED
    assert store.get(SENDER).escalation.status == EscalationStatus.NONE
    assert "escalation_status: none" in _system_prompt(agent_llm.calls[0])
    assert "qualification: collecting" in _system_prompt(agent_llm.calls[0])


# ---------------------------------------------------------------------------
# 20, 23, 24, 25. Existing orchestrator behaviour remains intact
# ---------------------------------------------------------------------------


def test_orchestrator_without_extractor_behaves_as_before(knowledge, store):
    llm = ScriptedLLM([text("hi")])
    result = run(make(llm, knowledge, store), "hello")

    assert len(llm.calls) == 1
    assert result.diagnostics.extraction_attempted is False
    assert result.diagnostics.extraction_success is False
    assert result.diagnostics.extraction_source is None
    assert result.diagnostics.extracted_field_names == []
    assert result.diagnostics.extraction_errors_count == 0
    assert result.diagnostics.extraction_error_type is None
    assert result.state_snapshot.lead.has_any_lead_data() is False
    assert result.state_snapshot.qualification == QualificationState.UNKNOWN


def test_maximum_tool_round_behaviour_remains_intact_with_extraction(knowledge, store):
    tool = scripted_tool("lookup", [OK_RESULT, OK_RESULT, OK_RESULT])
    agent_llm = ScriptedLLM(
        [
            tool_calls(call("c1", "lookup", "{}")),
            tool_calls(call("c2", "lookup", "{}")),
            text("final after limit"),
        ]
    )
    extractor_llm = ScriptedLLM([extraction_payload(contact_name="Rahul")])
    result = run(
        make_with_extraction(agent_llm, extractor_llm, knowledge, store, tools=registry_with(tool)),
        "I'm Rahul",
    )

    assert result.reply_text == "final after limit"
    assert result.diagnostics.tool_rounds == AGENT_MAX_TOOL_ROUNDS
    assert result.diagnostics.tool_round_limit_reached is True
    assert result.diagnostics.loop_exit == "tool_round_limit"
    assert len(result.tool_calls) == 2
    assert len(extractor_llm.calls) == 1


def test_final_text_only_call_behaviour_remains_intact_with_extraction(knowledge, store):
    tool = scripted_tool("lookup", [OK_RESULT, OK_RESULT])
    agent_llm = ScriptedLLM(
        [
            tool_calls(call("c1", "lookup", "{}")),
            tool_calls(call("c2", "lookup", "{}")),
            text("final"),
        ]
    )
    result = run(
        make_with_extraction(agent_llm, ScriptedLLM([extraction_payload()]), knowledge, store, tools=registry_with(tool)),
        "hi",
    )

    assert result.diagnostics.final_call_tools_omitted is True
    final_kwargs = agent_llm.calls[-1]["kwargs"]
    assert "tools" not in final_kwargs
    assert "tool_choice" not in final_kwargs
    assert all(c["kwargs"].get("tool_choice") != "none" for c in agent_llm.calls)
    assert all("tools" in c["kwargs"] for c in agent_llm.calls[:-1])


def test_safe_fallback_on_llm_failure_still_works_and_keeps_extracted_lead(knowledge, store):
    agent_llm = ScriptedLLM([LLMProviderError("Groq down")])
    extractor_llm = ScriptedLLM([extraction_payload(contact_name="Rahul")])
    result = run(make_with_extraction(agent_llm, extractor_llm, knowledge, store), "I'm Rahul")

    assert result.reply_text == SAFE_FALLBACK_REPLY
    assert result.diagnostics.fallback_used is True
    assert result.diagnostics.fallback_reason == "llm_error"
    assert result.diagnostics.error_type == "LLMProviderError"
    # Extraction succeeded independently and its state is still persisted.
    assert result.diagnostics.extraction_success is True
    assert store.get(SENDER).lead.contact_name == "Rahul"
    assert [m.role for m in store.get(SENDER).history] == ["user", "assistant"]


# ---------------------------------------------------------------------------
# 27. No network activity with extraction enabled
# ---------------------------------------------------------------------------


def test_no_network_activity_with_extraction_enabled(knowledge, store, monkeypatch):
    loopback = {"127.0.0.1", "::1", "localhost"}
    original_connect = socket.socket.connect
    original_getaddrinfo = socket.getaddrinfo

    def _guarded_connect(sock, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if host not in loopback:
            raise AssertionError(f"network access attempted during orchestrator turn: {host!r}")
        return original_connect(sock, address, *args, **kwargs)

    def _guarded_getaddrinfo(host, *args, **kwargs):
        if host not in loopback:
            raise AssertionError(f"DNS lookup attempted during orchestrator turn: {host!r}")
        return original_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", _guarded_getaddrinfo)

    agent_llm = ScriptedLLM([tool_calls(call("call_1")), text("done")])
    extractor_llm = ScriptedLLM([extraction_payload(**WHOLESALE_FIELDS_RAHUL)])
    result = run(make_with_extraction(agent_llm, extractor_llm, knowledge, store), "Rahul from Bean House, Ethiopia?")

    assert result.reply_text == "done"
    assert result.tool_calls[0].status == "ok"
    assert result.state_snapshot.qualification == QualificationState.QUALIFIED
    assert agent_llm.exhausted and extractor_llm.exhausted


def test_extraction_integration_is_not_wired_into_app_main():
    import app.main as main_module

    main_source = inspect.getsource(main_module)
    assert "LeadExtractor" not in main_source
    assert "extraction" not in main_source
