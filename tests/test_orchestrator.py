"""Tests for the deterministic agent orchestrator (app/agent/orchestrator.py).

All dependencies are fakes or scripted: a scripted ``LLMProvider`` that
returns canned ``LLMResponse`` objects (or raises), a real ``ToolRegistry``
holding either the real ``product_lookup`` tool or small scripted tools, the
real (fictional) knowledge base, and an in-memory ``ConversationStore``.
No Groq or WhatsApp network calls are made anywhere in this module.
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

from app.agent.orchestrator import (
    AGENT_MAX_TOOL_ROUNDS,
    SAFE_FALLBACK_REPLY,
    AgentOrchestrator,
    AgentTurnResult,
    ToolCallRecord,
    TurnDiagnostics,
)
from app.agent.state import MAX_CHAT_HISTORY, ConversationState
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
