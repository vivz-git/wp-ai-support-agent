"""Minimal tool registry.

Holds ``ToolSpec`` entries (name, description, input schema, handler) so a
future orchestrator can:

- look up a tool by name to execute it,
- list all registered tools' specs in a format ready to hand to an LLM's
  native function-calling API (Groq's OpenAI-compatible ``tools`` parameter).

This module does not implement an agent loop, does not call any LLM, and
does not know anything about Groq's HTTP client — it only knows how to
describe and execute tools.
"""

import json
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Type, Union

from pydantic import BaseModel


@dataclass(frozen=True)
class ToolSpec:
    """Describes one callable tool.

    Attributes:
        name: Stable tool identifier, used both for lookup and as the
            function name in the serialized tool-calling schema.
        description: Short, model-facing description of what the tool does.
        input_model: The Pydantic model validating this tool's arguments.
        handler: A pure function taking a raw ``dict`` of arguments and
            returning a JSON-serializable ``dict`` result. Must never raise.
    """

    name: str
    description: str
    input_model: Type[BaseModel]
    handler: Callable[[dict], dict]

    def to_groq_schema(self) -> dict:
        """Serialize to the OpenAI/Groq native function-calling tool format.

        Shape: ``{"type": "function", "function": {"name", "description",
        "parameters"}}``, where ``parameters`` is this tool's input model's
        JSON Schema. No secrets or implementation details are included —
        only the public argument contract.
        """
        schema = self.input_model.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


class ToolRegistry:
    """A small in-memory collection of ``ToolSpec`` entries."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def list_specs(self) -> List[dict]:
        """Return every registered tool's Groq-native schema, name-sorted."""
        return [self._tools[name].to_groq_schema() for name in sorted(self._tools)]

    def execute(self, name: str, arguments: Union[str, dict, None]) -> dict:
        """Execute a registered tool by name with raw arguments.

        Accepts either a ``dict`` (already-decoded) or a ``str`` (a raw JSON
        arguments string, as Groq's tool-calling API sends them). Never
        raises: malformed JSON, an unknown tool name, or a wrong argument
        type all produce a structured ``invalid_input``-shaped error dict
        rather than an exception, matching the contract each tool handler
        itself follows.
        """
        spec = self.get(name)
        if spec is None:
            return {
                "status": "invalid_input",
                "result_count": 0,
                "results": [],
                "suggestions": [],
                "error": {
                    "code": "unknown_tool",
                    "message": f"No tool registered with name '{name}'.",
                    "fields": [],
                },
            }

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return {
                    "status": "invalid_input",
                    "result_count": 0,
                    "results": [],
                    "suggestions": [],
                    "error": {
                        "code": "malformed_arguments_json",
                        "message": "Tool arguments were not valid JSON.",
                        "fields": [],
                    },
                }

        if arguments is None:
            arguments = {}

        return spec.handler(arguments)
