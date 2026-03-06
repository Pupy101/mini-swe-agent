import json
import logging
import os
import time
import copy
from typing import Any, Literal

from pydantic import BaseModel
from shooters.devices import ThreadShooter

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
    FormatError,
    StrictUndefined,
    Template,
)
from minisweagent.models.utils.anthropic_utils import _reorder_anthropic_thinking_blocks
from minisweagent.models.utils.cache_control import set_cache_control
from minisweagent.models.utils.openai_multimodal import expand_multimodal_content
from minisweagent.models.utils.retry import retry

logger = logging.getLogger("giga")


class GigaModelConfig(BaseModel):
    model_name: str
    model_kwargs: dict[str, Any] = {}
    set_cache_control: Literal["default_end"] | None = None
    """Set explicit cache control markers, for example for Anthropic models"""
    cost_tracking: Literal["default", "ignore_errors"] = os.getenv(
        "MSWEA_COST_TRACKING", "default"
    )
    """Cost tracking mode for this model. Can be "default" or "ignore_errors" (ignore errors/missing cost info)"""
    format_error_template: str = "{{ error }}"
    """Template used when the LM's output is not in the expected format."""
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    """Template used to render the observation after executing an action."""
    multimodal_regex: str = ""
    """Regex to extract multimodal content. Empty string disables multimodal processing."""


class GigaAPIError(Exception):
    """Custom exception for OpenRouter API errors."""


class GigaAuthenticationError(Exception):
    """Custom exception for OpenRouter authentication errors."""


class GigaRateLimitError(Exception):
    """Custom exception for OpenRouter rate limit errors."""


class GigaModel:
    abort_exceptions: list[type[Exception]] = []

    def __init__(self, **kwargs):
        self.config = GigaModelConfig(**kwargs)
        self.shooter = ThreadShooter()

    def _query(self, messages: list[dict[str, str]], **kwargs):
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "functions": [BASH_TOOL["function"]],
            "function_call": {"name": "bash"},
            **(self.config.model_kwargs | kwargs),
        }

        payload_ = copy.deepcopy(payload)
        for key in ["functions", "function_call"]:
            payload_.pop(key)
        message_ = payload_["messages"][-2]
        if "functions_state_id" in message_:
            message_.pop("functions_state_id")
        payload_["messages"] = [message_]
        if message_["role"] != "system":
            print(payload_)

        try:
            response = self.shooter.chat(payload)
            if response is None:
                raise GigaAPIError("Response is None")
            return response.model_dump(exclude_none=True)
        except Exception as e:
            raise GigaAPIError(f"Request failed: {e}") from e

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        prepared = [{k: v for k, v in msg.items() if k != "extra"} for msg in messages]
        prepared = _reorder_anthropic_thinking_blocks(prepared)
        return set_cache_control(prepared, mode=self.config.set_cache_control)

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = self._query(
                    self._prepare_messages_for_api(messages), **kwargs
                )
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        message = dict(response["choices"][0]["message"])
        message["extra"] = {
            "actions": self._parse_actions(response),
            "response": response,
            **cost_output,
            "timestamp": time.time(),
        }
        return message

    def _calculate_cost(self, response) -> dict[str, float]:
        return {"cost": 0.012}

    def _parse_actions(self, response: dict) -> list[dict]:
        """Parse function call from the response. Raises FormatError if unknown tool."""

        function_call = response["choices"][0]["message"].get("function_call")
        function_calls = []
        if function_call is not None:
            function_calls = [{"function": function_call}]
        assert len(function_calls) <= 1, "Only one function call is supported"
        tool_calls = [_DictToObj(_) for _ in function_calls]
        return parse_toolcall_actions(
            tool_calls, format_error_template=self.config.format_error_template
        )

    def format_message(self, **kwargs) -> dict:
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs into tool result messages."""
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }


class _DictToObj:
    """Simple wrapper to convert dict to object with attribute access."""

    def __init__(self, d: dict):
        self._d = d
        self.id = d.get("id")
        self.function = _DictToObj(d.get("function", {})) if "function" in d else None
        self.name = d.get("name")
        self.arguments = d.get("arguments")

    def __repr__(self) -> str:
        return str(
            {
                "id": self.id,
                "function": self.function,
                "name": self.name,
                "arguments": self.arguments,
            }
        )


def format_toolcall_observation_messages(
    *,
    actions: list[dict],
    outputs: list[dict],
    observation_template: str,
    template_vars: dict | None = None,
    multimodal_regex: str = "",
) -> list[dict]:
    """Format execution outputs into tool result messages."""
    not_executed = {
        "output": "",
        "returncode": -1,
        "exception_info": "action was not executed",
    }
    padded_outputs = outputs + [not_executed] * (len(actions) - len(outputs))
    results = []
    for action, output in zip(actions, padded_outputs):
        content = json.dumps({"returncode": output['returncode'], "output": output.get('output')}, ensure_ascii=False)
        msg = {
            "content": content,
            "extra": {
                "raw_output": output.get("output", ""),
                "returncode": output.get("returncode"),
                "timestamp": time.time(),
                "exception_info": output.get("exception_info"),
                **output.get("extra", {}),
            },
        }
        msg["role"] = "function"  # human issued commands
        if multimodal_regex:
            msg = expand_multimodal_content(msg, pattern=multimodal_regex)
        results.append(msg)
    return results


def parse_toolcall_actions(
    tool_calls: list, *, format_error_template: str
) -> list[dict]:
    """Parse tool calls from the response. Raises FormatError if unknown tool or invalid args."""
    if not tool_calls:
        raise FormatError(
            {
                "role": "user",
                "content": Template(
                    format_error_template, undefined=StrictUndefined
                ).render(
                    error="No tool calls found in the response. Every response MUST include at least one tool call.",
                    actions=[],
                ),
                "extra": {"interrupt_type": "FormatError"},
            }
        )
    actions = []
    for tool_call in tool_calls:
        error_msg = ""
        args = {}
        try:
            args = tool_call.function.arguments
        except Exception as e:
            error_msg = f"Error parsing tool call arguments: {e}."
        if tool_call.function.name != "bash":
            error_msg += f"Unknown tool '{tool_call.function.name}'."
        if not isinstance(args, dict) or "command" not in args:
            error_msg += "Missing 'command' argument in bash tool call."
        if error_msg:
            raise FormatError(
                {
                    "role": "user",
                    "content": Template(
                        format_error_template, undefined=StrictUndefined
                    ).render(actions=[], error=error_msg.strip()),
                    "extra": {"interrupt_type": "FormatError"},
                }
            )
        actions.append({"command": args["command"], "tool_call_id": tool_call.id})
    return actions
