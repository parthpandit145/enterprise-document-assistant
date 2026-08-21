"""The Anthropic provider, exercised against a mocked SDK client.

Verifies the request shape and the error handling without spending tokens or
needing a key in CI.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anthropic
import pytest

from docassist.config import load_settings
from docassist.llm import AnthropicLLM, LLMError


def _response(text="Employees get 30 days [1].", stop_reason="end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        model="claude-opus-5",
        stop_reason=stop_reason,
        stop_details=None,
        usage=SimpleNamespace(input_tokens=1200, output_tokens=80),
    )


def _llm(client):
    with patch("anthropic.Anthropic", return_value=client):
        return AnthropicLLM(load_settings())


def test_request_shape():
    client = MagicMock()
    client.messages.create.return_value = _response()

    result = _llm(client).generate("SYSTEM RULES", "CONTEXT + QUESTION")

    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-opus-5"
    assert kwargs["system"] == "SYSTEM RULES"
    assert kwargs["messages"] == [{"role": "user", "content": "CONTEXT + QUESTION"}]
    assert kwargs["output_config"] == {"effort": "medium"}
    assert kwargs["max_tokens"] == 4096
    assert result.text == "Employees get 30 days [1]."
    assert result.usage == {"input_tokens": 1200, "output_tokens": 80}


def test_only_text_blocks_are_kept():
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="reasoning that is not the answer"),
            SimpleNamespace(type="text", text="The answer [1]."),
        ],
        model="claude-opus-5",
        stop_reason="end_turn",
        stop_details=None,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )

    assert _llm(client).generate("s", "u").text == "The answer [1]."


def test_refusal_becomes_a_readable_error():
    client = MagicMock()
    response = _response(stop_reason="refusal")
    response.stop_details = SimpleNamespace(category="cyber", explanation="declined")
    client.messages.create.return_value = response

    with pytest.raises(LLMError, match="declined"):
        _llm(client).generate("s", "u")


def test_missing_credentials_produce_actionable_guidance():
    client = MagicMock()
    client.messages.create.side_effect = anthropic.AuthenticationError(
        "no key", response=MagicMock(status_code=401, headers={}), body=None
    )

    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        _llm(client).generate("s", "u")
