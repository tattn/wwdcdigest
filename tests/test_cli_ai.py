"""Tests for external AI CLI helpers."""

import json
import shlex
import sys

import pytest

from wwdcdigest.cli_ai import complete_json_with_cli, complete_text_with_cli
from wwdcdigest.models import AIConfig, OpenAIResponse


@pytest.mark.asyncio
async def test_complete_json_with_custom_command():
    """Test parsing structured output from a custom AI command."""
    code = "import json; print(json.dumps({'summary': 'ok', 'key_points': ['one']}))"
    config = AIConfig(
        provider="command",
        command=f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}",
    )

    response = await complete_json_with_cli("prompt", config, OpenAIResponse)

    assert isinstance(response, OpenAIResponse)
    assert response.summary == "ok"
    assert response.key_points == ["one"]


@pytest.mark.asyncio
async def test_complete_json_with_result_wrapper():
    """Test parsing wrapped JSON output from CLIs that return metadata."""
    result = json.dumps({"summary": "wrapped", "key_points": ["one"]})
    code = f"import json; print(json.dumps({{'result': {result!r}}}))"
    config = AIConfig(
        provider="command",
        command=f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}",
    )

    response = await complete_json_with_cli("prompt", config, OpenAIResponse)

    assert isinstance(response, OpenAIResponse)
    assert response.summary == "wrapped"


@pytest.mark.asyncio
async def test_complete_text_with_custom_command():
    """Test plain text output from a custom AI command."""
    code = "print('translated')"
    config = AIConfig(
        provider="command",
        command=f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}",
    )

    output = await complete_text_with_cli("prompt", config)

    assert output == "translated"
