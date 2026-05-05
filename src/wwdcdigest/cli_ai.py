"""Utilities for calling external AI CLIs."""

import asyncio
import json
import logging
import os
import shlex
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from typing import Any

from pydantic import BaseModel

from .models import AIConfig

logger = logging.getLogger("wwdcdigest")


class ExternalAIError(Exception):
    """Base class for external AI CLI errors."""

    pass


def _schema_for(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    schema["additionalProperties"] = False
    return schema


def _extract_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ExternalAIError("External AI output did not contain JSON") from None
        value = json.loads(text[start : end + 1])

    if not isinstance(value, dict):
        raise ExternalAIError("External AI output JSON was not an object")
    return value


async def _run_command(
    args: Sequence[str],
    timeout_seconds: int,
) -> str:
    logger.debug("Running external AI command: %s", shlex.join(args))
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=timeout_seconds
        )
    except TimeoutError as e:
        process.kill()
        await process.communicate()
        raise ExternalAIError("External AI command timed out") from e

    stdout_text = stdout.decode("utf-8", errors="replace").strip()
    stderr_text = stderr.decode("utf-8", errors="replace").strip()

    if process.returncode != 0:
        detail = stderr_text or stdout_text
        raise ExternalAIError(f"External AI command failed: {detail}")

    return stdout_text


def _build_codex_args(
    config: AIConfig,
    prompt: str,
    schema_path: str | None = None,
) -> list[str]:
    args = ["codex", "exec", "--sandbox", "read-only"]
    if config.model:
        args.extend(["--model", config.model])
    if schema_path:
        args.extend(["--output-schema", schema_path])
    args.append(prompt)
    return args


def _build_claude_args(
    config: AIConfig,
    prompt: str,
    schema: dict[str, Any] | None = None,
) -> list[str]:
    args = ["claude", "-p"]
    if config.model:
        args.extend(["--model", config.model])
    if schema:
        args.extend(["--output-format", "json", "--json-schema", json.dumps(schema)])
    args.append(prompt)
    return args


def _build_custom_args(config: AIConfig, prompt: str) -> list[str]:
    if not config.command:
        raise ExternalAIError("--ai-command is required when --ai command is used")

    args = shlex.split(config.command)
    if any("{prompt}" in arg for arg in args):
        return [arg.replace("{prompt}", prompt) for arg in args]

    return [*args, prompt]


async def complete_text_with_cli(prompt: str, config: AIConfig) -> str:
    """Run an external AI CLI and return plain text."""
    if config.provider == "codex":
        args = _build_codex_args(config, prompt)
    elif config.provider == "claude":
        args = _build_claude_args(config, prompt)
    elif config.provider == "command":
        args = _build_custom_args(config, prompt)
    else:
        raise ExternalAIError(f"Unsupported external AI provider: {config.provider}")

    return await _run_command(args, config.timeout_seconds)


async def complete_json_with_cli(
    prompt: str,
    config: AIConfig,
    response_model: type[BaseModel],
) -> BaseModel:
    """Run an external AI CLI and parse structured JSON output."""
    schema = _schema_for(response_model)

    if config.provider == "codex":
        fd, schema_path = tempfile.mkstemp(prefix="wwdcdigest_schema_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(schema, f)
            output = await _run_command(
                _build_codex_args(config, prompt, schema_path), config.timeout_seconds
            )
        finally:
            with suppress(OSError):
                os.unlink(schema_path)
    elif config.provider == "claude":
        output = await _run_command(
            _build_claude_args(config, prompt, schema), config.timeout_seconds
        )
    elif config.provider == "command":
        output = await _run_command(
            _build_custom_args(config, prompt), config.timeout_seconds
        )
    else:
        raise ExternalAIError(f"Unsupported external AI provider: {config.provider}")

    data = _extract_json_object(output)
    if "structured_output" in data and isinstance(data["structured_output"], dict):
        data = data["structured_output"]
    elif "result" in data and isinstance(data["result"], str):
        data = _extract_json_object(data["result"])

    return response_model.model_validate(data)
