import json
import os
import re
import time
from typing import Any, Dict, Union

import httpx
from loguru import logger
from pydantic import ValidationError

from .base import (
    SingleAssetStructuredGenerationChatEndPoint,
    SingleAssetStructureGenerationFailure,
    SingleAssetStructureOutputResponse,
)


def _repair_json(text: str) -> dict:
    """Extract and repair JSON from MiniMax response text.

    Handles:
    - Text before/after the JSON object
    - Missing closing braces (truncated response)
    - Trailing commas in arrays/objects
    """
    text = text.strip()
    first_brace = text.find("{")
    if first_brace == -1:
        raise json.JSONDecodeError("No JSON object found in response", text, 0)
    json_str = text[first_brace:]

    # Strategy 1: direct parse
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Strategy 2: fix trailing commas
    fixed = re.sub(r",\s*}", "}", json_str)
    fixed = re.sub(r",\s*]", "]", fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        fixed = json_str  # revert

    # Strategy 3: add missing closing braces/brackets by counting
    open_c = fixed.count("{")
    close_c = fixed.count("}")
    open_b = fixed.count("[")
    close_b = fixed.count("]")
    if open_c > close_c or open_b > close_b:
        fixed += "}" * (open_c - close_c) + "]" * (open_b - close_b)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    # Strategy 4: brute-force truncate at each closing brace from end
    for end in range(len(json_str), 0, -1):
        try:
            return json.loads(json_str[:end])
        except json.JSONDecodeError:
            continue

    raise json.JSONDecodeError("Could not extract valid JSON", json_str, 0)


def _flatten_schema_fields(d: dict) -> dict:
    """Convert JSON Schema-style fields back to plain values.

    MiniMax sometimes returns fields as JSON Schema descriptors:
      {"summary_reason": {"description": "text...", "type": "string"}}
    instead of:
      {"summary_reason": "text..."}
    """
    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            # Check if this is a JSON Schema field descriptor
            if "description" in value:
                # {"description": "actual value", "title": "...", "type": "string"}
                result[key] = value["description"]
            elif "items" in value and isinstance(value.get("items"), dict) and "enum" in value["items"]:
                # {"items": {"enum": ["a", "b"]}, "type": "array"}
                result[key] = value["items"]["enum"]
            else:
                result[key] = value
        else:
            result[key] = value
    return result


class SingleAssetOpenAICompatibleStructureGeneration(
    SingleAssetStructuredGenerationChatEndPoint
):
    def __init__(self, chat_config: Dict[str, Any]) -> None:
        logger.trace("CHAT-OpenAI-Compatible chat model initializing")
        self.chat_config = chat_config
        self.chat_model = chat_config["chat_model"]
        self.chat_max_new_token = chat_config["chat_max_new_token"]
        self.chat_model_type = chat_config["chat_model_type"]
        self.chat_system_message = chat_config.get(
            "chat_system_message", "You are a helpful assistant."
        )
        self.chat_request_timeout = chat_config["chat_request_timeout"]
        self.chat_parameters = chat_config["chat_parameters"]
        self.chat_endpoint = chat_config["chat_endpoint"]
        self.api_key_env = chat_config.get("api_key_env", "OPENAI_API_KEY")
        self.chat_request_sleep = chat_config.get("chat_request_sleep")
        self._request_count = 0
        logger.trace(f"CHAT-OpenAI-Compatible endpoint: {self.chat_endpoint}")
        logger.trace(f"CHAT-OpenAI-Compatible model: {self.chat_model}")

    def _maybe_sleep(self) -> None:
        if self.chat_request_sleep is None:
            return
        sleep_time = self.chat_request_sleep.get("sleep_time", 0)
        sleep_every = self.chat_request_sleep.get("sleep_every_count", 1)
        if sleep_time > 0 and self._request_count > 0 and self._request_count % sleep_every == 0:
            logger.trace(f"CHAT-OpenAI-Compatible sleeping for {sleep_time}s")
            time.sleep(sleep_time)

    def __call__(
        self, prompt: str, schema: Any
    ) -> Union[
        SingleAssetStructureGenerationFailure, SingleAssetStructureOutputResponse
    ]:
        self._maybe_sleep()
        self._request_count += 1

        # Build the full prompt with JSON schema instruction
        full_prompt = (
            f"{prompt}\n\n"
            "Please respond with a single JSON object that strictly follows this schema:\n"
            f"{json.dumps(schema, indent=2)}\n"
            "Do not include any text outside the JSON object."
        )

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            logger.error(
                f"CHAT-OpenAI-Compatible API key not found in env var: {self.api_key_env}"
            )
            return SingleAssetStructureGenerationFailure()

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        request_data = {
            "model": self.chat_model,
            "max_tokens": self.chat_max_new_token,
            "messages": [
                {"content": self.chat_system_message, "role": "system"},
                {"content": full_prompt, "role": "user"},
            ],
            **self.chat_parameters,
        }

        try:
            with httpx.Client(timeout=self.chat_request_timeout) as client:
                response = client.post(
                    url=self.chat_endpoint,
                    headers=headers,
                    json=request_data,
                )
        except Exception as e:
            logger.error(f"CHAT-OpenAI-Compatible request failed: {e}")
            return SingleAssetStructureGenerationFailure()

        if response.status_code != 200:
            logger.error(
                f"CHAT-OpenAI-Compatible response status code: {response.status_code}"
            )
            try:
                logger.error(f"CHAT-OpenAI-Compatible response: {response.json()}")
            except Exception:
                logger.error(f"CHAT-OpenAI-Compatible response text: {response.text}")
            return SingleAssetStructureGenerationFailure()

        try:
            response_json = response.json()
            content = response_json["choices"][0]["message"]["content"]
            # Try to extract JSON from the content (model may wrap it in markdown or thinking tags)
            content = content.strip()
            # Strip MiniMax thinking tags: <think>...</think>
            think_start = content.find("<think>")
            think_end = content.find("</think>")
            if think_start != -1 and think_end != -1:
                content = content[think_end + len("</think>"):].strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            response_dict = _repair_json(content)
            # MiniMax wraps data in a "properties" key (JSON Schema style)
            if "properties" in response_dict:
                response_dict = response_dict["properties"]
            # MiniMax sometimes returns JSON Schema descriptors instead of plain values
            response_dict = _flatten_schema_fields(response_dict)
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"CHAT-OpenAI-Compatible JSON decode error: {e}")
            try:
                logger.error(f"CHAT-OpenAI-Compatible raw response: {response.json()}")
            except Exception:
                logger.error(f"CHAT-OpenAI-Compatible raw text: {response.text}")
            return SingleAssetStructureGenerationFailure()

        # Normalize memory id fields (string -> int, dedup)
        for key in ["short_memory_ids", "mid_memory_ids", "long_memory_ids", "reflection_memory_ids"]:
            if key in response_dict:
                ids = response_dict[key]
                if isinstance(ids, list):
                    # Convert string ids to int if needed
                    response_dict[key] = list({
                        int(i) if isinstance(i, str) and i.isdigit() else i
                        for i in ids
                    })

        try:
            response_pydantic = SingleAssetStructureOutputResponse(**response_dict)
        except ValidationError as e:
            logger.error(f"CHAT-OpenAI-Compatible pydantic validation error: {e}")
            logger.error(f"CHAT-OpenAI-Compatible response dict: {response_dict}")
            return SingleAssetStructureGenerationFailure()

        return response_pydantic
