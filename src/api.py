# coding: utf-8
from datetime import datetime
import json
import re
from typing import Dict, Any

import httpx
from fastapi import (
    APIRouter,
    Body,
    Header,
    Request,
)
from mistralai import Mistral
from starlette.responses import StreamingResponse

main_api = ""
second_api = ""

# noinspection SpellCheckingInspection
api_key = ""
model = "mistral-small-latest"

mistral_client = Mistral(api_key=api_key)

router = APIRouter()
timeout = httpx.Timeout(600.0, read=None)


async def proxy_streaming_api(
    method: str, full_url: str, headers: dict[str, str], body: str
):
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            method, full_url, headers=headers, content=body
        ) as response:
            async for chunk in response.aiter_bytes():
                yield chunk


def text_to_chat_completion(prompt):
    messages = []
    pattern = r"(?P<System>\[INST](.*)\[/INST] Understood\.</s>)|(?P<User>(?<=\[INST])\s([\w|\s]*:)\s(.*?)(?=\[/INST]))|(?P<Assistant>(?<=\[/INST])\s([\w|\s]*:)\s(.*?)(?=</s>))"
    matches = re.finditer(pattern, prompt, re.MULTILINE | re.DOTALL)

    for matchNum, match in enumerate(matches):
        groups = match.groupdict()
        if groups["System"]:
            messages.append(
                {
                    "role": "system",
                    "content": groups["System"]
                    .replace("[INST]", "")
                    .replace("[/INST] Understood.</s>", "")
                    .strip(),
                }
            )
        elif groups["User"]:
            messages.append(
                {"role": "user", "content": groups["User"].replace("\\", "").strip()}
            )
        elif groups["Assistant"]:
            messages.append(
                {
                    "role": "assistant",
                    "content": groups["Assistant"].replace("\\", "").strip(),
                }
            )

    return messages


@router.get(
    "/health",
    tags=["default"],
    summary="Healthcheck",
)
async def healthcheck_health_get(request: Request) -> Dict[str, Any]:
    """Get the current service health status"""
    async with httpx.AsyncClient(timeout=timeout) as client:
        full_url = f"{main_api}{request.url.path}"
        response = await client.get(full_url)
        data = response.json()
        return data


# noinspection DuplicatedCode
@router.get(
    "/v1/model",
    tags=["default"],
    summary="Current Model",
)
async def current_model_v1_model_get(
    request: Request,
    x_api_key: str = Header(None, description=""),
    authorization: str = Header(None, description=""),
) -> Dict[str, Any]:
    """Returns the currently loaded model."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        headers = {"x-api-key": x_api_key, "Authorization": authorization}
        full_url = f"{main_api}{request.url.path}"
        response = await client.get(full_url, headers=headers)
        data = response.json()
        return data


# noinspection DuplicatedCode
@router.get(
    "/v1/model/list",
    tags=["default"],
    summary="List Models",
)
async def list_models_v1_model_list_get(
    request: Request,
    x_api_key: str = Header(None, description=""),
    authorization: str = Header(None, description=""),
) -> Dict[str, Any]:
    """Lists all models in the model directory.  Requires an admin key to see all models."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        headers = {"x-api-key": x_api_key, "Authorization": authorization}
        full_url = f"{main_api}{request.url.path}"
        response = await client.get(full_url, headers=headers)
        data = response.json()
        return data


# noinspection DuplicatedCode
@router.get(
    "/v1/models",
    tags=["default"],
    summary="List Models",
)
async def list_models_v1_models_get(
    request: Request,
    x_api_key: str = Header(None, description=""),
    authorization: str = Header(None, description=""),
) -> Dict[str, Any]:
    """Lists all models in the model directory.  Requires an admin key to see all models."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        headers = {"x-api-key": x_api_key, "Authorization": authorization}
        full_url = f"{main_api}{request.url.path}"
        response = await client.get(full_url, headers=headers)
        data = response.json()
        return data


@router.post(
    "/v1/completions",
    tags=["default"],
    summary="Completion Request",
    response_model=None,
)
async def completion_request_v1_completions_post(
    request: Request,
    completion_request: Dict = Body(None, description=""),
    x_api_key: str = Header(None, description=""),
    authorization: str = Header(None, description=""),
) -> StreamingResponse | Dict[str, Any]:
    """Generates a completion from a prompt.  If stream is true, this returns an SSE stream."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        headers = {"x-api-key": x_api_key, "Authorization": authorization}
        full_primary_url = f"{main_api}{request.url.path}"
        # full_second_url = f"{second_api}{request.url.path}"

        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        stream = completion_request.get("stream", False)
        prompt = completion_request.get("prompt", "")
        chat = text_to_chat_completion(prompt)

        username = "User"
        # noinspection PyBroadException
        try:
            user_regex = re.compile("\[INST]\s*([^\s:]+):\s*[^\[\]]*\[/INST](?!</s>)")
            user_match = user_regex.findall(prompt, re.MULTILINE)
            username = user_match[0]
        except:
            print("No username found, defaulting to User")

        end_index = prompt.rfind("]")
        end_index2 = prompt.rfind("[")
        character_raw = prompt[end_index + 1 :]
        character = character_raw.replace(":", "").strip()

        chat.append(
            {
                "role": "user",
                "content": f"{username}: YOUR COT PROMPT HERE (you can use username and character to insert names)",
            }
        )
        chat_response = mistral_client.chat.complete(model=model, messages=chat)

        spacing = "[/INST] "
        if prompt[end_index2 - 1 : end_index2] != ">":
            spacing = ""
        result = chat_response.choices[0].message.content
        expanded = (
            prompt[:end_index2]
            + spacing
            + "Thoughts: "
            + result
            + "</s>[/INST]"
            + character_raw
        )

        filename = f"../logs/thoughts_{current_datetime}.log"
        with open(filename, "w") as file:
            file.write(result)

        completion_request["prompt"] = expanded
        completion_request["stream"] = stream
        json_data = json.dumps(completion_request)

        filename = f"../logs/full_{current_datetime}.log"
        with open(filename, "w") as file:
            file.write(json_data)

        if stream:
            return StreamingResponse(
                proxy_streaming_api("POST", full_primary_url, headers, json_data),
                media_type="application/json",
            )
        else:
            response = await client.post(
                full_primary_url, headers=headers, content=json_data
            )
            data = response.json()
            return data


# noinspection DuplicatedCode
@router.post(
    "/v1/token/encode",
    tags=["default"],
    summary="Encode Tokens",
)
async def encode_tokens_v1_token_encode_post(
    request: Request,
    token_encode_request: Dict[str, Any] = Body(None, description=""),
    x_api_key: str = Header(None, description=""),
    authorization: str = Header(None, description=""),
) -> Dict[str, Any]:
    """Encodes a string or chat completion messages into tokens."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        headers = {"x-api-key": x_api_key, "Authorization": authorization}
        full_url = f"{main_api}{request.url.path}"
        json_data = json.dumps(token_encode_request)

        response = await client.post(full_url, headers=headers, content=json_data)
        data = response.json()
        return data


# noinspection DuplicatedCode
@router.post(
    "/v1/token/decode",
    tags=["default"],
    summary="Decode Tokens",
)
async def decode_tokens_v1_token_decode_post(
    request: Request,
    token_decode_request: Dict[str, Any] = Body(None, description=""),
    x_api_key: str = Header(None, description=""),
    authorization: str = Header(None, description=""),
) -> Dict[str, Any]:
    """Decodes tokens into a string."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        headers = {"x-api-key": x_api_key, "Authorization": authorization}
        full_url = f"{main_api}{request.url.path}"
        json_data = json.dumps(token_decode_request)

        response = await client.post(full_url, headers=headers, content=json_data)
        data = response.json()
        return data


# noinspection DuplicatedCode
@router.get(
    "/{path:path}",
    tags=["default"],
    summary="Decode Tokens",
)
async def _reverse_proxy_get(request: Request):
    async with httpx.AsyncClient(timeout=timeout) as client:
        headers = {
            "x-api-key": request.headers["x_api_key"],
            "Authorization": request.headers["authorization"],
        }
        full_url = f"{main_api}{request.url.path}"
        json_data = json.dumps(request.body())

        response = await client.post(full_url, headers=headers, content=json_data)
        data = response.json()
        return data


# noinspection DuplicatedCode
@router.post(
    "/{path:path}",
    tags=["default"],
    summary="Decode Tokens",
)
async def _reverse_proxy_post(request: Request):
    async with httpx.AsyncClient(timeout=timeout) as client:
        headers = {
            "x-api-key": request.headers["x_api_key"],
            "Authorization": request.headers["authorization"],
        }
        full_url = f"{main_api}{request.url.path}"
        json_data = json.dumps(request.body())

        response = await client.post(full_url, headers=headers, content=json_data)
        data = response.json()
        return data
