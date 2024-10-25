# coding: utf-8
from datetime import datetime
from typing import Dict, Any

import json
import httpx
from fastapi import (
    APIRouter,
    Body,
    Header,
    Request,
)
from starlette.responses import StreamingResponse

from src.config import Config
from src.inference import MistralInference, TabbyApiInference
from src.util import Utility

config = Config.from_yaml("config.yaml")
timeout = httpx.Timeout(config.configuration.http_client.timeout, read=None)
router = APIRouter()
utility = Utility(config, timeout)

match config.configuration.inference.secondary_api_handler:
    case "Mistral":
        inference = MistralInference(config)
    case "TabbyAPI":
        inference = TabbyApiInference(config)


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
    return await utility.proxy_generic_post_authenticated_content(
        request, x_api_key, authorization, token_encode_request
    )


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
    return await utility.proxy_generic_post_authenticated_content(
        request, x_api_key, authorization, token_decode_request
    )


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
        full_primary_url = (
            f"{config.configuration.inference.primary_url}{request.url.path}"
        )

        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        stream = completion_request.get("stream", False)
        prompt = completion_request.get("prompt", "")

        message, expanded = inference.completion(prompt)

        if config.configuration.general.write_logs:
            filename = (
                config.configuration.general.logs_path
                + f"thoughts_{current_datetime}.log"
            )
            with open(filename, "w") as file:
                file.write(message)

        completion_request["prompt"] = expanded
        completion_request["stream"] = stream
        json_data = json.dumps(completion_request)

        if config.configuration.general.write_logs:
            filename = (
                config.configuration.general.logs_path + f"full_{current_datetime}.log"
            )
            with open(filename, "w") as file:
                file.write(json_data)

        if stream:
            return StreamingResponse(
                utility.proxy_streaming_api(
                    "POST", full_primary_url, headers, json_data
                ),
                media_type="application/json",
            )
        else:
            response = await client.post(
                full_primary_url, headers=headers, content=json_data
            )
            data = response.json()
            return data


@router.get(
    "/{path:path}",
    tags=["default"],
    summary="Generic GET",
)
async def _reverse_proxy_get(request: Request):
    x_api_key = ""
    authorization = ""
    if "x_api_key" in request.headers.keys():
        x_api_key = request.headers.get("x_api_key")
    if "authorization" in request.headers.keys():
        authorization = request.headers.get("authorization")
    if x_api_key != "" or authorization != "":
        return await utility.proxy_generic_get_authenticated(
            request, x_api_key, authorization
        )
    else:
        return await utility.proxy_generic_get(request)


@router.post(
    "/{path:path}",
    tags=["default"],
    summary="Generic POST",
)
async def _reverse_proxy_post(request: Request):
    x_api_key = ""
    authorization = ""
    if "x_api_key" in request.headers.keys():
        x_api_key = request.headers.get("x_api_key")
    if "authorization" in request.headers.keys():
        authorization = request.headers.get("authorization")
    return await utility.proxy_generic_post_authenticated_content(
        request,
        x_api_key,
        authorization,
        json.dumps(request.body()),
    )
