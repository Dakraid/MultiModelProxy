import json
from typing import Any, Dict

import httpx
from fastapi import Request

from src import Variables


def get_headers(x_api_key: str, authorization: str):
    headers = {}
    if x_api_key != "" and authorization != "":
        headers = {"x-api-key": x_api_key, "Authorization": authorization}
    elif x_api_key != "" and authorization == "":
        headers = {"x-api-key": x_api_key}
    elif x_api_key == "" and authorization != "":
        headers = {"Authorization": authorization}
    return headers


async def proxy_streaming_api(variables: Variables, method: str, full_url: str, headers: dict[str, str], body: str):
    async with httpx.AsyncClient(timeout=variables.timeout) as client:
        async with client.stream(method, full_url, headers=headers, content=body) as response:
            async for chunk in response.aiter_bytes():
                yield chunk


async def proxy_generic_get(variables: Variables, request: Request):
    async with httpx.AsyncClient(timeout=variables.timeout) as client:
        full_url = f"{variables.primary_url}{request.url.path}"
        response = await client.get(full_url)
        data = response.json()
        return data


async def proxy_generic_get_authenticated(variables: Variables, request: Request, x_api_key: str, authorization: str):
    async with httpx.AsyncClient(timeout=variables.timeout) as client:
        headers = get_headers(x_api_key, authorization)
        full_url = f"{variables.primary_url}{request.url.path}"
        response = await client.get(full_url, headers=headers)
        data = response.json()
        return data


async def proxy_generic_post_authenticated_content(
    variables: Variables,
    request: Request,
    x_api_key: str,
    authorization: str,
    body: str | Dict[str, Any],
):
    async with httpx.AsyncClient(timeout=variables.timeout) as client:
        headers = get_headers(x_api_key, authorization)

        full_url = f"{variables.primary_url}{request.url.path}"
        json_data = json.dumps(body)

        response = await client.post(full_url, headers=headers, content=json_data)
        data = response.json()
        return data
