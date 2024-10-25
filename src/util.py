import json

import httpx
from httpx import Timeout
from fastapi import (
    Request,
)
from typing import Dict, Any

from src.config import Config


class Utility:
    def __init__(self, config: Config, timeout: Timeout):
        self.config = config
        self.timeout = timeout

    @staticmethod
    def get_headers(x_api_key: str, authorization: str):
        headers = {}
        if x_api_key != "" and authorization != "":
            headers = {"x-api-key": x_api_key, "Authorization": authorization}
        elif x_api_key != "" and authorization == "":
            headers = {"x-api-key": x_api_key}
        elif x_api_key == "" and authorization != "":
            headers = {"Authorization": authorization}
        return headers

    async def proxy_streaming_api(
        self, method: str, full_url: str, headers: dict[str, str], body: str
    ):
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                method, full_url, headers=headers, content=body
            ) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk

    async def proxy_generic_get(self, request: Request):
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            full_url = (
                f"{self.config.configuration.inference.primary_url}{request.url.path}"
            )
            response = await client.get(full_url)
            data = response.json()
            return data

    async def proxy_generic_get_authenticated(
        self, request: Request, x_api_key: str, authorization: str
    ):
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            headers = self.get_headers(x_api_key, authorization)
            full_url = (
                f"{self.config.configuration.inference.primary_url}{request.url.path}"
            )
            response = await client.get(full_url, headers=headers)
            data = response.json()
            return data

    async def proxy_generic_post_authenticated_content(
        self,
        request: Request,
        x_api_key: str,
        authorization: str,
        body: str | Dict[str, Any],
    ):
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            headers = self.get_headers(x_api_key, authorization)

            full_url = (
                f"{self.config.configuration.inference.primary_url}{request.url.path}"
            )
            json_data = json.dumps(body)

            response = await client.post(full_url, headers=headers, content=json_data)
            data = response.json()
            return data
