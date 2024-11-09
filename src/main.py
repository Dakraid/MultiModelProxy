# coding: utf-8
import json
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any, Dict

import httpx
from fastapi import APIRouter, Body, FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from src import Config, MistralInference, OpenRouterInference, TabbyApiInference, Variables
from src.utility import database
from src.utility.logger import setup_logger

# Initialize core components
config = Config.from_yaml("config.yaml")
timeout = httpx.Timeout(config.configuration.http_client.timeout, read=None)
client = httpx.AsyncClient(timeout=timeout)
router = APIRouter()
variables = Variables(config, timeout)

# Initialize inference handler based on config
INFERENCE_HANDLERS = {"mistral": MistralInference, "tabbyapi": TabbyApiInference, "openrouter": OpenRouterInference}

handler = config.configuration.inference.secondary_api_handler.lower()
if handler not in INFERENCE_HANDLERS:
    raise ValueError(f"Invalid secondary API handler: {handler}")

inference = INFERENCE_HANDLERS[handler](config)


async def make_request(method: str, url: str, headers: dict = None, body: Any = None, stream: bool = False):
    """Unified request handler"""
    kwargs = {"headers": headers or {}}
    if body:
        kwargs["content"] = json.dumps(body) if isinstance(body, (dict, list)) else body

    if not stream:
        response = await getattr(client, method.lower())(url, **kwargs)
        # noinspection PyBroadException
        try:
            return response.json()
        except:  # noqa: E722
            return response.text
    else:
        return client.stream(method, url, **kwargs)


async def stream_response(response):
    """Helper function to stream response data"""
    async with response as streaming_response:
        async for chunk in streaming_response.aiter_bytes():
            yield chunk


@lru_cache(maxsize=1)
async def get_cached_thought():
    return await database.get_latest_log()


@lru_cache(maxsize=1)
async def get_cached_thoughts():
    return await database.get_logs()


@router.get("/v1/thought")
async def get_last_thought():
    """Get last thought"""
    if thought := await get_cached_thought():
        return {"content": thought[0].response, "tokens": thought[0].tokens, "timestamp": thought[0].timestamp}
    return {"content": "No thoughts available."}


@router.get("/v1/thoughts")
async def get_thoughts(request: Request, authorization: str = Header(None)):
    """Get all thoughts"""
    if authorization != config.configuration.general.dev_api_key:
        raise HTTPException(status_code=401, detail="Missing or invalid authorization.")

    if thoughts := await get_cached_thoughts():
        return [{"content": t.response, "tokens": t.tokens, "timestamp": t.timestamp} for t in thoughts]
    return {"content": "No thoughts available."}


@router.post("/v1/completions")
async def completion_request_handler(
    request: Request,
    completion_request: Dict = Body(None),
    x_api_key: str = Header(None),
    authorization: str = Header(None),
):
    """Handle completion requests"""
    # Check TabbyAPI health
    health = await make_request("GET", f"{config.configuration.inference.primary_url}/health")
    if not health:
        raise HTTPException(status_code=502, detail="The TabbyAPI instance is unavailable")

    # Generate Chain of Thought
    message, expanded, last_message = await inference.cot_completion(
        completion_request=completion_request, stored_last_message=variables.last_message
    )

    if last_message != variables.last_message:
        variables.last_expanded = expanded
        variables.last_message = last_message

    completion_request["prompt"] = variables.last_expanded

    url = f"{config.configuration.inference.primary_url}{request.url.path}"
    headers = {"x-api-key": x_api_key, "Authorization": authorization}

    if completion_request.get("stream"):
        response = await make_request("POST", url, headers, completion_request, stream=True)
        return StreamingResponse(stream_response(response), media_type="application/json")

    return await make_request("POST", url, headers, completion_request)


@router.post("/v1/token/{action}")
async def token_endpoint_handler(
    action: str,
    request: Request,
    body: Dict[str, Any] = Body(None),
    x_api_key: str = Header(None),
    authorization: str = Header(None),
):
    """Handle token encode/decode requests"""
    url = f"{variables.primary_url}{request.url.path}"
    headers = {"x-api-key": x_api_key, "Authorization": authorization}
    return await make_request("POST", url, headers, body)


@router.api_route("/{path:path}", methods=["GET", "POST"])
async def proxy_endpoint(request: Request, path: str, x_api_key: str = Header(None), authorization: str = Header(None)):
    """Generic proxy endpoint"""
    url = f"{variables.primary_url}/{path}"
    headers = {"x-api-key": x_api_key, "Authorization": authorization}

    if request.method == "GET":
        if path == "v1/model/list":
            health = await make_request("GET", f"{config.configuration.inference.primary_url}/health")
            if not health:
                raise HTTPException(status_code=502, detail="The TabbyAPI instance is unavailable")
        result = await make_request("GET", url, headers)
        return result

    body = await request.body()
    return await make_request("POST", url, headers, body)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager"""
    await database.handler()
    yield
    await client.aclose()


# Initialize FastAPI app
app = FastAPI(
    title="MultiModelProxy", description="API proxy for multiple LLM endpoints", version="0.0.1", lifespan=lifespan
)

app.include_router(router)


@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    """Global exception handler"""
    try:
        return await call_next(request)
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


if __name__ == "__main__":
    import uvicorn
    import uvloop
    import asyncio

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    setup_logger()
    logger.info("Starting MMP")

    uvicorn.run("src.main:app", host="127.0.0.1", port=5000, reload=True, reload_excludes="config.py", loop="uvloop")
