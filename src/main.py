# coding: utf-8
from fastapi import FastAPI

from src.api import router

app = FastAPI(
    title="MultiModelProxy",
    description="This docs page is not meant to send requests! Please use a service like Postman or a frontend UI.",
    version="0.0.1",
)

app.include_router(router)
