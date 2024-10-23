# coding: utf-8


import uvicorn
from fastapi import FastAPI

from src.api import router

app = FastAPI(
    title="MultiModelProxy",
    description="This docs page is not meant to send requests! Please use a service like Postman or a frontend UI.",
    version="0.1.0",
)

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
