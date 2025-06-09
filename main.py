# main.py
from fastapi import FastAPI
from routers.visualize import router as visualize_router

app = FastAPI(debug=True)


# Existing ping endpoint
@app.get("/ping")
async def ping():
    return {"message": "pong"}


# Include our visualization router
app.include_router(visualize_router)

import logging

logging.basicConfig(level=logging.INFO)
