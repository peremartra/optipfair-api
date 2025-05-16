# main.py
from fastapi import FastAPI
from routers.visualize import router as visualize_router

app = FastAPI(debug=True)

# ping endpoint ya existente
@app.get("/ping")
async def ping():
    return {"message": "pong"}

# incluir nuestro router de visualización
app.include_router(visualize_router)

import logging
logging.basicConfig(level=logging.INFO)
