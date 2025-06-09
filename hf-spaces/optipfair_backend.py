from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # ← NUEVO: Para CORS
from routers.visualize import router as visualize_router

# Create FastAPI app with HF Spaces configuration
app = FastAPI(
    title="OptiPFair API",
    description="Backend API for OptiPFair bias visualization",
    version="1.0.0",
)

# ← NUEVO: CORS middleware for HF Spaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite requests desde cualquier origen
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos HTTP
    allow_headers=["*"],  # Permite todos los headers
)


# Existing endpoints
@app.get("/ping")
async def ping():
    return {"message": "pong"}


app.include_router(visualize_router)

import logging

logging.basicConfig(level=logging.INFO)
