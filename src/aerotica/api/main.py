"""AEROTICA FastAPI Application."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import Dict, Any
import time

from aerotica import __version__
from aerotica.api.routes import router


# Create FastAPI app
app = FastAPI(
    title="AEROTICA API",
    description="Atmospheric Kinetic Energy Mapping Framework",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "AEROTICA",
        "version": __version__,
        "status": "operational",
        "documentation": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": __version__
    }


@app.get("/metrics")
async def metrics():
    """Metrics endpoint."""
    import psutil
    import torch
    
    metrics = {
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        },
        "model": {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "cuda_available": torch.cuda.is_available()
        }
    }
    
    if torch.cuda.is_available():
        metrics["cuda"] = {
            "device_count": torch.cuda.device_count(),
            "memory_allocated": torch.cuda.memory_allocated() / 1e9,
            "memory_cached": torch.cuda.memory_reserved() / 1e9
        }
    
    return metrics


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Generic exception handler."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "status_code": 500
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
