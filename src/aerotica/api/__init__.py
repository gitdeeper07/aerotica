"""AEROTICA API Module."""

from aerotica.api.main import app
from aerotica.api.routes import router

__all__ = [
    'app',
    'router'
]
