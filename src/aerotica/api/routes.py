"""AEROTICA API Routes."""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import numpy as np

from aerotica.ake import AKEComposite, AKEWeights, AKETHRESHOLDS
from aerotica.parameters import KED, TII, VSR, AOD, THD, PGF, HCI, ASI, LRC
from aerotica.pinn import AeroticaPINN
from aerotica.alerts import GustPreAlertEngine


router = APIRouter()


# Pydantic models
class AKERequest(BaseModel):
    """AKE computation request."""
    site_id: str = Field(..., description="Site identifier")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude")
    climate_zone: str = Field("temperate", description="Climate zone")
    site_type: str = Field("unknown", description="Site type")
    parameters: Dict[str, float] = Field(..., description="Parameter scores")


class AKEResponse(BaseModel):
    """AKE computation response."""
    site_id: str
    score: float
    classification: str
    gust_risk: str
    confidence: float
    contributions: Dict[str, Any]


class GustAlertRequest(BaseModel):
    """Gust alert request."""
    site_id: str
    wind_speed: List[float]
    wind_direction: List[float]
    temperature: List[float]
    pressure: List[float]
    humidity: List[float]


class GustAlertResponse(BaseModel):
    """Gust alert response."""
    alert_id: Optional[str]
    triggered: bool
    thd_value: float
    lead_time_seconds: Optional[int]
    gust_speed: Optional[float]
    confidence: Optional[float]


# Routes
@router.post("/ake/compute", response_model=AKEResponse)
async def compute_ake(request: AKERequest):
    """Compute AKE index for a site."""
    try:
        ake = AKEComposite(
            site_id=request.site_id,
            climate_zone=request.climate_zone,
            site_type=request.site_type
        )
        
        ake.load_parameters(request.parameters)
        result = ake.compute()
        
        return AKEResponse(
            site_id=request.site_id,
            score=result['score'],
            classification=result['classification'],
            gust_risk=result['gust_risk'],
            confidence=result['confidence'],
            contributions=result['contributions']
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/ake/weights")
async def get_weights(climate_zone: str = "temperate"):
    """Get AKE parameter weights."""
    weights = AKEWeights(climate_zone=climate_zone)
    return weights.to_dict()


@router.get("/ake/thresholds")
async def get_thresholds(climate_zone: str = "temperate"):
    """Get AKE classification thresholds."""
    thresholds = AKETHRESHOLDS(climate_zone=climate_zone)
    return thresholds.to_dict()


@router.post("/gust/detect")
async def detect_gust(request: GustAlertRequest):
    """Detect gust from observations."""
    try:
        import pandas as pd
        
        # Create DataFrame
        df = pd.DataFrame({
            'wind_speed': request.wind_speed,
            'wind_direction': request.wind_direction,
            'temperature': request.temperature,
            'pressure': request.pressure,
            'humidity': request.humidity
        })
        
        # Initialize engine
        engine = GustPreAlertEngine(
            site_config={'location': request.site_id}
        )
        
        # Evaluate
        alert = engine.evaluate(df)
        
        if alert:
            return GustAlertResponse(
                alert_id=alert['alert_id'],
                triggered=True,
                thd_value=alert['thd_value'],
                lead_time_seconds=alert['lead_time_seconds'],
                gust_speed=alert['expected_gust_speed'],
                confidence=alert['confidence']
            )
        else:
            return GustAlertResponse(
                triggered=False,
                thd_value=0.0
            )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/parameters/{param_name}")
async def get_parameter_info(param_name: str):
    """Get information about a parameter."""
    params = {
        'KED': {'name': 'Kinetic Energy Density', 'weight': 0.22, 'unit': 'W/m²'},
        'TII': {'name': 'Turbulence Intensity Index', 'weight': 0.16, 'unit': ''},
        'VSR': {'name': 'Vertical Shear Ratio', 'weight': 0.14, 'unit': ''},
        'AOD': {'name': 'Aerosol Optical Depth', 'weight': 0.12, 'unit': ''},
        'THD': {'name': 'Thermal Helicity Dynamics', 'weight': 0.10, 'unit': ''},
        'PGF': {'name': 'Pressure Gradient Force', 'weight': 0.08, 'unit': 'm/s²'},
        'HCI': {'name': 'Humidity-Convection Interaction', 'weight': 0.07, 'unit': ''},
        'ASI': {'name': 'Atmospheric Stability Integration', 'weight': 0.06, 'unit': ''},
        'LRC': {'name': 'Local Roughness Coefficient', 'weight': 0.05, 'unit': 'm'}
    }
    
    if param_name.upper() not in params:
        raise HTTPException(status_code=404, detail="Parameter not found")
    
    return params[param_name.upper()]


@router.get("/version")
async def get_version():
    """Get API version."""
    from aerotica import __version__
    return {"version": __version__}


@router.get("/sites")
async def list_sites():
    """List available sites."""
    # This would query a database in production
    return {
        "sites": [
            {"id": "tokyo", "name": "Tokyo", "country": "Japan"},
            {"id": "brest", "name": "Brest", "country": "France"},
            {"id": "edinburgh", "name": "Edinburgh", "country": "UK"},
            {"id": "north_sea", "name": "North Sea", "country": "International"}
        ]
    }


@router.get("/status")
async def get_status():
    """Get system status."""
    import torch
    
    return {
        "status": "operational",
        "version": __version__,
        "cuda_available": torch.cuda.is_available(),
        "parameters": 9,
        "models_loaded": {
            "pinn": True,
            "wake": True,
            "alert": True
        }
    }
