
"""FastAPI endpoints for the simulated boring-bar vibration-control demo."""
import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from deployment_service import (
    DEFAULT_MODEL_PATH,
    compare_controllers,
    get_model_status,
    run_simulation,
)

app = FastAPI(
    title="Boring-Bar Vibration Control API",
    description=(
        "Educational API for a single-mode simulated vibration-control model. "
        "This API does not control physical machinery."
    ),
    version="1.0.0",
)

MODEL_PATH = os.getenv("SAC_MODEL_PATH", DEFAULT_MODEL_PATH)


class SimulationRequest(BaseModel):
    controller: Literal["No control", "PID", "SAC"] = "SAC"
    amplitude_n: float = Field(default=1.0, ge=0.0, le=10.0)
    frequency_hz: float = Field(default=174.0, ge=0.0, le=1000.0)
    phase_rad: float = Field(default=0.0, ge=-100.0, le=100.0)
    duration_s: float = Field(default=0.2, gt=0.0, le=2.0)


class ComparisonRequest(BaseModel):
    amplitude_n: float = Field(default=1.0, ge=0.0, le=10.0)
    frequency_hz: float = Field(default=174.0, ge=0.0, le=1000.0)
    phase_rad: float = Field(default=0.0, ge=-100.0, le=100.0)
    duration_s: float = Field(default=0.2, gt=0.0, le=2.0)


@app.get("/")
def root():
    return {
        "service": "Boring-Bar Vibration Control API",
        "simulation_only": True,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    status = get_model_status(MODEL_PATH)
    return {
        "status": "ok",
        "simulation_only": True,
        "sac_model": status,
    }


@app.get("/model")
def model_status():
    return get_model_status(MODEL_PATH)


@app.post("/simulate")
def simulate(request: SimulationRequest):
    try:
        return run_simulation(
            controller_name=request.controller,
            amplitude_n=request.amplitude_n,
            frequency_hz=request.frequency_hz,
            phase_rad=request.phase_rad,
            duration_s=request.duration_s,
            model_path=MODEL_PATH,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        # Avoid returning internal exception details to API clients.
        raise HTTPException(
            status_code=500,
            detail="Simulation failed. Check the server terminal logs.",
        ) from exc


@app.post("/compare")
def compare(request: ComparisonRequest):
    try:
        return {
            "simulation_only": True,
            "results": compare_controllers(
                amplitude_n=request.amplitude_n,
                frequency_hz=request.frequency_hz,
                phase_rad=request.phase_rad,
                duration_s=request.duration_s,
                model_path=MODEL_PATH,
            ),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Controller comparison failed. Check server logs.",
        ) from exc
