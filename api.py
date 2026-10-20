
"""FastAPI backend for the machining-aware boring-bar simulation."""

import logging
import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from deployment_service import (
    DEFAULT_MODEL_PATH,
    ALLOWED_CONTROLLERS,
    compare_controllers,
    get_model_status,
    run_simulation,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

MODEL_PATH = os.getenv("SAC_MODEL_PATH", DEFAULT_MODEL_PATH)

app = FastAPI(
    title="Boring-Bar Vibration Control API",
    description="Simulation API for machining-aware vibration control.",
    version="2.0.0",
)


class MachiningRequest(BaseModel):
    spindle_rpm: float = Field(default=9000.0, gt=0, le=100000)
    feed_mm_rev: float = Field(default=0.1, gt=0, le=10)
    depth_of_cut_mm: float = Field(default=0.5, gt=0, le=20)
    specific_cutting_force_n_mm2: float = Field(default=1800.0, gt=0, le=10000)
    force_fluctuation_fraction: float = Field(default=0.2, ge=0, le=2)
    random_force_fraction: float = Field(default=0.08, ge=0, le=2)
    modal_participation: float = Field(default=0.02, gt=0, le=1)
    transient_rate_hz: float = Field(default=3.0, ge=0, le=1000)
    transient_fraction: float = Field(default=0.5, ge=0, le=5)
    phase_rad: float = Field(default=0.0, ge=-100, le=100)
    duration_s: float = Field(default=0.2, gt=0, le=2.0)
    seed: int = Field(default=12345, ge=0, le=2147483647)


class SimulationRequest(MachiningRequest):
    controller: Literal["No control", "PID", "SAC"] = "SAC"


@app.get("/")
def root():
    return {
        "service": "Boring-Bar Vibration Control API",
        "status": "running",
        "simulation_only": True,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    try:
        status = get_model_status(MODEL_PATH)
    except Exception:
        logger.exception("Could not inspect model status")
        status = {
            "model_path": str(MODEL_PATH),
            "model_exists": False,
            "message": "Could not inspect the model file.",
        }

    return {
        "status": "ok",
        "simulation_only": True,
        "sac_model": status,
    }


@app.get("/model")
def model():
    try:
        return get_model_status(MODEL_PATH)
    except Exception as exc:
        logger.exception("Model status request failed")
        raise HTTPException(
            status_code=500,
            detail="Could not read model status. Check server logs.",
        ) from exc


def _machining_kwargs(request: MachiningRequest) -> dict:
    """Build only arguments supported by the machining-aware service."""
    return {
        "spindle_rpm": request.spindle_rpm,
        "feed_mm_rev": request.feed_mm_rev,
        "depth_of_cut_mm": request.depth_of_cut_mm,
        "specific_cutting_force_n_mm2": request.specific_cutting_force_n_mm2,
        "force_fluctuation_fraction": request.force_fluctuation_fraction,
        "random_force_fraction": request.random_force_fraction,
        "modal_participation": request.modal_participation,
        "transient_rate_hz": request.transient_rate_hz,
        "transient_fraction": request.transient_fraction,
        "phase_rad": request.phase_rad,
        "duration_s": request.duration_s,
        "seed": request.seed,
        "model_path": MODEL_PATH,
    }


@app.post("/simulate")
def simulate(request: SimulationRequest):
    logger.info(
        "Simulation requested: controller=%s rpm=%.1f feed=%.4f "
        "depth=%.4f duration=%.4f seed=%d",
        request.controller,
        request.spindle_rpm,
        request.feed_mm_rev,
        request.depth_of_cut_mm,
        request.duration_s,
        request.seed,
    )

    try:
        result = run_simulation(
            controller_name=request.controller,
            **_machining_kwargs(request),
        )

        if not isinstance(result, dict):
            raise RuntimeError("Simulation service returned a non-object result.")

        if "metrics" not in result or "timeseries" not in result:
            raise RuntimeError(
                "Simulation result must contain 'metrics' and 'timeseries'."
            )

        return result

    except FileNotFoundError as exc:
        logger.warning("Model unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    except ValueError as exc:
        logger.warning("Invalid simulation input: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Unhandled error in POST /simulate")
        raise HTTPException(
            status_code=500,
            detail="Simulation failed. Read the API terminal traceback.",
        ) from exc


@app.post("/compare")
def compare(request: MachiningRequest):
    logger.info(
        "Comparison requested: rpm=%.1f feed=%.4f depth=%.4f duration=%.4f",
        request.spindle_rpm,
        request.feed_mm_rev,
        request.depth_of_cut_mm,
        request.duration_s,
    )

    try:
        results = compare_controllers(**_machining_kwargs(request))
        return {
            "simulation_only": True,
            "results": results,
        }

    except FileNotFoundError as exc:
        logger.warning("Model unavailable during comparison: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    except ValueError as exc:
        logger.warning("Invalid comparison input: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Unhandled error in POST /compare")
        raise HTTPException(
            status_code=500,
            detail="Controller comparison failed. Read the API terminal traceback.",
        ) from exc
