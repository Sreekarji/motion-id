"""
Motion ID FastAPI Backend
Serves the biometric authentication pipeline as a REST API.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
from model_runner import ModelManager, cfg
import os, json, torch

# ─────────────────────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Motion ID API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────

manager: Optional[ModelManager] = None


@app.on_event("startup")
async def startup():
    global manager
    manager = ModelManager(
        checkpoints_dir=r"D:\motionid\checkpoints",
        uv_processed_dir=r"D:\motionid\uv_processed",
        mpi_processed_dir=r"D:\motionid\mpi_processed",
        inventory_path=r"D:\motionid\inventory.json"
    )
    print(f"Models loaded. Users: {manager.get_available_users()}")


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────────────────────

class SensorWindow3s(BaseModel):
    """6 sensors x 3 axes x ~150 samples for MPI stage."""
    acc:  List[List[float]]
    grav: List[List[float]]
    gyro: List[List[float]]
    lin:  List[List[float]]
    mag:  List[List[float]]
    rot:  List[List[float]]


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    gpu_available = torch.cuda.is_available()
    return {
        "status": "ok",
        "gpu": gpu_available,
        "gpu_name": torch.cuda.get_device_name(0) if gpu_available else None,
        "users_loaded": len(manager.get_available_users()) if manager else 0,
        "users": manager.get_available_users() if manager else [],
        "mpi_models": len(manager.mpi_models) if manager else 0,
        "mpi_stubbed": len(manager.mpi_models) == 0 if manager else True,
    }


@app.get("/users")
async def list_users():
    if manager is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    return {"users": manager.get_available_users()}


@app.get("/users/{user_id}/sample")
async def get_sample(user_id: int):
    if manager is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    if user_id not in manager.get_available_users():
        raise HTTPException(status_code=404, detail=f"User {user_id} not available")
    try:
        sample = manager.get_random_sample(user_id)
        return {
            "user_id": user_id,
            "features": sample["features"],
            "n_trials_total": sample["n_trials_total"],
            "trial_index": sample["trial_index"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/mpi")
async def predict_mpi(window: SensorWindow3s):
    if manager is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    try:
        sensor_data = {
            "acc": window.acc, "grav": window.grav, "gyro": window.gyro,
            "lin": window.lin, "mag": window.mag, "rot": window.rot,
        }
        result = manager.predict_mpi(sensor_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/demo/{user_id}")
async def predict_demo(user_id: int):
    if manager is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    if user_id not in manager.get_available_users():
        raise HTTPException(status_code=404, detail=f"User {user_id} not available")
    try:
        # Get UV sample
        sample = manager.get_random_sample(user_id)
        features = np.array(sample["features"], dtype=np.float32)

        # Get MPI sample and run real MPI inference
        mpi_sample = manager.get_random_mpi_sample()
        sensor_data_3s = mpi_sample["sensor_data"] if mpi_sample else None

        result = manager.predict_full(user_id, features, sensor_data_3s=sensor_data_3s)
        result["sample"] = {
            "features": sample["features"],
            "trial_index": sample["trial_index"],
            "n_trials_total": sample["n_trials_total"],
        }
        # Include MPI source info
        if mpi_sample:
            result["mpi"]["source_file"] = mpi_sample["source_file"]
            result["mpi"]["sample_index"] = mpi_sample["sample_index"]
            result["mpi"]["true_label"] = mpi_sample["label"]
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# SERVE FRONTEND (production)
# ─────────────────────────────────────────────────────────────────────────────

frontend_build = r"D:\motionid\frontend\dist"
if os.path.exists(frontend_build):
    app.mount("/", StaticFiles(directory=frontend_build, html=True), name="static")
