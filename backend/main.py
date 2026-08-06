"""
Motion ID FastAPI Backend
Serves the biometric authentication pipeline as a REST API.
"""
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
from model_runner import ModelManager, cfg
import os, json, torch
from pathlib import Path

_BASE = Path(__file__).parent.parent   # = D:\motionid  (one level above backend/)

# ─────────────────────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────────────────────

manager: Optional[ModelManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager
    manager = ModelManager(
        checkpoints_dir=str(_BASE / "checkpoints"),
        uv_processed_dir=str(_BASE / "uv_processed"),
        mpi_processed_dir=str(_BASE / "mpi_processed"),
        inventory_path=str(_BASE / "inventory.json")
    )
    print(f"Models loaded. Users: {manager.get_available_users()}")
    yield
    # shutdown: nothing to clean up for a demo


app = FastAPI(title="Motion ID API", version="1.0", lifespan=lifespan)

_ngrok_url = os.environ.get("NGROK_URL", "")   # set in env if using ngrok
_origins = ["http://localhost:5173", "http://localhost:3000", "http://localhost:8000"]
if _ngrok_url:
    _origins.append(_ngrok_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
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


@app.post("/predict/demo/{user_id}")
async def predict_demo(user_id: int):
    """Full verification: one genuine attempt plus one impostor attempt per other user.

    Every attempt is scored against the CLAIMED user's fine-tuned model, so the
    impostor runs are exactly the attack this system is meant to stop: someone
    else's motion presented under user_id's identity.
    """
    if manager is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    if user_id not in manager.get_available_users():
        raise HTTPException(status_code=404, detail=f"User {user_id} not available")
    try:
        # ---- genuine attempt: the claimed user's own data ----
        sample = manager.get_random_sample(user_id)
        genuine_res = manager.predict_full(
            user_id, np.array(sample["features"], dtype=np.float32))
        genuine_uv = genuine_res["uv"]

        genuine = {
            "decision": genuine_uv["decision"],
            "uv_score": genuine_uv["score"],
            "threshold": genuine_uv["threshold"],
            "trial_index": sample["trial_index"],
            "n_trials_total": sample["n_trials_total"],
        }

        # ---- impostor attempts: everyone else's data, claimed user's model ----
        impostors = []
        for other_id in manager.get_available_users():
            if other_id == user_id:
                continue
            try:
                imp_sample = manager.get_random_sample(other_id)
                imp_res = manager.predict_full(
                    user_id, np.array(imp_sample["features"], dtype=np.float32))
                imp_uv = imp_res["uv"]
                impostors.append({
                    "impostor_user_id": other_id,
                    "decision": imp_uv["decision"],
                    "uv_score": imp_uv["score"],
                    "threshold": imp_uv["threshold"],
                })
            except Exception as e:
                # One unreadable user must not sink the whole run.
                print(f"  WARNING: impostor {other_id} vs {user_id} failed: {e}")

        total = len(impostors)
        accepted = sum(1 for i in impostors if i["decision"] == "ACCEPT")
        rejected = total - accepted
        far = (accepted / total) if total else 0.0

        if far == 0:
            far_display = "0.00% · 1/∞"
        else:
            far_display = f"{far * 100:.2f}% · 1/{round(1 / far)}"

        return {
            "claimed_user_id": user_id,
            "genuine": genuine,
            "impostors": impostors,
            "summary": {
                "total_impostors": total,
                "correctly_rejected": rejected,
                "incorrectly_accepted": accepted,
                "far": round(far, 6),
                "far_display": far_display,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# SERVE FRONTEND (production)
# ─────────────────────────────────────────────────────────────────────────────

frontend_build = _BASE / "frontend" / "dist"
if frontend_build.exists():
    # SPA fallback: the client router owns /verify, but StaticFiles only knows
    # about files on disk, so a hard refresh there would 404. Serve index.html
    # for any unmatched non-API path and let the router resolve it.
    from fastapi.responses import FileResponse

    _index = frontend_build / "index.html"
    _assets = frontend_build / "assets"
    if _assets.exists():
        app.mount("/assets", StaticFiles(directory=str(_assets)), name="assets")

    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str):
        # Resolve and confirm containment before serving. Without this,
        # "../../backend/main.py" escapes the build dir, and an absolute
        # full_path makes pathlib discard frontend_build entirely
        # (Path("/a/b") / "/etc/passwd" == Path("/etc/passwd")).
        if full_path:
            root = frontend_build.resolve()
            candidate = (root / full_path).resolve()
            if candidate.is_file() and candidate.is_relative_to(root):
                return FileResponse(str(candidate))
        return FileResponse(str(_index))
