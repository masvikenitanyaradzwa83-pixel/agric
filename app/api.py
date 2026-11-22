"""
FastAPI backend for AgroVision – ready for mobile clients or other front-ends.
"""

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from typing import List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from advisory_engine import AdvisoryEngine

app = FastAPI(
    title="AgroVision API",
    description="Climate-smart advisory for smallholder farmers in Gweru, Zimbabwe",
    version="1.0.0",
)

engine = AdvisoryEngine()


class AdvisoryRequest(BaseModel):
    soil_pH: float = Field(5.2, ge=4.0, le=8.0)
    farm_size_ha: float = Field(1.0, ge=0.1, le=100)
    risk_pref: str = Field("neutral", pattern="^(risk_averse|neutral|risk_seeking)$")
    input_access: str = Field("medium", pattern="^(low|medium|high)$")
    past_yield: str = Field("medium", pattern="^(low|medium|high)$")
    focus_crop: str = "maize"
    crop_stage: str = "vegetative"


@app.get("/health")
def health():
    return {"status": "ok", "location": "Gweru, Zimbabwe"}


@app.post("/advisory")
def get_advisory(req: AdvisoryRequest):
    return engine.full_advisory(
        soil_pH=req.soil_pH,
        farm_size_ha=req.farm_size_ha,
        risk_pref=req.risk_pref,
        input_access=req.input_access,
        past_yield=req.past_yield,
        focus_crop=req.focus_crop,
        crop_stage=req.crop_stage,
    )


@app.get("/crops")
def list_crops():
    from crop_knowledge import CROPS
    return {k: {"name": v["name"], "local_names": v["local_names"]} for k, v in CROPS.items()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
