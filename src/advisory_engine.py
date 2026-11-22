"""
Main advisory engine that ties weather analysis, crop knowledge and the neural net together.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import sys

sys.path.insert(0, str(Path(__file__).parent))

from crop_knowledge import (
    CROPS,
    PEST_RULES,
    get_crop_recommendation_features,
)
from weather_analysis import (
    load_weather,
    annual_summary,
    climatology,
    recommend_planting_window,
    current_season_risk,
    season_progress,
    agri_year_summary,
)
from models import CropSuitabilityNet, predict_crop_scores, CROP_TO_IDX


class AdvisoryEngine:
    def __init__(self, model_path: Optional[Path] = None):
        self.weather = load_weather()
        self.yearly = annual_summary(self.weather)
        self.clim = climatology(self.weather)

        # Long-term averages for Gweru
        self.avg_annual_rain = float(self.yearly["total_rain_mm"].mean())
        self.avg_temp = float(self.weather["T2M"].mean())

        self.model = CropSuitabilityNet()
        path = model_path or Path(__file__).parent.parent / "models" / "crop_suitability_net.pt"
        if path.exists():
            self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            self.model.eval()
        else:
            print("Warning: model weights not found, using random init")

    def get_crop_recommendations(
        self,
        soil_pH: float = 5.2,
        farm_size_ha: float = 1.0,
        risk_pref: str = "neutral",
        input_access: str = "medium",
        past_yield: str = "medium",  # low / medium / high (qualitative)
        current_crops: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        # Use long-term climate + optional adjustment for recent years
        recent_rain = float(self.yearly.tail(3)["total_rain_mm"].mean())
        annual_rain = 0.6 * self.avg_annual_rain + 0.4 * recent_rain

        # Neural net refined scores
        nn_scores = predict_crop_scores(
            self.model,
            soil_pH=soil_pH,
            annual_rain=annual_rain,
            mean_temp=self.avg_temp,
            risk_pref=risk_pref,
            input_access=input_access,
        )

        # Knowledge-base scores for transparency
        kb_scores = get_crop_recommendation_features(
            soil_pH, annual_rain, self.avg_temp, risk_pref, input_access
        )

        # Combine (70% NN + 30% KB for stability)
        ranked = []
        for key in CROPS:
            combined = 0.7 * nn_scores[key] + 0.3 * kb_scores[key]["composite"]
            ranked.append(
                {
                    "crop_key": key,
                    "name": CROPS[key]["name"],
                    "local_names": CROPS[key]["local_names"],
                    "score": round(combined, 3),
                    "nn_score": nn_scores[key],
                    "kb_score": kb_scores[key]["composite"],
                    "drought_tolerance": CROPS[key]["drought_tolerance"],
                    "input_requirement": CROPS[key]["input_requirement"],
                    "notes": CROPS[key]["notes"],
                    "typical_yield_t_ha": CROPS[key]["typical_yield_t_ha"],
                }
            )
        ranked.sort(key=lambda x: -x["score"])

        # Simple farm-size / past-yield commentary
        commentary = []
        if farm_size_ha < 0.5:
            commentary.append("Very small holding – prioritise high-value or dual-purpose crops (cowpea, groundnut, beans).")
        if past_yield == "low" and risk_pref != "risk_seeking":
            commentary.append("Past low yields suggest focusing on drought-tolerant and low-input options.")
        if input_access == "low":
            commentary.append("Limited input access – favour sorghum, millets and cowpea.")

        return {
            "recommendations": ranked,
            "climate_context": {
                "avg_annual_rain_mm": round(self.avg_annual_rain, 1),
                "recent_3yr_rain_mm": round(recent_rain, 1),
                "used_rain_mm": round(annual_rain, 1),
                "avg_temp_C": round(self.avg_temp, 1),
                "soil_pH_used": soil_pH,
            },
            "commentary": commentary,
            "top_3": ranked[:3],
        }

    def get_planting_advice(self) -> Dict[str, Any]:
        return recommend_planting_window(self.weather)

    def get_pest_alerts(
        self,
        crop_key: str = "maize",
        crop_stage: str = "vegetative",
        lookback_days: int = 14,
    ) -> List[Dict[str, Any]]:
        risk = current_season_risk(self.weather, lookback_days)
        alerts = []
        for pest, rule in PEST_RULES.items():
            if crop_key not in rule["crops"]:
                continue
            temp_ok = rule["temp_favor_C"][0] <= risk["mean_temp"] <= rule["temp_favor_C"][1]
            rh_ok = rule["humidity_favor"][0] <= risk["mean_rh"] <= rule["humidity_favor"][1]
            stage_ok = crop_stage in rule["critical_stages"]

            risk_level = "low"
            if temp_ok and rh_ok and stage_ok:
                risk_level = "high"
            elif (temp_ok and rh_ok) or (temp_ok and stage_ok):
                risk_level = "medium"

            if risk_level != "low":
                alerts.append(
                    {
                        "pest": pest.replace("_", " ").title(),
                        "risk_level": risk_level,
                        "advice": rule["advice"],
                        "current_mean_temp": round(risk["mean_temp"], 1),
                        "current_mean_rh": round(risk["mean_rh"], 1),
                    }
                )
        return alerts

    def full_advisory(
        self,
        soil_pH: float = 5.2,
        farm_size_ha: float = 1.0,
        risk_pref: str = "neutral",
        input_access: str = "medium",
        past_yield: str = "medium",
        current_crops: Optional[List[str]] = None,
        focus_crop: str = "maize",
        crop_stage: str = "vegetative",
    ) -> Dict[str, Any]:
        crops = self.get_crop_recommendations(
            soil_pH, farm_size_ha, risk_pref, input_access, past_yield, current_crops
        )
        planting = self.get_planting_advice()
        pests = self.get_pest_alerts(focus_crop, crop_stage)

        progress = season_progress(self.weather)
        return {
            "crop_recommendations": crops,
            "planting_window": planting,
            "pest_alerts": pests,
            "season_progress": progress,
            "location": "Gweru, Midlands, Zimbabwe",
            "data_source": "NASA POWER (2015–2024) + local agronomic knowledge",
        }
