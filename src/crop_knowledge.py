"""
Crop knowledge base for Gweru / Midlands, Zimbabwe.
Based on local optima (pH from Business Times / research papers),
EcoCrop-style ranges, drought tolerance, and common smallholder crops.
"""

from typing import Dict, List, Any

# Risk preference multipliers / preferred traits
RISK_PROFILES = {
    "risk_averse": {"prefer_drought_tolerant": True, "prefer_stable_yield": True, "avoid_high_input": True},
    "neutral": {"prefer_drought_tolerant": False, "prefer_stable_yield": True, "avoid_high_input": False},
    "risk_seeking": {"prefer_drought_tolerant": False, "prefer_stable_yield": False, "avoid_high_input": False},
}

CROPS: Dict[str, Dict[str, Any]] = {
    "maize": {
        "name": "Maize",
        "local_names": ["Chibage", "Mealies"],
        "optimal_pH": (5.0, 5.8),
        "tolerable_pH": (4.5, 6.5),
        "temp_opt_C": (18, 30),
        "temp_range_C": (10, 35),
        "rainfall_opt_mm": (500, 800),
        "rainfall_min_mm": 400,
        "growing_days": (90, 140),
        "drought_tolerance": 0.4,  # 0-1 scale (low)
        "input_requirement": "high",  # fertilizer responsive
        "typical_yield_t_ha": (1.5, 4.0),  # smallholder range
        "season": "summer",
        "notes": "Staple. Sensitive to dry spells at flowering. Prefers NR III.",
        "common_pests": ["fall_armyworm", "stalk_borer", "maize_streak_virus"],
    },
    "sorghum": {
        "name": "Sorghum",
        "local_names": ["Mapfunde", "Amabele"],
        "optimal_pH": (5.0, 5.5),
        "tolerable_pH": (4.5, 7.0),
        "temp_opt_C": (20, 32),
        "temp_range_C": (12, 38),
        "rainfall_opt_mm": (400, 650),
        "rainfall_min_mm": 300,
        "growing_days": (90, 130),
        "drought_tolerance": 0.85,
        "input_requirement": "low",
        "typical_yield_t_ha": (0.8, 2.5),
        "season": "summer",
        "notes": "Excellent drought tolerance. Good for NR IV.",
        "common_pests": ["sorghum_midge", "stem_borer", "birds"],
    },
    "pearl_millet": {
        "name": "Pearl Millet",
        "local_names": ["Mhunga", "Unyawuthi"],
        "optimal_pH": (5.0, 6.5),
        "tolerable_pH": (4.5, 7.5),
        "temp_opt_C": (22, 35),
        "temp_range_C": (12, 40),
        "rainfall_opt_mm": (300, 550),
        "rainfall_min_mm": 250,
        "growing_days": (70, 110),
        "drought_tolerance": 0.95,
        "input_requirement": "low",
        "typical_yield_t_ha": (0.6, 1.8),
        "season": "summer",
        "notes": "Most drought tolerant cereal. Ideal for risk-averse farmers in drier wards.",
        "common_pests": ["millet_head_miner", "birds"],
    },
    "finger_millet": {
        "name": "Finger Millet",
        "local_names": ["Rapoko", "Rukweza", "Uphoko"],
        "optimal_pH": (5.0, 6.0),
        "tolerable_pH": (4.5, 7.0),
        "temp_opt_C": (18, 28),
        "temp_range_C": (10, 32),
        "rainfall_opt_mm": (450, 700),
        "rainfall_min_mm": 350,
        "growing_days": (100, 140),
        "drought_tolerance": 0.75,
        "input_requirement": "low",
        "typical_yield_t_ha": (0.5, 1.5),
        "season": "summer",
        "notes": "Nutritious, good storage. Tolerates poorer soils.",
        "common_pests": ["blast", "stem_borer"],
    },
    "groundnut": {
        "name": "Groundnut",
        "local_names": ["Nzungu", "Peanuts"],
        "optimal_pH": (5.5, 6.5),
        "tolerable_pH": (5.0, 7.0),
        "temp_opt_C": (20, 30),
        "temp_range_C": (15, 35),
        "rainfall_opt_mm": (450, 700),
        "rainfall_min_mm": 350,
        "growing_days": (90, 130),
        "drought_tolerance": 0.55,
        "input_requirement": "medium",
        "typical_yield_t_ha": (0.6, 1.8),
        "season": "summer",
        "notes": "Important legume and cash crop. Benefits from calcium.",
        "common_pests": ["leaf_spot", "rosette_virus", "aphids"],
    },
    "cowpea": {
        "name": "Cowpea",
        "local_names": ["Nyemba", "Black-eyed pea"],
        "optimal_pH": (5.5, 6.5),
        "tolerable_pH": (5.0, 7.5),
        "temp_opt_C": (20, 32),
        "temp_range_C": (15, 38),
        "rainfall_opt_mm": (400, 650),
        "rainfall_min_mm": 300,
        "growing_days": (60, 100),
        "drought_tolerance": 0.80,
        "input_requirement": "low",
        "typical_yield_t_ha": (0.4, 1.2),
        "season": "summer",
        "notes": "Fast maturing legume. Improves soil N. Dual purpose (grain + leaves).",
        "common_pests": ["aphids", "pod_borer", "thrips"],
    },
    "sugar_bean": {
        "name": "Sugar Bean / Common Bean",
        "local_names": ["Nyemba", "Beans"],
        "optimal_pH": (5.5, 6.5),
        "tolerable_pH": (5.3, 7.0),
        "temp_opt_C": (18, 28),
        "temp_range_C": (12, 32),
        "rainfall_opt_mm": (450, 650),
        "rainfall_min_mm": 350,
        "growing_days": (70, 100),
        "drought_tolerance": 0.45,
        "input_requirement": "medium",
        "typical_yield_t_ha": (0.5, 1.5),
        "season": "summer",
        "notes": "Good market crop. Sensitive to waterlogging and extreme heat.",
        "common_pests": ["bean_fly", "anthracnose", "rust"],
    },
    "sunflower": {
        "name": "Sunflower",
        "local_names": ["Sunflower"],
        "optimal_pH": (5.5, 7.0),
        "tolerable_pH": (5.0, 7.5),
        "temp_opt_C": (18, 28),
        "temp_range_C": (10, 35),
        "rainfall_opt_mm": (400, 700),
        "rainfall_min_mm": 300,
        "growing_days": (80, 120),
        "drought_tolerance": 0.70,
        "input_requirement": "medium",
        "typical_yield_t_ha": (0.8, 2.0),
        "season": "summer",
        "notes": "Oilseed. Deep rooting, reasonably drought tolerant.",
        "common_pests": ["birds", "head_borer", "rust"],
    },
}

# Pest risk rules (simplified phenology-inspired, based on literature)
PEST_RULES = {
    "fall_armyworm": {
        "crops": ["maize", "sorghum"],
        "temp_favor_C": (20, 30),
        "humidity_favor": (60, 90),
        "critical_stages": ["vegetative", "whorl", "tasseling"],
        "advice": "High temperatures + humidity favor rapid development. Scout early.",
    },
    "stalk_borer": {
        "crops": ["maize", "sorghum"],
        "temp_favor_C": (18, 28),
        "humidity_favor": (50, 80),
        "critical_stages": ["vegetative", "stem_elongation"],
        "advice": "Plant early or use resistant varieties. Monitor for eggs on leaves.",
    },
    "leaf_spot": {
        "crops": ["groundnut", "sorghum"],
        "temp_favor_C": (20, 30),
        "humidity_favor": (70, 95),
        "critical_stages": ["vegetative", "flowering", "podding"],
        "advice": "High humidity increases risk. Rotate crops, use fungicides if available.",
    },
    "aphids": {
        "crops": ["cowpea", "groundnut", "sugar_bean"],
        "temp_favor_C": (15, 25),
        "humidity_favor": (50, 80),
        "critical_stages": ["seedling", "vegetative", "flowering"],
        "advice": "Watch for virus transmission. Encourage natural enemies.",
    },
}




def score_pH(crop_key: str, soil_pH: float) -> float:
    """Return 0-1 suitability score for pH."""
    c = CROPS[crop_key]
    opt_lo, opt_hi = c["optimal_pH"]
    tol_lo, tol_hi = c["tolerable_pH"]
    if opt_lo <= soil_pH <= opt_hi:
        return 1.0
    if soil_pH < tol_lo or soil_pH > tol_hi:
        return 0.1
    # Linear fall-off outside optimal
    if soil_pH < opt_lo:
        return 0.3 + 0.7 * (soil_pH - tol_lo) / (opt_lo - tol_lo)
    else:
        return 0.3 + 0.7 * (tol_hi - soil_pH) / (tol_hi - opt_hi)


def score_rainfall(crop_key: str, annual_rain_mm: float) -> float:
    c = CROPS[crop_key]
    opt_lo, opt_hi = c["rainfall_opt_mm"]
    min_r = c["rainfall_min_mm"]
    if opt_lo <= annual_rain_mm <= opt_hi:
        return 1.0
    if annual_rain_mm < min_r:
        return max(0.05, annual_rain_mm / min_r * 0.3)
    if annual_rain_mm < opt_lo:
        return 0.4 + 0.6 * (annual_rain_mm - min_r) / (opt_lo - min_r)
    # Too much rain – mild penalty
    return max(0.5, 1.0 - (annual_rain_mm - opt_hi) / 400)


def score_temperature(crop_key: str, mean_temp_C: float) -> float:
    c = CROPS[crop_key]
    opt_lo, opt_hi = c["temp_opt_C"]
    t_lo, t_hi = c["temp_range_C"]
    if opt_lo <= mean_temp_C <= opt_hi:
        return 1.0
    if mean_temp_C < t_lo or mean_temp_C > t_hi:
        return 0.1
    if mean_temp_C < opt_lo:
        return 0.3 + 0.7 * (mean_temp_C - t_lo) / (opt_lo - t_lo)
    return 0.3 + 0.7 * (t_hi - mean_temp_C) / (t_hi - opt_hi)


def score_risk_preference(crop_key: str, risk_pref: str, input_access: str) -> float:
    c = CROPS[crop_key]
    score = 0.7
    if risk_pref == "risk_averse":
        score += 0.25 * c["drought_tolerance"]
        if c["input_requirement"] == "high" and input_access == "low":
            score -= 0.3
        elif c["input_requirement"] == "low":
            score += 0.15
    elif risk_pref == "risk_seeking":
        if c["input_requirement"] == "high" and input_access in ("medium", "high"):
            score += 0.2  # can chase higher yields
        score -= 0.1 * c["drought_tolerance"]  # slightly prefer higher potential crops
    # Neutral: balanced
    if input_access == "low" and c["input_requirement"] == "high":
        score -= 0.25
    return max(0.05, min(1.0, score))


def get_crop_recommendation_features(
    soil_pH: float,
    annual_rain_mm: float,
    mean_temp_C: float,
    risk_pref: str,
    input_access: str,
) -> Dict[str, Dict[str, float]]:
    """Return per-crop feature scores + composite."""
    results = {}
    for key in CROPS:
        pH_s = score_pH(key, soil_pH)
        rain_s = score_rainfall(key, annual_rain_mm)
        temp_s = score_temperature(key, mean_temp_C)
        risk_s = score_risk_preference(key, risk_pref, input_access)
        # Weighted composite (can be replaced / refined by NN)
        composite = 0.30 * pH_s + 0.30 * rain_s + 0.20 * temp_s + 0.20 * risk_s
        results[key] = {
            "pH_score": round(pH_s, 3),
            "rain_score": round(rain_s, 3),
            "temp_score": round(temp_s, 3),
            "risk_score": round(risk_s, 3),
            "composite": round(composite, 3),
            "name": CROPS[key]["name"],
            "drought_tolerance": CROPS[key]["drought_tolerance"],
            "notes": CROPS[key]["notes"],
        }
    return results
