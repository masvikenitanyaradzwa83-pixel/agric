"""
Weather analysis utilities for Gweru using real NASA POWER daily data.
Planting window detection uses proper agricultural seasons (Oct–Sep)
and rainfall-onset criteria commonly applied in Southern Africa / Zimbabwe.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

DATA_PATH = Path(__file__).parent.parent / "data" / "gweru_nasa_power_2015_2024.csv"


def load_weather(path: Optional[Path] = None) -> pd.DataFrame:
    path = path or DATA_PATH
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["doy"] = df["date"].dt.dayofyear
    # Agricultural year: season that starts in October of year Y and ends Sep of Y+1
    # is labelled as agri_year = Y+1 (the harvest year)
    df["agri_year"] = np.where(df["month"] >= 10, df["year"] + 1, df["year"])
    return df


def annual_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per calendar-year rainfall and temperature stats."""
    return (
        df.groupby("year")
        .agg(
            total_rain_mm=("PRECTOTCORR", "sum"),
            mean_temp=("T2M", "mean"),
            mean_max=("T2M_MAX", "mean"),
            mean_min=("T2M_MIN", "mean"),
            mean_rh=("RH2M", "mean"),
            rainy_days=("PRECTOTCORR", lambda x: (x >= 1.0).sum()),
        )
        .reset_index()
    )


def agri_year_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per agricultural year (Oct–Sep) rainfall."""
    return (
        df.groupby("agri_year")
        .agg(
            total_rain_mm=("PRECTOTCORR", "sum"),
            mean_temp=("T2M", "mean"),
            rainy_days=("PRECTOTCORR", lambda x: (x >= 1.0).sum()),
            start_date=("date", "min"),
            end_date=("date", "max"),
        )
        .reset_index()
    )


def climatology(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly averages (daily mean rainfall × 30.4 ≈ monthly total)."""
    return (
        df.groupby("month")
        .agg(
            mean_temp=("T2M", "mean"),
            mean_max=("T2M_MAX", "mean"),
            mean_min=("T2M_MIN", "mean"),
            daily_rain=("PRECTOTCORR", "mean"),
            mean_rh=("RH2M", "mean"),
        )
        .reset_index()
    )


def detect_rainfall_onset(
    period_df: pd.DataFrame,
    threshold_mm: float = 20.0,
    window_days: int = 3,
    min_follow_up_rain: float = 10.0,
    follow_up_days: int = 10,
) -> Optional[pd.Timestamp]:
    """
    Detect planting onset inside a pre-filtered period (usually 1 Oct – 28 Feb).

    Criterion (widely used in Zimbabwe / Southern Africa literature):
    - First occurrence of ≥ threshold_mm falling in ≤ window_days consecutive days
    - Followed by at least min_follow_up_rain in the subsequent follow_up_days
      (filters false starts / dry spells after a single heavy shower).
    """
    period = period_df.sort_values("date").reset_index(drop=True)
    if len(period) < window_days + follow_up_days + 2:
        return None

    rains = period["PRECTOTCORR"].fillna(0).values
    dates = period["date"].values

    for i in range(len(rains) - window_days - follow_up_days):
        window_sum = rains[i : i + window_days].sum()
        if window_sum >= threshold_mm:
            follow = rains[i + window_days : i + window_days + follow_up_days].sum()
            if follow >= min_follow_up_rain:
                return pd.Timestamp(dates[i])
    return None


def planting_windows_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute onset date for each agricultural season (Oct Y – Feb Y+1).
    agri_year is labelled by the harvest / main growing year (Y+1).
    """
    records = []
    agri_years = sorted(df["agri_year"].unique())

    for ay in agri_years:
        # Potential onset window: 1 Oct of previous calendar year → end of Feb of ay
        # For agri_year = 2020 this is 2019-10-01 to 2020-02-29
        start = pd.Timestamp(year=ay - 1, month=10, day=1)
        end = pd.Timestamp(year=ay, month=2, day=28)
        # Handle leap years roughly
        if ay % 4 == 0:
            end = pd.Timestamp(year=ay, month=2, day=29)

        mask = (df["date"] >= start) & (df["date"] <= end)
        period = df.loc[mask].copy()
        if period.empty:
            continue

        onset = detect_rainfall_onset(period)
        # Season rainfall = full agri year
        season_rain = df.loc[df["agri_year"] == ay, "PRECTOTCORR"].sum()

        records.append(
            {
                "agri_year": ay,
                "onset_date": onset,
                "onset_doy": int(onset.dayofyear) if onset is not None else None,
                "onset_month": int(onset.month) if onset is not None else None,
                "season_rain_mm": round(float(season_rain), 1),
                "search_start": start.date().isoformat(),
                "search_end": end.date().isoformat(),
            }
        )
    return pd.DataFrame(records)


def _doy_to_readable(doy: int, reference_year: int = 2024) -> str:
    """Convert day-of-year to a human-readable date string (non-leap reference)."""
    base = datetime(reference_year, 1, 1) + timedelta(days=int(doy) - 1)
    return base.strftime("%d %b")


def recommend_planting_window(
    df: pd.DataFrame, recent_seasons: int = 6
) -> Dict[str, Any]:
    """
    Recommend optimal planting window from historical onset statistics
    using proper agricultural seasons.
    """
    hist = planting_windows_history(df)
    valid = hist.dropna(subset=["onset_doy"]).copy()

    if len(valid) == 0:
        return {
            "median_onset_doy": 320,  # ~mid-Nov fallback
            "p25": 305,
            "p75": 340,
            "median_date_str": "15 Nov",
            "window_str": "01 Nov – 05 Dec",
            "advice": (
                "Insufficient onset data in the archive. "
                "Use local extension guidance – typical planting in Gweru is mid-November "
                "after effective rains (≥20 mm in 3 days)."
            ),
            "years_used": 0,
            "historical_onsets": [],
        }

    # Prefer more recent seasons
    recent = valid.tail(recent_seasons) if len(valid) >= recent_seasons else valid
    median = int(recent["onset_doy"].median())
    p25 = int(recent["onset_doy"].quantile(0.25))
    p75 = int(recent["onset_doy"].quantile(0.75))

    # Because onset can be in Dec (high DOY) or Jan (low DOY) we also report month distribution
    month_counts = recent["onset_month"].value_counts().to_dict()
    dominant_months = sorted(month_counts.items(), key=lambda x: -x[1])

    advice = (
        f"Based on the last {len(recent)} agricultural seasons (NASA POWER), "
        f"effective rainfall onset most often falls between {_doy_to_readable(p25)} and "
        f"{_doy_to_readable(p75)} (median ≈ {_doy_to_readable(median)}). "
        "Plant within 7–14 days after the first effective rains (≥20 mm in ≤3 consecutive days) "
        "to avoid false starts. Early-planted crops usually yield better when moisture is reliable."
    )

    return {
        "median_onset_doy": median,
        "p25": p25,
        "p75": p75,
        "median_date_str": _doy_to_readable(median),
        "window_str": f"{_doy_to_readable(p25)} – {_doy_to_readable(p75)}",
        "advice": advice,
        "years_used": len(recent),
        "month_distribution": dominant_months,
        "historical_onsets": recent[
            ["agri_year", "onset_date", "onset_doy", "onset_month", "season_rain_mm"]
        ].to_dict(orient="records"),
        "all_seasons_table": valid.to_dict(orient="records"),
    }


def current_season_risk(
    df: pd.DataFrame,
    lookback_days: int = 14,
) -> Dict[str, float]:
    """Recent weather summary used by the pest-risk engine."""
    recent = df.tail(lookback_days)
    return {
        "mean_temp": float(recent["T2M"].mean()),
        "mean_max": float(recent["T2M_MAX"].mean()),
        "mean_rh": float(recent["RH2M"].mean()),
        "total_rain": float(recent["PRECTOTCORR"].sum()),
        "rainy_days": int((recent["PRECTOTCORR"] >= 1).sum()),
    }


def season_progress(df: pd.DataFrame) -> Dict[str, Any]:
    """Quick status of the current / most recent agricultural season."""
    today = df["date"].max()
    current_ay = int(df.loc[df["date"] == today, "agri_year"].iloc[0])
    season = df[df["agri_year"] == current_ay]
    rain_so_far = float(season["PRECTOTCORR"].sum())
    return {
        "agri_year": current_ay,
        "rain_to_date_mm": round(rain_so_far, 1),
        "days_in_season": len(season),
        "last_data_date": today.date().isoformat(),
    }
