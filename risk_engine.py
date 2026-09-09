"""
risk_engine.py — SHIELD Unified Road Risk Engine
=================================================
Fuses three signal streams into a single actionable risk score:

  Signal 1 — Vision   (YOLO hazard detections)         weight: 50%
  Signal 2 — Weather  (OWM conditions + visibility)    weight: 30%
  Signal 3 — Speed    (user-reported / GPS estimate)   weight: 20%

Output:
  • RiskAssessment dataclass — score, level, recommendations, per-signal breakdown
  • Rule-based recommendation engine (no LLM required)
  • Alert system — generates priority-ordered alerts list
  • speed_risk() utility — maps speed (km/h) → 0-100 risk
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from config import settings
from yolo_detector import FrameAnalysis
from weather_service import WeatherData


# ── Risk level ────────────────────────────────────────────────────────────────

def _score_to_level(score: float) -> str:
    if score >= settings.RISK_THRESHOLD_HIGH:   return "CRITICAL"
    if score >= settings.RISK_THRESHOLD_MEDIUM: return "HIGH"
    if score >= settings.RISK_THRESHOLD_LOW:    return "MEDIUM"
    return "LOW"


_LEVEL_EMOJI = {
    "LOW":      "🟢",
    "MEDIUM":   "🟡",
    "HIGH":     "🟠",
    "CRITICAL": "🔴",
}


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class RiskAssessment:
    # Scores (0–100 each)
    vision_score:  float
    weather_score: float
    speed_score:   float
    total_score:   float

    # Derived
    risk_level:      str                  # LOW | MEDIUM | HIGH | CRITICAL
    hazard_summary:  str                  # "2× person, 1× pothole"
    recommendations: List[str]           = field(default_factory=list)
    alerts:          List[dict]          = field(default_factory=list)
    weather_info:    Optional[dict]      = None

    @property
    def emoji(self) -> str:
        return _LEVEL_EMOJI.get(self.risk_level, "⚪")

    @property
    def colour(self) -> str:
        return {
            "LOW":      "#2ecc71",
            "MEDIUM":   "#f39c12",
            "HIGH":     "#e67e22",
            "CRITICAL": "#e74c3c",
        }.get(self.risk_level, "#95a5a6")

    def to_dict(self) -> dict:
        return {
            "vision_score":   round(self.vision_score, 2),
            "weather_score":  round(self.weather_score, 2),
            "speed_score":    round(self.speed_score, 2),
            "total_score":    round(self.total_score, 2),
            "risk_level":     self.risk_level,
            "hazard_summary": self.hazard_summary,
            "recommendations":self.recommendations,
            "alerts":         self.alerts,
        }


# ══════════════════════════════════════════════════════════════════════════════
# RISK ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class RiskEngine:

    # ── Public ────────────────────────────────────────────────────────────────

    def assess(
        self,
        frame_analysis: FrameAnalysis,
        weather:        WeatherData,
        speed_kmh:      float = 0.0,
    ) -> RiskAssessment:
        """
        Compute a unified RiskAssessment from vision + weather + speed.
        """
        vision_score  = frame_analysis.vision_score
        weather_score = weather.weather_risk
        speed_score   = self.speed_risk(speed_kmh)

        total = (
            vision_score  * settings.RISK_WEIGHT_VISION
            + weather_score * settings.RISK_WEIGHT_WEATHER
            + speed_score   * settings.RISK_WEIGHT_SPEED
        )
        total = round(min(total, 100.0), 2)

        level   = _score_to_level(total)
        summary = frame_analysis.hazard_summary
        recs    = self._recommendations(
            level, frame_analysis, weather, speed_kmh
        )
        alerts  = self._alerts(level, frame_analysis, weather, speed_kmh)

        return RiskAssessment(
            vision_score  = vision_score,
            weather_score = weather_score,
            speed_score   = speed_score,
            total_score   = total,
            risk_level    = level,
            hazard_summary= summary,
            recommendations=recs,
            alerts        = alerts,
            weather_info  = weather.to_dict(),
        )

    @staticmethod
    def speed_risk(speed_kmh: float) -> float:
        """
        Map speed to a 0–100 risk contribution.
          0 km/h  →  0
          60 km/h →  40 (city speed)
          100 km/h → 70 (highway, still safe)
          140+ km/h → 100
        """
        if speed_kmh <= 0:    return 0.0
        if speed_kmh <= 60:   return round(speed_kmh / 60 * 40, 2)
        if speed_kmh <= 100:  return round(40 + (speed_kmh - 60) / 40 * 30, 2)
        return round(min(70 + (speed_kmh - 100) / 40 * 30, 100.0), 2)

    def assess_batch(
        self,
        analyses: List[FrameAnalysis],
        weather:  WeatherData,
        speed_kmh:float = 0.0,
    ) -> List[RiskAssessment]:
        """Assess a list of frames (e.g. video frames)."""
        return [self.assess(a, weather, speed_kmh) for a in analyses]

    # ── Recommendations ───────────────────────────────────────────────────────

    def _recommendations(
        self,
        level:    str,
        analysis: FrameAnalysis,
        weather:  WeatherData,
        speed:    float,
    ) -> List[str]:
        recs: List[str] = []
        labels = {d.label for d in analysis.detections}

        # Vision-based
        if "person" in labels or "bicycle" in labels:
            recs.append("🚶 Pedestrians or cyclists detected — reduce speed and give way.")
        if "pothole" in labels:
            recs.append("🕳️ Pothole ahead — slow down and steer gently.")
        if "dog" in labels or "cat" in labels or "cow" in labels or "horse" in labels:
            recs.append("🐄 Animal on road — brake gradually, avoid sudden swerves.")
        if "traffic light" in labels:
            recs.append("🚦 Traffic signal ahead — prepare to stop.")
        if "truck" in labels or "bus" in labels:
            recs.append("🚛 Large vehicle nearby — increase following distance.")
        if "cone" in labels or "barrier" in labels:
            recs.append("🚧 Road work or barriers — merge carefully.")

        # Weather-based
        if weather.visibility < 1000:
            recs.append(f"🌫️ Low visibility ({weather.visibility_km} km) — use fog lights and slow down.")
        if weather.rain_1h > 5:
            recs.append("🌧️ Heavy rain — increase following distance, avoid sudden braking.")
        if weather.wind_speed > 15:
            recs.append("💨 Strong winds — grip the wheel firmly, watch for debris.")
        if 200 <= weather.condition_code <= 299:
            recs.append("⛈️ Thunderstorm — pull over safely if conditions worsen.")

        # Speed-based
        if speed > 120:
            recs.append("⚡ Excessive speed detected — reduce to safe limits immediately.")
        elif speed > 80 and level in ("HIGH", "CRITICAL"):
            recs.append("🔻 Reduce speed — road conditions do not support current speed.")

        # Level-based catch-all
        if level == "CRITICAL" and not recs:
            recs.append("🔴 Critical risk — reduce speed, increase alertness, consider stopping.")
        elif level == "HIGH" and not recs:
            recs.append("🟠 High risk — drive with extreme caution.")
        elif level == "LOW" and not recs:
            recs.append("🟢 Conditions look safe — maintain awareness.")

        return recs

    # ── Alerts ────────────────────────────────────────────────────────────────

    def _alerts(
        self,
        level:    str,
        analysis: FrameAnalysis,
        weather:  WeatherData,
        speed:    float,
    ) -> List[dict]:
        """
        Priority-ordered alert list for the dashboard alert panel.
        Each alert: {priority, type, message}
        """
        alerts: List[dict] = []

        def add(priority: int, atype: str, msg: str):
            alerts.append({"priority": priority, "type": atype, "message": msg})

        if level == "CRITICAL":
            add(1, "CRITICAL", "⛔ CRITICAL RISK — Immediate driver action required!")
        elif level == "HIGH":
            add(2, "HIGH", "🟠 High hazard level detected on this route.")

        labels = {d.label for d in analysis.detections}

        if "person" in labels:
            add(1, "PEDESTRIAN", "⚠️ Pedestrian in roadway detected.")
        if "pothole" in labels:
            add(2, "ROAD_DEFECT", "🕳️ Pothole detected on current path.")
        if weather.visibility < 500:
            add(1, "VISIBILITY", f"🌫️ Dangerously low visibility: {weather.visibility} m.")
        if 200 <= weather.condition_code <= 299:
            add(1, "WEATHER", "⛈️ Thunderstorm conditions — extreme caution.")
        if speed > 130:
            add(1, "SPEED", f"🚨 Speed {speed:.0f} km/h exceeds safe limit.")

        alerts.sort(key=lambda x: x["priority"])
        return alerts
