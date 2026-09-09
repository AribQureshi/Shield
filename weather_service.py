"""
weather_service.py — SHIELD Weather Intelligence Service (v2 — Image-Aware)
============================================================================
ROOT FIX: Weather is now analysed from TWO sources and merged:

  SOURCE 1 — Image pixel analysis (weight 0.65)
    Detects rain streaks, fog haze, darkness, wet-road glare
    directly from the dashcam frame.  This is what is ACTUALLY
    happening in the scene — regardless of city/API.

  SOURCE 2 — OpenWeatherMap API (weight 0.35)
    Real-world context: temperature, wind, location-level conditions.

Result: rainy image in a "sunny" city → HIGH rain score.  ✅
        foggy image + foggy API        → CRITICAL.         ✅
        clear image + rainy API        → moderate risk.    ✅

Public API (unchanged from v1 so etl_pipeline / routes need only one line change):
  get_weather(city)                          → API-only (sidebar card)
  get_weather_for_image(image, city)         → FUSED  ← use this in pipeline
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np
import requests

from config import settings


# ── OWM condition-id → base risk ─────────────────────────────────────────────
_CONDITION_RISK: dict[tuple[int, int], float] = {
    (200, 299): 90.0,
    (300, 399): 40.0,
    (500, 501): 55.0,
    (502, 531): 80.0,
    (600, 699): 70.0,
    (700, 799): 75.0,
    (800, 800): 10.0,
    (801, 802): 15.0,
    (803, 804): 20.0,
}


def _condition_base_risk(code: int) -> float:
    for (lo, hi), risk in _CONDITION_RISK.items():
        if lo <= code <= hi:
            return risk
    return 20.0


# ── Image-analysis result dataclass ──────────────────────────────────────────

@dataclass
class ImageWeatherAnalysis:
    rain_score:      float = 0.0
    fog_score:       float = 0.0
    darkness_score:  float = 0.0
    glare_score:     float = 0.0
    combined_score:  float = 0.0
    condition_label: str   = "Clear"
    condition_code:  int   = 800
    visibility_est:  int   = 10000


# ── Main weather dataclass ────────────────────────────────────────────────────

@dataclass
class WeatherData:
    city:            str
    temperature:     float
    humidity:        int
    wind_speed:      float
    visibility:      int
    condition:       str
    condition_code:  int
    weather_risk:    float
    description:     str   = ""
    rain_1h:         float = 0.0
    raw:             dict  = field(default_factory=dict)
    image_analysis:  Optional[ImageWeatherAnalysis] = None
    image_weight:    float = 0.0

    @property
    def risk_label(self) -> str:
        if self.weather_risk >= 80: return "CRITICAL"
        if self.weather_risk >= 60: return "HIGH"
        if self.weather_risk >= 35: return "MEDIUM"
        return "LOW"

    @property
    def visibility_km(self) -> float:
        return round(self.visibility / 1000, 2)

    def to_dict(self) -> dict:
        d = {
            "city":           self.city,
            "temperature":    self.temperature,
            "humidity":       self.humidity,
            "wind_speed":     self.wind_speed,
            "visibility":     self.visibility,
            "condition":      self.condition,
            "condition_code": self.condition_code,
            "weather_risk":   round(self.weather_risk, 2),
            "rain_1h":        self.rain_1h,
            "description":    self.description,
        }
        if self.image_analysis:
            d["image_rain_score"] = round(self.image_analysis.rain_score, 1)
            d["image_fog_score"]  = round(self.image_analysis.fog_score, 1)
            d["image_condition"]  = self.image_analysis.condition_label
        return d

    @property
    def emoji(self) -> str:
        code = self.condition_code
        if 200 <= code <= 299: return "⛈️"
        if 300 <= code <= 399: return "🌦️"
        if 500 <= code <= 531: return "🌧️"
        if 600 <= code <= 699: return "❄️"
        if 700 <= code <= 799: return "🌫️"
        if code == 800:        return "☀️"
        return "⛅"


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE WEATHER ANALYSER
# Analyses weather conditions directly from pixel data.
# ══════════════════════════════════════════════════════════════════════════════

class ImageWeatherAnalyser:

    def analyse(self, image: np.ndarray) -> ImageWeatherAnalysis:
        if image is None or image.size == 0:
            return ImageWeatherAnalysis()

        # Resize for speed and normalisation
        h, w   = image.shape[:2]
        scale  = 640 / max(h, w)
        if scale < 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        rain_score     = self._detect_rain(gray, image)
        fog_score      = self._detect_fog(gray, hsv)
        darkness_score = self._detect_darkness(gray)
        glare_score    = self._detect_glare(hsv)

        combined = self._combine(rain_score, fog_score, darkness_score, glare_score)
        label, code, vis = self._classify(rain_score, fog_score, darkness_score, glare_score)

        return ImageWeatherAnalysis(
            rain_score      = round(rain_score, 1),
            fog_score       = round(fog_score, 1),
            darkness_score  = round(darkness_score, 1),
            glare_score     = round(glare_score, 1),
            combined_score  = round(combined, 1),
            condition_label = label,
            condition_code  = code,
            visibility_est  = vis,
        )

    # ── Rain ──────────────────────────────────────────────────────────────────
    def _detect_rain(self, gray: np.ndarray, bgr: np.ndarray) -> float:
        """
        Four cues:
        (a) Vertical streak ratio — rain streaks are vertical in dashcam
        (b) Mid-range Laplacian variance — rain adds mid-freq texture noise
        (c) Colour desaturation — rain washes out colours
        (d) Wet-road specular highlights in bottom third
        """
        # (a) Vertical vs horizontal gradient energy
        sobel_v = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_h = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        v_e = float(np.mean(np.abs(sobel_v)))
        h_e = float(np.mean(np.abs(sobel_h))) + 1e-6
        streak = float(np.clip((v_e / h_e - 1.0) / 0.5, 0, 1))

        # (b) Laplacian noise in mid-range — rain: ~200–1500 variance
        lap_var = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))
        noise   = float(np.clip((lap_var - 100) / 1400, 0, 1))

        # (c) HSV saturation drop
        hsv      = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mean_sat = float(np.mean(hsv[:, :, 1]))
        desat    = float(np.clip((80 - mean_sat) / 80, 0, 1))

        # (d) Bright specular in bottom 1/3 (wet road reflection)
        bottom      = gray[gray.shape[0] * 2 // 3:, :]
        bright_frac = float(np.mean(bottom > 180))
        wet_road    = float(np.clip(bright_frac / 0.40, 0, 1))

        score = streak * 0.35 + noise * 0.30 + desat * 0.20 + wet_road * 0.15
        return float(np.clip(score * 100, 0, 100))

    # ── Fog ───────────────────────────────────────────────────────────────────
    def _detect_fog(self, gray: np.ndarray, hsv: np.ndarray) -> float:
        """
        (a) Low contrast (small std-dev) — the hallmark of fog
        (b) High mean brightness — bright white/grey haze
        (c) Low saturation — fog desaturates
        (d) Depth gradient loss — upper horizon loses detail before lower road
        """
        # (a) Contrast
        std_dev   = float(np.std(gray))
        contrast  = float(np.clip(1.0 - std_dev / 60.0, 0, 1))

        # (b) Brightness
        mean_b    = float(np.mean(gray))
        bright    = float(np.clip((mean_b - 120) / 100, 0, 1))

        # (c) Saturation
        mean_sat  = float(np.mean(hsv[:, :, 1]))
        sat       = float(np.clip((60 - mean_sat) / 60, 0, 1))

        # (d) Upper vs lower contrast ratio
        h         = gray.shape[0]
        u_std     = float(np.std(gray[:h // 3, :]))
        l_std     = float(np.std(gray[h * 2 // 3:, :])) + 1e-6
        depth     = float(np.clip(1.0 - u_std / l_std, 0, 1))

        score = contrast * 0.40 + bright * 0.25 + sat * 0.20 + depth * 0.15
        return float(np.clip(score * 100, 0, 100))

    # ── Darkness ──────────────────────────────────────────────────────────────
    def _detect_darkness(self, gray: np.ndarray) -> float:
        """25th percentile avoids headlight-blown-mean bias."""
        p25   = float(np.percentile(gray, 25))
        score = float(np.clip(1.0 - p25 / 100.0, 0, 1))
        return score * 100

    # ── Glare ─────────────────────────────────────────────────────────────────
    def _detect_glare(self, hsv: np.ndarray) -> float:
        """High-V + low-S pixels = specular highlights / sun on wet road."""
        v = hsv[:, :, 2].astype(float)
        s = hsv[:, :, 1].astype(float)
        glare_frac = float(np.mean((v > 220) & (s < 40)))
        return float(np.clip(glare_frac / 0.15, 0, 1)) * 100

    # ── Combine ───────────────────────────────────────────────────────────────
    def _combine(self, rain: float, fog: float, dark: float, glare: float) -> float:
        """60% on dominant signal, 40% on weighted average."""
        avg = rain * 0.35 + fog * 0.30 + dark * 0.20 + glare * 0.15
        return float(np.clip(max(rain, fog, dark, glare) * 0.60 + avg * 0.40, 0, 100))

    # ── Classify ──────────────────────────────────────────────────────────────
    def _classify(
        self, rain: float, fog: float, dark: float, glare: float
    ) -> Tuple[str, int, int]:
        """Returns (label, synthetic_OWM_code, visibility_metres)."""
        if max(rain, fog, dark, glare) < 15:
            return "Clear", 800, 10000

        dominant = max(
            ("Rain",  rain,  0),
            ("Fog",   fog,   1),
            ("Dark",  dark,  2),
            ("Glare", glare, 3),
            key=lambda x: x[1],
        )[0]

        if dominant == "Rain":
            if rain > 70: return "Heavy Rain",  502, 1500
            if rain > 40: return "Rain",         500, 3000
            return "Drizzle", 300, 6000

        if dominant == "Fog":
            if fog > 70: return "Dense Fog", 741, 200
            if fog > 40: return "Fog",        701, 1000
            return "Mist", 701, 3000

        if dominant == "Dark":
            if dark > 70: return "Storm/Night", 211, 2000
            return "Overcast", 804, 8000

        return "Glare/Wet Road", 800, 9000


# ══════════════════════════════════════════════════════════════════════════════
# WEATHER SERVICE
# ══════════════════════════════════════════════════════════════════════════════

class WeatherService:

    _CACHE_TTL = 60

    def __init__(self, api_key: Optional[str] = None):
        self._api_key      = api_key or settings.WEATHER_API_KEY
        self._cache:       dict[str, tuple[WeatherData, float]] = {}
        self._img_analyser = ImageWeatherAnalyser()

    # ── API-only (sidebar / no-image contexts) ────────────────────────────────
    def get_weather(self, city: Optional[str] = None) -> WeatherData:
        city = (city or settings.WEATHER_CITY).strip()
        if city in self._cache:
            data, ts = self._cache[city]
            if time.time() - ts < self._CACHE_TTL:
                return data
        if not self._api_key:
            return self._fallback(city, reason="no_api_key")
        try:
            raw  = self._fetch_owm(city)
            data = self._parse(raw, city)
            self._cache[city] = (data, time.time())
            return data
        except Exception as exc:
            return self._fallback(city, reason=str(exc))

    # ── FUSED — use this in etl_pipeline.py ──────────────────────────────────
    def get_weather_for_image(
        self,
        image:        np.ndarray,
        city:         Optional[str] = None,
        api_weight:   float = 0.35,
        image_weight: float = 0.65,
    ) -> WeatherData:
        """
        PRIMARY method when a dashcam image is available.
        Blends image-pixel weather analysis (65%) with OWM API data (35%).

        Call from etl_pipeline.py:
            weather = self._weather.get_weather_for_image(image, city)
        instead of:
            weather = self._weather.get_weather(city)
        """
        img_analysis = self._img_analyser.analyse(image)
        api_data     = self.get_weather(city)

        # Blend risk
        blended_risk = round(
            img_analysis.combined_score * image_weight
            + api_data.weather_risk     * api_weight,
            2,
        )
        blended_risk = min(blended_risk, 100.0)

        # Condition: use image condition when it is at least as bad as API
        if img_analysis.combined_score >= api_data.weather_risk - 10:
            final_condition = img_analysis.condition_label
            final_code      = img_analysis.condition_code
            final_vis       = img_analysis.visibility_est
            final_desc      = f"Image-detected: {img_analysis.condition_label}"
        else:
            final_condition = api_data.condition
            final_code      = api_data.condition_code
            final_vis       = api_data.visibility
            final_desc      = api_data.description

        # Infer rain_1h from image when API shows 0 but image shows rain
        inferred_rain = api_data.rain_1h
        if img_analysis.rain_score > 40 and api_data.rain_1h < 0.5:
            inferred_rain = round((img_analysis.rain_score - 40) / 60 * 8, 1)

        return WeatherData(
            city           = api_data.city,
            temperature    = api_data.temperature,
            humidity       = api_data.humidity,
            wind_speed     = api_data.wind_speed,
            visibility     = final_vis,
            condition      = final_condition,
            condition_code = final_code,
            description    = final_desc,
            rain_1h        = inferred_rain,
            weather_risk   = blended_risk,
            raw            = api_data.raw,
            image_analysis = img_analysis,
            image_weight   = image_weight,
        )

    # ── Risk scorer ───────────────────────────────────────────────────────────
    def compute_weather_risk(self, data: WeatherData) -> float:
        c  = _condition_base_risk(data.condition_code)
        v  = max(0.0, min(100.0, (1 - data.visibility / 10_000) * 100))
        w  = min(data.wind_speed / 25 * 100, 100.0)
        r  = min(data.rain_1h   / 10 * 100, 100.0)
        return round(min(c * 0.40 + v * 0.30 + w * 0.20 + r * 0.10, 100.0), 2)

    # ── OWM fetch + parse ─────────────────────────────────────────────────────
    def _fetch_owm(self, city: str) -> dict:
        resp = requests.get(
            f"{settings.WEATHER_BASE_URL}/weather",
            params={"q": city, "appid": self._api_key, "units": settings.WEATHER_UNITS},
            timeout=6,
        )
        resp.raise_for_status()
        return resp.json()

    def _parse(self, raw: dict, city: str) -> WeatherData:
        main    = raw.get("main",    {})
        wind    = raw.get("wind",    {})
        weather = raw.get("weather", [{}])[0]
        rain    = raw.get("rain",    {})

        placeholder = WeatherData(
            city           = raw.get("name", city),
            temperature    = float(main.get("temp",      25.0)),
            humidity       = int(main.get("humidity",    60)),
            wind_speed     = float(wind.get("speed",     0.0)),
            visibility     = int(raw.get("visibility",   10_000)),
            condition      = weather.get("main",        "Clear"),
            condition_code = int(weather.get("id",       800)),
            description    = weather.get("description", "clear sky"),
            rain_1h        = float(rain.get("1h",        0.0)),
            weather_risk   = 0.0,
            raw            = raw,
        )
        placeholder.weather_risk = self.compute_weather_risk(placeholder)
        return placeholder

    def _fallback(self, city: str, reason: str = "") -> WeatherData:
        return WeatherData(
            city="unknown" if not city else city,
            temperature=25.0, humidity=60, wind_speed=2.0,
            visibility=10_000, condition="Unknown", condition_code=800,
            weather_risk=10.0,
            description=f"Weather unavailable ({reason})",
        )