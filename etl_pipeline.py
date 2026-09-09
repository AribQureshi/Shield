"""
etl_pipeline.py — SHIELD ETL Pipeline (v2)
===========================================
KEY CHANGE vs v1:
  process_image() now calls:
      self._weather.get_weather_for_image(image, city)   ← FUSED
  instead of:
      self._weather.get_weather(city)                    ← API-only (WRONG)

  process_video() does the same per-frame, with a single API call cached
  and image analysis run fresh per sampled frame.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from config import settings
from db import get_session
from models import (
    HazardDetection, RiskEvent, SessionMode,
    TripSession, WeatherSnapshot, RiskLevel,
)
from risk_engine import RiskAssessment, RiskEngine
from weather_service import WeatherData, WeatherService
from yolo_detector import FrameAnalysis, YOLODetector


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


# ── ETL Result ────────────────────────────────────────────────────────────────

@dataclass
class ETLResult:
    session_id:       int
    session_name:     str
    frames_processed: int                = 0
    hazards_found:    int                = 0
    avg_risk:         float              = 0.0
    peak_risk:        float              = 0.0
    risk_level:       str                = "LOW"
    weather:          Optional[dict]     = None
    duration_sec:     float              = 0.0
    report_text:      str                = ""
    errors:           List[str]          = field(default_factory=list)
    frame_analyses:   List[FrameAnalysis]  = field(default_factory=list)
    assessments:      List[RiskAssessment] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# ETL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class ETLPipeline:

    def __init__(
        self,
        yolo:    Optional[YOLODetector]   = None,
        weather: Optional[WeatherService] = None,
        risk:    Optional[RiskEngine]     = None,
    ):
        self._yolo    = yolo    or YOLODetector()
        self._weather = weather or WeatherService()
        self._risk    = risk    or RiskEngine()

        for d in [settings.ETL_UPLOAD_DIR, settings.ETL_PROCESSED_DIR, settings.ETL_REPORTS_DIR]:
            os.makedirs(d, exist_ok=True)

    # ── Process single image ──────────────────────────────────────────────────

    def process_image(
        self,
        image_bytes:  bytes,
        filename:     str   = "upload.jpg",
        city:         str   = "",
        speed_kmh:    float = 0.0,
        session_name: str   = "",
    ) -> ETLResult:
        t0   = time.perf_counter()
        city = city or settings.WEATHER_CITY

        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return ETLResult(session_id=-1, session_name="error",
                             errors=["Could not decode image."])

        saved_path = self._save_upload(image_bytes, filename)

        # ── YOLO detection ────────────────────────────────────────────────────
        analysis = self._yolo.detect(image, frame_id=1)

        # ── FUSED weather (image + API) ───────────────────────────────────────
        # THIS IS THE KEY FIX: pass the actual image so weather cues are
        # extracted from pixels, not just fetched from the API for the city.
        weather_data = self._weather.get_weather_for_image(image, city)

        # ── Risk assessment ───────────────────────────────────────────────────
        assessment = self._risk.assess(analysis, weather_data, speed_kmh)

        name = session_name or f"trip_{Path(filename).stem}_{uuid.uuid4().hex[:6]}"

        session_id = self._persist(
            session_name = name,
            mode         = SessionMode.UPLOAD,
            city         = city,
            source_file  = str(saved_path),
            analyses     = [analysis],
            weather      = weather_data,
            assessments  = [assessment],
        )

        report = self._build_report(name, [analysis], [assessment], weather_data)
        self._save_report(report, name)

        return ETLResult(
            session_id       = session_id,
            session_name     = name,
            frames_processed = 1,
            hazards_found    = len(analysis.detections),
            avg_risk         = assessment.total_score,
            peak_risk        = assessment.total_score,
            risk_level       = assessment.risk_level,
            weather          = weather_data.to_dict(),
            duration_sec     = round(time.perf_counter() - t0, 3),
            report_text      = report,
            frame_analyses   = [analysis],
            assessments      = [assessment],
        )

    # ── Process video ─────────────────────────────────────────────────────────

    def process_video(
        self,
        video_bytes:  bytes,
        filename:     str   = "upload.mp4",
        city:         str   = "",
        speed_kmh:    float = 0.0,
        session_name: str   = "",
        sample_every: int   = 10,
        max_frames:   int   = 100,
    ) -> ETLResult:
        t0   = time.perf_counter()
        city = city or settings.WEATHER_CITY

        tmp_path = Path(settings.ETL_UPLOAD_DIR) / f"tmp_{uuid.uuid4().hex}.mp4"
        tmp_path.write_bytes(video_bytes)

        cap = cv2.VideoCapture(str(tmp_path))
        if not cap.isOpened():
            tmp_path.unlink(missing_ok=True)
            return ETLResult(session_id=-1, session_name="error",
                             errors=["Could not open video file."])

        analyses:    List[FrameAnalysis]  = []
        assessments: List[RiskAssessment] = []

        # Fetch API weather once for the city (cached), then fuse with each frame
        frame_idx = 0
        while len(analyses) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % sample_every != 0:
                continue

            analysis = self._yolo.detect(frame, frame_id=frame_idx)

            # FUSED weather per sampled frame — image analysis is cheap (pure numpy/cv2)
            weather_data = self._weather.get_weather_for_image(frame, city)

            assessment   = self._risk.assess(analysis, weather_data, speed_kmh)
            analyses.append(analysis)
            assessments.append(assessment)

        cap.release()
        tmp_path.unlink(missing_ok=True)

        if not analyses:
            return ETLResult(session_id=-1, session_name="error",
                             errors=["No frames extracted from video."])

        # Use last frame's weather as the session representative
        final_weather = self._weather.get_weather_for_image(
            np.zeros((64, 64, 3), dtype=np.uint8), city  # dummy image → API-only fallback
        ) if not analyses else self._weather.get_weather(city)

        scores    = [a.total_score for a in assessments]
        avg_risk  = round(sum(scores) / len(scores), 2)
        peak_risk = round(max(scores), 2)
        top_level = max(
            (a.risk_level for a in assessments),
            key=lambda l: ["LOW", "MEDIUM", "HIGH", "CRITICAL"].index(l),
            default="LOW",
        )

        name = session_name or f"trip_{Path(filename).stem}_{uuid.uuid4().hex[:6]}"

        session_id = self._persist(
            session_name = name,
            mode         = SessionMode.UPLOAD,
            city         = city,
            source_file  = filename,
            analyses     = analyses,
            weather      = final_weather,
            assessments  = assessments,
        )

        report = self._build_report(name, analyses, assessments, final_weather)
        self._save_report(report, name)

        return ETLResult(
            session_id       = session_id,
            session_name     = name,
            frames_processed = len(analyses),
            hazards_found    = sum(len(a.detections) for a in analyses),
            avg_risk         = avg_risk,
            peak_risk        = peak_risk,
            risk_level       = top_level,
            weather          = final_weather.to_dict(),
            duration_sec     = round(time.perf_counter() - t0, 3),
            report_text      = report,
            frame_analyses   = analyses,
            assessments      = assessments,
        )

    # ── Persist to DB ─────────────────────────────────────────────────────────

    def _persist(
        self,
        session_name: str,
        mode:         SessionMode,
        city:         str,
        source_file:  str,
        analyses:     List[FrameAnalysis],
        weather:      WeatherData,
        assessments:  List[RiskAssessment],
    ) -> int:
        scores    = [a.total_score for a in assessments]
        avg_risk  = round(sum(scores) / len(scores), 2) if scores else 0.0
        peak_risk = round(max(scores), 2) if scores else 0.0
        top_level = max(
            (a.risk_level for a in assessments),
            key=lambda l: ["LOW", "MEDIUM", "HIGH", "CRITICAL"].index(l),
            default="LOW",
        )

        with get_session() as db:
            session = TripSession(
                session_name     = session_name,
                mode             = mode,
                city             = city,
                source_file      = source_file,
                frame_count      = len(analyses),
                avg_risk_score   = avg_risk,
                peak_risk_score  = peak_risk,
                risk_level       = RiskLevel(top_level),
                is_completed     = True,
                completed_at     = _now(),
            )
            db.add(session)
            db.flush()

            db.add(WeatherSnapshot(
                session_id     = session.id,
                city           = weather.city,
                temperature    = weather.temperature,
                humidity       = weather.humidity,
                wind_speed     = weather.wind_speed,
                visibility     = weather.visibility,
                condition      = weather.condition,
                condition_code = weather.condition_code,
                weather_risk   = weather.weather_risk,
                raw_response   = weather.raw,
            ))

            for analysis, assessment in zip(analyses, assessments):
                for det in analysis.detections:
                    db.add(HazardDetection(
                        session_id    = session.id,
                        frame_id      = analysis.frame_id,
                        label         = det.label,
                        confidence    = det.confidence,
                        danger_weight = det.danger_weight,
                        bbox_x1=det.bbox[0], bbox_y1=det.bbox[1],
                        bbox_x2=det.bbox[2], bbox_y2=det.bbox[3],
                    ))
                db.add(RiskEvent(
                    session_id     = session.id,
                    frame_id       = analysis.frame_id,
                    vision_score   = assessment.vision_score,
                    weather_score  = assessment.weather_score,
                    speed_score    = assessment.speed_score,
                    total_score    = assessment.total_score,
                    risk_level     = RiskLevel(assessment.risk_level),
                    hazard_summary = assessment.hazard_summary,
                    recommendation = " | ".join(assessment.recommendations),
                ))

            db.commit()
            return session.id

    # ── Report ────────────────────────────────────────────────────────────────

    def _build_report(
        self,
        name:        str,
        analyses:    List[FrameAnalysis],
        assessments: List[RiskAssessment],
        weather:     WeatherData,
    ) -> str:
        scores   = [a.total_score for a in assessments]
        avg_risk = round(sum(scores) / len(scores), 2) if scores else 0.0
        peak     = round(max(scores), 2) if scores else 0.0

        label_counts: dict[str, int] = {}
        for a in analyses:
            for d in a.detections:
                label_counts[d.label] = label_counts.get(d.label, 0) + 1

        top_recs, seen = [], set()
        for a in assessments:
            for r in a.recommendations:
                if r not in seen:
                    top_recs.append(r); seen.add(r)
                if len(top_recs) >= 5:
                    break
            if len(top_recs) >= 5:
                break

        # Include image-analysis breakdown if available
        img_section = ""
        if weather.image_analysis:
            ia = weather.image_analysis
            img_section = f"""
IMAGE WEATHER ANALYSIS (from pixel data)
  Rain score      : {ia.rain_score}/100
  Fog score       : {ia.fog_score}/100
  Darkness score  : {ia.darkness_score}/100
  Glare score     : {ia.glare_score}/100
  Combined        : {ia.combined_score}/100
  Detected cond.  : {ia.condition_label}
  Est. visibility : {round(ia.visibility_est/1000,1)} km
"""

        return f"""
SHIELD TRIP REPORT — {name}
Generated: {_now().isoformat()}

TRIP SUMMARY
  Frames analysed : {len(analyses)}
  Average risk    : {avg_risk}/100
  Peak risk       : {peak}/100
  Overall level   : {assessments[-1].risk_level if assessments else 'N/A'}

WEATHER CONDITIONS (FUSED — image + API)
  City        : {weather.city}
  Condition   : {weather.condition} ({weather.description})
  Temperature : {weather.temperature}°C
  Visibility  : {weather.visibility_km} km
  Wind speed  : {weather.wind_speed} m/s
  Rain (1h)   : {weather.rain_1h} mm
  Weather risk: {weather.weather_risk}/100
{img_section}
HAZARDS DETECTED
{chr(10).join(f'  {k}: {v} occurrence(s)' for k, v in sorted(label_counts.items())) or '  None detected'}

SAFETY RECOMMENDATIONS
{chr(10).join(f'  {r}' for r in top_recs) or '  Drive safely — conditions look good.'}

RISK SCORE BREAKDOWN
  Vision  : {assessments[-1].vision_score if assessments else 0}/100
  Weather : {assessments[-1].weather_score if assessments else 0}/100
  Speed   : {assessments[-1].speed_score if assessments else 0}/100
  Total   : {assessments[-1].total_score if assessments else 0}/100
""".strip()

    def _save_report(self, report: str, name: str) -> None:
        Path(settings.ETL_REPORTS_DIR).mkdir(parents=True, exist_ok=True)
        (Path(settings.ETL_REPORTS_DIR) / f"{name}.txt").write_text(report, encoding="utf-8")

    def _save_upload(self, data: bytes, filename: str) -> Path:
        Path(settings.ETL_UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        path = Path(settings.ETL_UPLOAD_DIR) / f"{uuid.uuid4().hex}_{filename}"
        path.write_bytes(data)
        return path