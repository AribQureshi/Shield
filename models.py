"""
models.py — SHIELD SQLAlchemy ORM Models
=========================================
Tables:
  • TripSession      — one driving session (upload or live)
  • HazardDetection  — each YOLO-detected hazard in a session
  • WeatherSnapshot  — weather reading per session
  • RiskEvent        — computed risk score per frame/session
  • ChatMessage      — RAG conversation history
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer,
    String, Text, ForeignKey, JSON,
    Enum as SAEnum,
)
from sqlalchemy.orm import relationship

from db import Base


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


# ── Enums ─────────────────────────────────────────────────────────────────────

class SessionMode(str, enum.Enum):
    UPLOAD = "upload"     # user uploaded image/video
    LIVE   = "live"       # webcam stream
    DEMO   = "demo"       # demo mode (sample images)


class RiskLevel(str, enum.Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


# ── Trip Session ──────────────────────────────────────────────────────────────

class TripSession(Base):
    __tablename__ = "trip_sessions"

    id           = Column(Integer, primary_key=True, index=True)
    session_name = Column(String(128), nullable=False)
    mode         = Column(SAEnum(SessionMode), default=SessionMode.UPLOAD)
    city         = Column(String(64),  nullable=True)
    source_file  = Column(String(512), nullable=True)   # path to uploaded file
    frame_count  = Column(Integer, default=0)
    duration_sec = Column(Float,   nullable=True)
    avg_risk_score = Column(Float, nullable=True)
    peak_risk_score= Column(Float, nullable=True)
    risk_level   = Column(SAEnum(RiskLevel), nullable=True)
    is_completed = Column(Boolean, default=False)
    started_at   = Column(DateTime(timezone=True), default=_now)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    meta         = Column(JSON, nullable=True)

    # relationships
    detections      = relationship("HazardDetection", back_populates="session", cascade="all, delete-orphan")
    weather_records = relationship("WeatherSnapshot",  back_populates="session", cascade="all, delete-orphan")
    risk_events     = relationship("RiskEvent",        back_populates="session", cascade="all, delete-orphan")
    chat_messages   = relationship("ChatMessage",      back_populates="session", cascade="all, delete-orphan")


# ── Hazard Detection ──────────────────────────────────────────────────────────

class HazardDetection(Base):
    __tablename__ = "hazard_detections"

    id           = Column(Integer, primary_key=True, index=True)
    session_id   = Column(Integer, ForeignKey("trip_sessions.id"), nullable=False)
    frame_id     = Column(Integer, default=0)
    label        = Column(String(64),  nullable=False, index=True)
    confidence   = Column(Float,       nullable=False)
    danger_weight= Column(Float,       nullable=True)    # from config danger table
    bbox_x1      = Column(Integer,     nullable=True)
    bbox_y1      = Column(Integer,     nullable=True)
    bbox_x2      = Column(Integer,     nullable=True)
    bbox_y2      = Column(Integer,     nullable=True)
    created_at   = Column(DateTime(timezone=True), default=_now)

    session = relationship("TripSession", back_populates="detections")


# ── Weather Snapshot ──────────────────────────────────────────────────────────

class WeatherSnapshot(Base):
    __tablename__ = "weather_snapshots"

    id             = Column(Integer, primary_key=True, index=True)
    session_id     = Column(Integer, ForeignKey("trip_sessions.id"), nullable=False)
    city           = Column(String(64),  nullable=True)
    temperature    = Column(Float,       nullable=True)   # °C
    humidity       = Column(Integer,     nullable=True)   # %
    wind_speed     = Column(Float,       nullable=True)   # m/s
    visibility     = Column(Integer,     nullable=True)   # metres
    condition      = Column(String(64),  nullable=True)   # "Rain", "Fog", etc.
    condition_code = Column(Integer,     nullable=True)   # OWM condition id
    weather_risk   = Column(Float,       nullable=True)   # 0–100 computed score
    raw_response   = Column(JSON,        nullable=True)   # full OWM payload
    fetched_at     = Column(DateTime(timezone=True), default=_now)

    session = relationship("TripSession", back_populates="weather_records")


# ── Risk Event ────────────────────────────────────────────────────────────────

class RiskEvent(Base):
    __tablename__ = "risk_events"

    id              = Column(Integer, primary_key=True, index=True)
    session_id      = Column(Integer, ForeignKey("trip_sessions.id"), nullable=False)
    frame_id        = Column(Integer, default=0)
    vision_score    = Column(Float, nullable=True)    # 0–100
    weather_score   = Column(Float, nullable=True)    # 0–100
    speed_score     = Column(Float, nullable=True)    # 0–100
    total_score     = Column(Float, nullable=False)   # 0–100 weighted
    risk_level      = Column(SAEnum(RiskLevel), nullable=False)
    hazard_summary  = Column(Text,  nullable=True)    # "2× person, 1× pothole"
    recommendation  = Column(Text,  nullable=True)    # LLM/rule-based advice
    created_at      = Column(DateTime(timezone=True), default=_now)

    session = relationship("TripSession", back_populates="risk_events")


# ── Chat Message ──────────────────────────────────────────────────────────────

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id           = Column(Integer, primary_key=True, index=True)
    session_id   = Column(Integer, ForeignKey("trip_sessions.id"), nullable=True)
    role         = Column(String(16), nullable=False)    # "user" | "assistant"
    content      = Column(Text,       nullable=False)
    chunks_used  = Column(Integer,    nullable=True)
    latency_ms   = Column(Integer,    nullable=True)
    created_at   = Column(DateTime(timezone=True), default=_now)

    session = relationship("TripSession", back_populates="chat_messages")
