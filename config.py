"""
config.py — SHIELD Centralised Settings
========================================
All environment variables, thresholds, API keys, and model paths live here.
Copy .env.example to .env and fill in your values before running.
"""

from __future__ import annotations

import secrets
from enum import Enum
from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
import os

load_dotenv()

class Environment(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION  = "production"


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file          = ".env",
        env_file_encoding = "utf-8",
        case_sensitive    = False,
        extra             = "ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    APP_NAME:    str         = "SHIELD"
    APP_VERSION: str         = "1.0.0"
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")    # ── Database (SQLite default — zero config for MTech demo) ───────────────
    DATABASE_URL: str = "sqlite:///./shield.db"

    # ── OpenAI (RAG chatbot) ──────────────────────────────────────────────────

    OPENAI_MODEL:       str   = "llama-3.3-70b-versatile"
    EMBED_MODEL: str = "all-MiniLM-L6-v2"    
    OPENAI_MAX_TOKENS:  int   = 1024
    OPENAI_TEMPERATURE: float = 0.2

    # ── Weather API (OpenWeatherMap — free tier works) ────────────────────────
    WEATHER_API_KEY:  str = Field(default="", description="OpenWeatherMap API key")
    WEATHER_BASE_URL: str = "https://api.openweathermap.org/data/2.5"
    WEATHER_CITY:     str = "Mumbai"          # default city; overridable from UI
    WEATHER_UNITS:    str = "metric"          # metric | imperial

    # ── YOLO ──────────────────────────────────────────────────────────────────
    YOLO_MODEL_PATH:     str   = "yolov8n.pt"   # auto-downloaded on first run
    YOLO_CONFIDENCE:     float = 0.40
    YOLO_IOU_THRESHOLD:  float = 0.45
    YOLO_MAX_DETECTIONS: int   = 50
    YOLO_DEVICE:         str   = "cpu"           # "cpu" | "cuda" | "mps"

    # Hazard class names from COCO that SHIELD treats as road hazards
    # (YOLOv8 COCO labels)
    HAZARD_CLASSES: str = (
        "person,bicycle,car,motorcycle,bus,truck,"
        "traffic light,stop sign,dog,cat,cow,horse,"
        "pothole,cone,barrier"
    )

    # ── Risk Engine ───────────────────────────────────────────────────────────
    RISK_WEIGHT_VISION:  float = 0.50   # weight of YOLO hazard score
    RISK_WEIGHT_WEATHER: float = 0.30   # weight of weather risk
    RISK_WEIGHT_SPEED:   float = 0.20   # weight of speed risk

    RISK_THRESHOLD_LOW:    float = 30.0
    RISK_THRESHOLD_MEDIUM: float = 60.0
    RISK_THRESHOLD_HIGH:   float = 80.0

    # Per-class danger weights (higher = more dangerous)
    DANGER_WEIGHTS: str = (
        "person:0.9,bicycle:0.6,car:0.5,motorcycle:0.65,"
        "bus:0.55,truck:0.55,traffic light:0.3,stop sign:0.3,"
        "dog:0.5,cat:0.4,cow:0.75,horse:0.75,"
        "pothole:0.8,cone:0.4,barrier:0.5"
    )

    # ── ETL ───────────────────────────────────────────────────────────────────
    ETL_UPLOAD_DIR:    str = "data/uploads"
    ETL_PROCESSED_DIR: str = "data/processed"
    ETL_REPORTS_DIR:   str = "data/reports"

    # ── RAG / ChromaDB ────────────────────────────────────────────────────────
    RAG_CHUNK_SIZE:     int   = 200
    RAG_CHUNK_OVERLAP:  int   = 40
    RAG_TOP_K:          int   = 5
    RAG_SIM_THRESHOLD:  float = 0.40
    RAG_HISTORY_TURNS:  int   = 6
    CHROMA_PERSIST_DIR: str   = "data/chroma"

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL:    str = "INFO"
    LOG_DIR:      str = "logs"
    APP_LOG_FILE: str = "logs/shield.log"

    # ── Streamlit ─────────────────────────────────────────────────────────────
    STREAMLIT_THEME:       str = "dark"
    MAX_UPLOAD_SIZE_MB:    int = 50
    LIVE_FRAME_RATE:       int = 10
    LIVE_JPEG_QUALITY:     int = 75
    WEBCAM_INDEX:          int = 0

    # ── Validators ────────────────────────────────────────────────────────────

   
    # ── Computed Properties ───────────────────────────────────────────────────

    @property
    def hazard_classes_list(self) -> List[str]:
        return [c.strip() for c in self.HAZARD_CLASSES.split(",") if c.strip()]

    @property
    def danger_weights_dict(self) -> dict[str, float]:
        result = {}
        for pair in self.DANGER_WEIGHTS.split(","):
            pair = pair.strip()
            if ":" in pair:
                k, v = pair.split(":", 1)
                try:
                    result[k.strip()] = float(v.strip())
                except ValueError:
                    pass
        return result

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == Environment.DEVELOPMENT

    @property
    def chroma_in_memory(self) -> bool:
        return self.CHROMA_PERSIST_DIR.strip() == ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings: Settings = get_settings()
