import datetime
import json
import os
from pathlib import Path

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DB_DIR = Path(os.environ.get("DB_DIR", Path(__file__).resolve().parent))
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "results.db"
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String, nullable=False)
    quality_score = Column(Float, nullable=False)
    quality_label = Column(String, nullable=False, default="UNKNOWN")
    predicted_distortion = Column(String, nullable=False)
    issues_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "quality_score": self.quality_score,
            "quality_label": self.quality_label,
            "predicted_distortion": self.predicted_distortion,
            "issues": json.loads(self.issues_json),
            "created_at": self.created_at.isoformat(),
        }


def init_db():
    Base.metadata.create_all(engine)
