from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml

from sociomancy.analysis.models import ONNXSignalProvider, NullONNXProvider

# ============================================================
# Data structures
# ============================================================

@dataclass
class Signal:
    category: str
    source: str
    score: float
    matched_text: str | None = None
    pattern: str | None = None


@dataclass
class AnalysisResult:
    score: float
    classification: str
    signals: list[Signal] = field(default_factory=list)

    @property
    def is_parasocial(self) -> bool:
        return self.score >= 0.40

    def categories(self) -> list[str]:
        return sorted(set(signal.category for signal in self.signals))

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "classification": self.classification,
            "is_parasocial": self.is_parasocial,
            "categories": self.categories(),
            "signals": [
                {
                    "category": signal.category,
                    "source": signal.source,
                    "score": round(signal.score, 4),
                    "matched_text": signal.matched_text,
                    "pattern": signal.pattern,
                }
                for signal in self.signals
            ],
        }


# ============================================================
# Configuration
# ============================================================

class ParasocialConfig:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Parasocial configuration not found: {self.path}")

        with self.path.open("r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f)

        self.categories = self._compile_categories(self.data.get("categories", {}))
        self.linguistic_signals = self._compile_categories(self.data.get("linguistic_signals", {}))
        self.negative_patterns = self._compile_categories(self.data.get("negative_patterns", {}))
        self.thresholds = self.data.get(
            "thresholds",
            {"weak": 0.20, "moderate": 0.40, "strong": 0.65, "very_strong": 0.85},
        )

    @staticmethod
    def _compile_categories(categories: dict[str, dict]) -> dict[str, dict]:
        compiled = {}
        for name, config in categories.items():
            patterns = config.get("patterns", [])
            compiled[name] = {
                "weight": float(config.get("weight", 1.0)),
                "description": config.get("description", ""),
                "patterns": [re.compile(p, re.IGNORECASE) for p in patterns],
            }
        return compiled


# ============================================================
# Detector
# ============================================================

class ParasocialDetector:
    def __init__(
        self,
        config: ParasocialConfig,
        onnx_provider: ONNXSignalProvider | None = None,
    ):
        self.config = config
        self.onnx_provider = onnx_provider or NullONNXProvider()

    def analyze(self, text: str, emotion_data: dict | None = None) -> AnalysisResult:
        if not isinstance(text, str) or not text.strip():
            return AnalysisResult(score=0.0, classification="none")

        norm_text = self._normalize(text)
        signals: list[Signal] = []

        # 1. Regex Category Triggers
        for category, config in self.config.categories.items():
            for pattern in config["patterns"]:
                match = pattern.search(norm_text)
                if match:
                    signals.append(
                        Signal(
                            category=category,
                            source="regex",
                            score=config["weight"],
                            matched_text=match.group(0),
                            pattern=pattern.pattern,
                        )
                    )
                    break

        # 2. Linguistic Context
        for category, config in self.config.linguistic_signals.items():
            for pattern in config["patterns"]:
                if pattern.search(norm_text):
                    signals.append(
                        Signal(
                            category=category,
                            source="linguistic",
                            score=config["weight"],
                        )
                    )
                    break

        # 3. Negative Evidence Suppression
        for category, config in self.config.negative_patterns.items():
            for pattern in config["patterns"]:
                match = pattern.search(norm_text)
                if match:
                    signals.append(
                        Signal(
                            category=category,
                            source="negative",
                            score=config["weight"],
                            matched_text=match.group(0),
                            pattern=pattern.pattern,
                        )
                    )
                    break

        # 4. ONNX Model Signals
        onnx_res = self.onnx_provider.analyze(norm_text, emotion_data)
        for sig in onnx_res.get("signals", []):
            signals.append(
                Signal(
                    category=sig["category"],
                    source=sig["source"],
                    score=sig["score"],
                    matched_text=sig.get("matched_text"),
                )
            )

        # 5. Score Calculation
        score = self._calculate_score(signals)
        classification = self._classify(score)

        return AnalysisResult(score=score, classification=classification, signals=signals)

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    def _calculate_score(self, signals: list[Signal]) -> float:
        if not signals:
            return 0.0

        positive = sum(s.score for s in signals if s.score > 0)
        negative = sum(s.score for s in signals if s.score < 0)
        raw_score = max(0.0, positive + negative)

        # Saturating function: 1 - (1 / (1 + raw_score))
        return min(max(1.0 - (1.0 / (1.0 + raw_score)), 0.0), 1.0)

    def _classify(self, score: float) -> str:
        if score >= self.config.thresholds["very_strong"]:
            return "very_strong"
        if score >= self.config.thresholds["strong"]:
            return "strong"
        if score >= self.config.thresholds["moderate"]:
            return "moderate"
        if score >= self.config.thresholds["weak"]:
            return "weak"
        return "none"