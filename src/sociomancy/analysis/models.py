from __future__ import annotations
from typing import Any

class ONNXSignalProvider:
    """Interface for ONNX-powered parasocial signal extraction."""
    def analyze(self, text: str, emotion_data: dict | None = None) -> dict[str, Any]:
        return {}

class NullONNXProvider(ONNXSignalProvider):
    """Fallback provider when ONNX models are disabled."""
    def analyze(self, text: str, emotion_data: dict | None = None) -> dict[str, Any]:
        return {}

class SociomancyONNXSignalProvider(ONNXSignalProvider):
    """
    Leverages ONNX emotion predictions to feed high-conviction
    emotional signals into the parasocial scoring matrix.
    """
    def analyze(self, text: str, emotion_data: dict | None = None) -> dict[str, Any]:
        if not emotion_data:
            return {"signals": []}

        signals = []
        top_emotion = emotion_data.get("top_emotion", "")
        toxicity_score = emotion_data.get("toxicity_score", 0.0)

        # High love/gratitude combined with low toxicity indicates positive attachment
        if top_emotion in ["love", "gratitude", "admiration"] and toxicity_score < 0.10:
            signals.append({
                "category": "emotional_comfort",
                "source": "onnx_emotion",
                "score": 0.35,
                "matched_text": f"Dominant Emotion: {top_emotion}",
            })

        # Deep grief/sadness without hostility indicates heavy emotional reliance
        if top_emotion in ["sadness", "remorse"] and toxicity_score < 0.15:
            signals.append({
                "category": "emotional_reliance",
                "source": "onnx_emotion",
                "score": 0.40,
                "matched_text": f"Dominant Emotion: {top_emotion}",
            })

        return {"signals": signals}