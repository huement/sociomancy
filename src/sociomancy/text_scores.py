"""Shared ONNX scoring for the topic pipeline and study configs.

Toxicity and emotion come from SociomancyONNXPipeline. Sentiment is the
GoEmotions label collapsed into Positive / Neutral / Negative — the same
split Act 3 already uses for titles. Zero-shot labels come from
facebook/bart-large-mnli. Complaint hypotheses live in the study JSON.
"""

from __future__ import annotations

NEGATIVE_EMOTIONS = {"anger", "annoyance", "disapproval", "disgust", "fear", "sadness"}
POSITIVE_EMOTIONS = {"admiration", "excitement", "joy", "optimism", "curiosity", "approval"}

def sentiment_bucket(emotion: str, is_toxic: bool) -> str:
    """Collapse a GoEmotions label into the three-way sentiment used on the charts."""
    if is_toxic or emotion in NEGATIVE_EMOTIONS:
        return "Negative"
    if emotion in POSITIVE_EMOTIONS:
        return "Positive"
    return "Neutral"


def score_texts_with_sociomancy_onnx(texts: list[str]) -> tuple[list[float], list[bool], list[str]]:
    """Score texts with unitary/unbiased-toxic-roberta and GoEmotions."""
    from sociomancy.models_onnx import SociomancyONNXPipeline

    nlp = SociomancyONNXPipeline()
    clean_texts = [str(t or "").strip() or "neutral" for t in texts]
    evaluations = nlp.analyze_comments(clean_texts)

    tox_scores = [round(float(e["toxicity_score"]), 4) for e in evaluations]
    is_toxic = [bool(e["is_toxic"]) for e in evaluations]
    top_emotions = [
        e["top_emotions"][0]["label"] if e.get("top_emotions") else "neutral"
        for e in evaluations
    ]
    return tox_scores, is_toxic, top_emotions


def classify_zero_shot_onnx(
    texts: list[str],
    candidate_labels: list[str],
    log_every: int = 0,
) -> list[str]:
    """Label texts with the cached facebook/bart-large-mnli ONNX model."""
    from optimum.pipelines import pipeline

    zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", accelerator="ort")
    results = []
    total = len(texts)
    for index, txt in enumerate(texts, start=1):
        clean = str(txt or "").strip()[:350]
        if not clean:
            results.append("Unclassified")
        else:
            out = zs(clean, candidate_labels=candidate_labels)
            results.append(out["labels"][0])
        if log_every and index % log_every == 0:
            print(f"  [onnx] zero-shot {index}/{total}")
    return results


def classify_complaints(
    texts: list[str],
    labels: dict[str, str],
    log_every: int = 25,
) -> list[str]:
    """Map zero-shot hypotheses (the dict keys) back to short chart names."""
    raw = classify_zero_shot_onnx(texts, list(labels), log_every=log_every)
    return [labels.get(label, "Unclassified") for label in raw]
