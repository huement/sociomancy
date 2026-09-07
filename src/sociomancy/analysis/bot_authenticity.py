from __future__ import annotations
import re
import hashlib
from dataclasses import dataclass, field
from typing import Any
import pandas as pd
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import HDBSCAN
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False


@dataclass
class BotSignal:
    category: str
    score: float
    details: str


class BotAuthenticityDetector:
    """
    Computes comment authenticity metrics and logs specific anomaly 
    flag reasons for individual comment inspection.
    """
    def __init__(self, suspicious_threshold: float = 60.0):
        self.suspicious_threshold = suspicious_threshold
        self.embedder = None
        if HAS_EMBEDDINGS:
            try:
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self.embedder = None

    @staticmethod
    def normalize_text(text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^\w\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def analyze_dataframe(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        if df.empty:
            return df, {}

        work_df = df.copy()
        total_comments = len(work_df)

        # 1. Exact & Near-Duplicate Hashing
        work_df["norm_text"] = work_df["text"].apply(self.normalize_text)
        work_df["text_hash"] = work_df["norm_text"].apply(
            lambda t: hashlib.md5(t.encode("utf-8")).hexdigest() if t else ""
        )
        
        dup_counts = work_df["text_hash"].value_counts().to_dict()
        work_df["exact_dup_count"] = work_df["text_hash"].map(dup_counts).fillna(1)
        work_df["dup_score"] = np.clip((work_df["exact_dup_count"] - 1) / 4.0, 0.0, 1.0)

        # 2. Author Frequency Anomaly
        if "author_id" in work_df.columns:
            valid_authors = work_df[work_df["author_id"] != "ANONYMOUS"]
            author_counts = valid_authors["author_id"].value_counts().to_dict()
            work_df["author_comment_count"] = work_df["author_id"].map(author_counts).fillna(1)
            work_df["author_anomaly_score"] = np.clip((work_df["author_comment_count"] - 2) / 8.0, 0.0, 1.0)
        else:
            work_df["author_comment_count"] = 1
            work_df["author_anomaly_score"] = 0.0

        # 3. Text Repetition Anomaly
        def compute_text_anomaly(text: str) -> float:
            if not text or len(text) < 3:
                return 0.2
            words = text.split()
            if not words:
                return 0.0
            unique_ratio = len(set(words)) / len(words)
            repetition_score = 1.0 - unique_ratio if len(words) > 4 else 0.0
            return float(np.clip(repetition_score, 0.0, 1.0))

        work_df["text_anomaly_score"] = work_df["norm_text"].apply(compute_text_anomaly)

        # 4. Semantic Clustering via Embeddings + HDBSCAN
        work_df["cluster_anomaly_score"] = 0.0
        cluster_count = 0
        
        if self.embedder is not None and total_comments >= 10:
            try:
                embeddings = self.embedder.encode(work_df["text"].tolist(), show_progress_bar=False)
                clusterer = HDBSCAN(min_cluster_size=max(3, int(total_comments * 0.015)))
                labels = clusterer.fit_predict(embeddings)
                
                work_df["cluster_label"] = labels
                unique_clusters = set(labels) - {-1}
                cluster_count = len(unique_clusters)
                
                cluster_sizes = pd.Series(labels).value_counts().to_dict()
                work_df["cluster_size"] = work_df["cluster_label"].map(cluster_sizes).fillna(0)
                work_df["cluster_anomaly_score"] = np.where(
                    work_df["cluster_label"] != -1,
                    np.clip(work_df["cluster_size"] / (total_comments * 0.1), 0.3, 1.0),
                    0.0
                )
            except Exception:
                pass

        # 5. Composite Bot Score (0-100)
        work_df["bot_confidence"] = (
            (work_df["dup_score"] * 0.35) +
            (work_df["author_anomaly_score"] * 0.25) +
            (work_df["text_anomaly_score"] * 0.20) +
            (work_df["cluster_anomaly_score"] * 0.20)
        ) * 100.0

        work_df["bot_confidence"] = work_df["bot_confidence"].round(2)
        work_df["is_suspicious"] = work_df["bot_confidence"] >= self.suspicious_threshold

        # 6. Generate Explainable Flag Reasons Column
        def compile_reasons(row: pd.Series) -> str:
            reasons = []
            if row["dup_score"] > 0:
                reasons.append(f"exact_duplicate(n={int(row['exact_dup_count'])})")
            if row["author_anomaly_score"] > 0.3:
                reasons.append(f"high_author_activity(n={int(row['author_comment_count'])})")
            if row["text_anomaly_score"] > 0.3:
                reasons.append("high_token_repetition")
            if row["cluster_anomaly_score"] > 0.3:
                reasons.append("semantic_copy_cluster")

            return " | ".join(reasons) if reasons else "clean"

        work_df["bot_reasons"] = work_df.apply(compile_reasons, axis=1)

        # Aggregation Summary
        suspicious_comments_count = int(work_df["is_suspicious"].sum())
        suspicious_rate_pct = round((suspicious_comments_count / total_comments) * 100, 2)
        comment_authenticity_score = max(0.0, round(100.0 - suspicious_rate_pct, 2))

        summary = {
            "total_comments_analyzed": total_comments,
            "suspicious_comments_count": suspicious_comments_count,
            "suspicious_rate_pct": suspicious_rate_pct,
            "comment_authenticity_score": comment_authenticity_score,
            "exact_duplicate_groups": len([c for c in dup_counts.values() if c > 1]),
            "semantic_copy_clusters": cluster_count,
            "avg_bot_confidence": round(float(work_df["bot_confidence"].mean()), 2),
            "flagged_reasons_breakdown": pd.Series(
                [r for sublist in work_df[work_df["is_suspicious"]]["bot_reasons"].str.split(" | ") for r in sublist if r != "clean"]
            ).value_counts().to_dict()
        }

        clean_df = work_df.drop(
            columns=["norm_text", "text_hash", "dup_score", "author_anomaly_score", "text_anomaly_score", "cluster_anomaly_score"],
            errors="ignore"
        )

        return clean_df, summary