from pathlib import Path
from typing import List, Dict
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, pipeline

class SociomancyONNXPipeline:
    def __init__(self, cache_dir: str = "models/onnx_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print("[*] Initializing ONNX Toxicity Model...")
        # Load toxicity model exported to ONNX
        self.toxic_model_id = "unitary/unbiased-toxic-roberta"
        self.toxic_tokenizer = AutoTokenizer.from_pretrained(self.toxic_model_id)
        self.toxic_model = ORTModelForSequenceClassification.from_pretrained(
            self.toxic_model_id, 
            export=True, 
            cache_dir=self.cache_dir
        )
        self.toxic_pipe = pipeline(
            "text-classification", 
            model=self.toxic_model, 
            tokenizer=self.toxic_tokenizer, 
            top_k=None
        )

        print("[*] Initializing ONNX Emotion Model...")
        # Load emotion model exported to ONNX
        self.emotion_model_id = "SamLowe/roberta-base-go_emotions"
        self.emotion_tokenizer = AutoTokenizer.from_pretrained(self.emotion_model_id)
        self.emotion_model = ORTModelForSequenceClassification.from_pretrained(
            self.emotion_model_id, 
            export=True, 
            cache_dir=self.cache_dir
        )
        self.emotion_pipe = pipeline(
            "text-classification", 
            model=self.emotion_model, 
            tokenizer=self.emotion_tokenizer, 
            top_k=3
        )

    def analyze_comments(self, comments: List[str]) -> List[Dict]:
        """Runs toxicity and emotion analysis over a list of comment strings."""
        results = []
        if not comments:
            return results

        # Truncate text to avoid model token length errors
        cleaned_comments = [c[:512] for c in comments]

        print(f"[*] Running inference on {len(cleaned_comments)} comments...")
        # Pass token-level truncation directly to the pipeline
        toxic_outputs = self.toxic_pipe(
            comments, 
            truncation=True, 
            max_length=512
        )
        emotion_outputs = self.emotion_pipe(
            comments, 
            truncation=True, 
            max_length=512
        )

        for text, toxic_scores, emotion_scores in zip(comments, toxic_outputs, emotion_outputs):
            # Extract highest toxicity score
            toxic_dict = {item['label']: item['score'] for item in toxic_scores}
            is_toxic = toxic_dict.get('toxicity', 0.0) > 0.5
            
            results.append({
                "text": text,
                "toxicity_score": round(toxic_dict.get('toxicity', 0.0), 4),
                "is_toxic": is_toxic,
                "top_emotions": [
                    {"label": e['label'], "score": round(e['score'], 4)} 
                    for e in emotion_scores
                ]
            })

        return results

if __name__ == "__main__":
    # Quick sanity check
    nlp = SociomancyONNXPipeline()
    sample_text = [
        "This video completely changed how I think about video editing! Thank you!",
        "This content is garbage and you don't know what you are talking about."
    ]
    
    evaluations = nlp.analyze_comments(sample_text)
    for ev in evaluations:
        print("\n--------------------------------")
        print(f"Text: {ev['text']}")
        print(f"Toxicity Score: {ev['toxicity_score']} (Toxic: {ev['is_toxic']})")
        print(f"Top Emotions: {ev['top_emotions']}")