
from pathlib import Path
import numpy as np
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

class EmbeddingService:
    def __init__(self, cache_dir: str = "models/onnx_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print("[*] Initializing ONNX Embedding Model (BAAI/bge-small-en-v1.5)...")
        self.model_id = "BAAI/bge-small-en-v1.5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = ORTModelForFeatureExtraction.from_pretrained(
            self.model_id, 
            export=True, 
            cache_dir=self.cache_dir
        )

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generates a single embedding for a given text."""
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[0][0].numpy()

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Generates embeddings for a list of texts."""
        if not texts:
            return np.array([])
        
        # For simplicity, embedding one by one. Can be optimized for batching.
        embeddings = [self._generate_embedding(text) for text in texts]
        return np.array(embeddings)

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculates cosine similarity between two vectors."""
        if v1.size == 0 or v2.size == 0:
            return 0.0
        
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        return np.dot(v1_norm, v2_norm).item()

if __name__ == "__main__":
    # Quick sanity check
    embed_service = EmbeddingService()
    
    sample_texts = [
        "This is a test sentence.",
        "This is another test sentence, but slightly different."
    ]
    
    embeddings = embed_service.generate_embeddings(sample_texts)
    
    print(f"Generated {len(embeddings)} embeddings.")
    print(f"Embedding for first sentence (shape): {embeddings[0].shape}")

    similarity = embed_service.cosine_similarity(embeddings[0], embeddings[1])
    print(f"Cosine similarity between the two sentences: {similarity:.4f}")
