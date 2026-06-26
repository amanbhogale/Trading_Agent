# news_classifier.py
import logging
import threading
from transformers import pipeline

logger = logging.getLogger(__name__)

class NewsClassifier:
    def __init__(self):
        self.lock = threading.Lock()
        self.classifier = None

    def _load_model(self):
        """Loads the Hugging Face zero-shot model lazily."""
        if self.classifier is None:
            try:
                logger.info("Lazily loading Hugging Face zero-shot classification pipeline...")
                # Load the popular, highly rated zero-shot classification model from Hugging Face
                self.classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
                logger.info("Hugging Face zero-shot classification model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load Hugging Face zero-shot model: {e}")
                self.classifier = None

    def predict(self, title: str, description: str, threshold: float = 0.5) -> str:
        res = self.predict_with_confidence(title, description, threshold)
        return res["region"]

    def predict_with_confidence(self, title: str, description: str, threshold: float = 0.5) -> dict:
        """
        Classifies news based on title and description.
        Returns a dict: {'region': 'Indian'|'Global', 'confidence': float, 'method': 'Hugging Face...'}
        """
        combined = f"{title} {description}".strip()
        if not combined:
            return {"region": "Global", "confidence": 1.0, "method": "Fallback Default"}

        with self.lock:
            self._load_model()
            if not self.classifier:
                return {"region": "Global", "confidence": 1.0, "method": "Fallback Default"}

            try:
                # Run Hugging Face zero-shot classification
                res = self.classifier(
                    combined, 
                    candidate_labels=["Indian Markets", "Global Markets"]
                )
                # Find the top label and its score
                top_label = res["labels"][0]
                confidence = float(res["scores"][0])
                
                # Map candidate label to region name
                region = "Indian" if top_label == "Indian Markets" else "Global"
                
                return {
                    "region": region,
                    "confidence": round(confidence, 2),
                    "method": "Hugging Face Zero-Shot (DistilBERT)"
                }
            except Exception as e:
                logger.error(f"Inference error in NewsClassifier Hugging Face pipeline: {e}")
                return {"region": "Global", "confidence": 1.0, "method": "Hugging Face Error"}
