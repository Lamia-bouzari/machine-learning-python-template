"""Production inference workflow for WeLoveReviews sentiment analysis."""

from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
INPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "reviews.csv"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "reviews_with_sentiment.csv"


def stars_to_sentiment(stars: int) -> str:
	"""Map one-to-five-star predictions to business sentiment labels."""
	if stars <= 2:
		return "NEGATIVE"
	if stars == 3:
		return "NEUTRAL"
	return "POSITIVE"


def load_reviews(path: Path = INPUT_PATH) -> pd.DataFrame:
	"""Load and validate the review dataset."""
	reviews = pd.read_csv(path)
	required_columns = {"review_id", "rating", "review_text"}
	missing = required_columns.difference(reviews.columns)
	if missing:
		raise ValueError(f"Missing required columns: {sorted(missing)}")
	return reviews


def run_inference(reviews: pd.DataFrame) -> pd.DataFrame:
	"""Run the single loaded classifier over every review."""
	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
	model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
	model.eval()

	texts = reviews["review_text"].fillna("").astype(str).tolist()
	predictions: list[int] = []
	confidences: list[float] = []
	batch_size = 16

	with torch.inference_mode():
		for start in range(0, len(texts), batch_size):
			batch = tokenizer(
				texts[start : start + batch_size],
				padding=True,
				truncation=True,
				max_length=512,
				return_tensors="pt",
			)
			probabilities = torch.softmax(model(**batch).logits, dim=-1)
			confidence, class_indexes = probabilities.max(dim=-1)
			predictions.extend((class_indexes + 1).tolist())
			confidences.extend(confidence.tolist())

	result = reviews.copy()
	result["predicted_stars"] = predictions
	result["predicted_sentiment"] = [stars_to_sentiment(stars) for stars in predictions]
	result["confidence"] = confidences
	return result


def write_output(results: pd.DataFrame, path: Path = OUTPUT_PATH) -> None:
	"""Write scored reviews to the processed-data directory."""
	path.parent.mkdir(parents=True, exist_ok=True)
	results.to_csv(path, index=False)


def main() -> None:
	reviews = load_reviews()
	results = run_inference(reviews)
	write_output(results)

	counts = results["predicted_sentiment"].value_counts().reindex(
		["NEGATIVE", "NEUTRAL", "POSITIVE"], fill_value=0
	)
	percentages = (counts / len(results) * 100).round(2)
	print(f"Processed reviews: {len(results)}")
	print("Sentiment counts:")
	print(counts.to_string())
	print("Sentiment percentages:")
	print(percentages.to_string())
	print(f"Wrote: {OUTPUT_PATH}")


if __name__ == "__main__":
	main()
