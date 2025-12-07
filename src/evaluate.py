# Evaluation was done in notebooks/evaluate.ipynb for better computational power. Code is here for reference and documentation.

import numpy as np
from sentence_transformers import CrossEncoder
from tqdm import tqdm
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Generate negative pairs using TF-IDF similarity and positive pairs.
def generate_negatives(dataset, num_negatives=5):
    all_snippets = []
    snippet_to_question = {}

    for ex in dataset:
        for snippet in ex["snippets"]:
            text = snippet["text"]
            all_snippets.append(text)
            snippet_to_question[text] = ex["question"]

    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    corpus_embeddings = vectorizer.fit_transform(all_snippets)

    eval_data = []
    for ex in tqdm(dataset, desc="Generating negatives"):
        query = ex["question"]
        positives = [s["text"] for s in ex["snippets"]]

        query_vec = vectorizer.transform([query])
        scores = linear_kernel(query_vec, corpus_embeddings).flatten()

        top_indices = np.argsort(-scores)
        negatives = []
        for idx in top_indices:
            candidate = all_snippets[idx]
            if candidate not in positives and len(negatives) < num_negatives:
                negatives.append(candidate)

        eval_data.append({
            "question": query,
            "positives": positives,
            "negatives": negatives
        })

    return eval_data

# Evaluate
def evaluate_reranker(model, eval_data, k=10, threshold=0.5):
    all_predictions = []
    all_labels = []

    for ex in tqdm(eval_data, desc="Evaluating"):
        query = ex["question"]
        texts = ex["positives"] + ex["negatives"]
        labels = np.array([1] * len(ex["positives"]) + [0] * len(ex["negatives"]))

        pairs = [[query, t] for t in texts]
        scores = np.array(model.predict(pairs)).flatten()

        predictions = (scores > threshold).astype(int)
        all_predictions.extend(predictions)
        all_labels.extend(labels)

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    tp = np.sum((all_predictions == 1) & (all_labels == 1))
    fp = np.sum((all_predictions == 1) & (all_labels == 0))
    fn = np.sum((all_predictions == 0) & (all_labels == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = np.mean(all_predictions == all_labels)

    return {
        "F1": f1,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
    }

ds = load_dataset("mattmorgis/bioasq-12b-rag", "question-answer-passages")
val = ds["eval"]

eval_data = generate_negatives(val, num_negatives=5)

model = CrossEncoder("./model")
results = evaluate_reranker(model, eval_data, k=10, threshold=0.5)
print(results)
