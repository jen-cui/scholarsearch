from datasets import load_dataset
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import random
from sklearn.metrics.pairwise import linear_kernel
from sentence_transformers import InputExample, CrossEncoder
from torch.utils.data import DataLoader

ds = load_dataset("mattmorgis/bioasq-12b-rag", "question-answer-passages")
train_ds = ds["dev"]
positive_pairs = []

for ex in tqdm(train_ds):
    q = ex["question"]

    snippets = ex.get("snippets", [])
    for s in snippets:
        text = s.get("text", "")

        if text.strip():
            positive_pairs.append({
                "query": q,
                "positive": text
            })

corpus = [p["positive"] for p in positive_pairs]
vectorizer = TfidfVectorizer(max_features=50000, stop_words='english')
corpus_embeddings = vectorizer.fit_transform(corpus)

triples = []
queries = [p["query"] for p in positive_pairs]
query_embeddings = vectorizer.transform(queries)

batch_size = 256

for i in tqdm(range(0, len(queries), batch_size)):
    end = min(i + batch_size, len(queries))
    batch_scores = linear_kernel(query_embeddings[i:end], corpus_embeddings)

    for j, idx in enumerate(range(i, end)):
        scores = batch_scores[j]
        q = queries[idx]
        pos = positive_pairs[idx]["positive"]
        top_k_indices = np.argpartition(scores, -50)[-50:]

        cand = []
        for cand_idx in top_k_indices:
            if corpus[cand_idx] != pos:
                cand.append(corpus[cand_idx])

        if cand:
            neg = random.choice(cand)
            triples.append({
                "query": q,
                "positive": pos,
                "negative": neg
            })

train_samples = []
for t in triples:
    train_samples.append(
        InputExample(
            texts=[t["query"], t["positive"]],
            label=1.0
        )
    )
    train_samples.append(
        InputExample(
            texts=[t["query"], t["negative"]],
            label=0.0
        )
    )

model_name = "dmis-lab/biobert-base-cased-v1.1"

model = CrossEncoder(model_name, num_labels=1, device='cpu')
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)
model.fit(
    train_dataloader=train_dataloader,
    epochs=2,
    warmup_steps=100,
    output_path="model"
)

model.save("model")