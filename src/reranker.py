from typing import List, Dict, Optional, Union
from sentence_transformers import CrossEncoder

DEFAULT_MODEL = "dmis-lab/biobert-base-cased-v1.1"

class ReRanker:
    """
    Reranks papers based on relevance using the model.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
        self.reranker = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, papers: List[Dict[str, str]], top_k: Optional[int] = None) -> List[Dict[str, Union[str, float]]]:
        """
        Rerank given list of papers.

        Pape example: {"title": "...", "abstract": "..."}
        """
       
        scores = self.reranker.predict([[query, str(p.get('title', '') + p.get('abstract', ''))] for p in papers])
        ranked = sorted(zip(papers, scores), key=lambda x: -x[1])
        for paper, score in ranked[:top_k]:
            paper['relevance_score'] = float(score)
        return [paper for paper, _ in ranked[:top_k]]