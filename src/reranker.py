from typing import List, Dict, Optional, Union
from sentence_transformers import CrossEncoder

DEFAULT_MODEL = "dmis-lab/biobert-base-cased-v1.1"

class ReRanker:
    """
    Reranks papers based on relevance using a BioBERT cross-encoder model.
    
    Attributes:
        reranker (CrossEncoder): The cross-encoder model for scoring query-document pairs.
    """
    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
        self.reranker = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, papers: List[Dict[str, str]], top_k: Optional[int] = None) -> List[Dict[str, Union[str, float]]]:
        """
        Rerank given list of papers based on relevance to the query.

        Args:
            query (str): The search query.
            papers (list): List of paper dictionaries with 'title' and 'abstract' keys.
            top_k (int, optional): Number of top papers to return. Default is None (return all).
        
        Returns:
            list: List of top-k papers sorted by relevance score, with added 'relevance_score' key.
        """
        scores = self.reranker.predict([[query, str(p.get('title', '') + p.get('abstract', ''))] for p in papers])
        ranked = sorted(zip(papers, scores), key=lambda x: -x[1])
        for paper, score in ranked[:top_k]:
            paper['relevance_score'] = float(score)
        return [paper for paper, _ in ranked[:top_k]]