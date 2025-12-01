import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from .knowledge_base import pubmed_search, pubmed_fetch
from .reranker import ReRanker

"""
Complete search system combining study retrieval with reranking.
"""
class PaperSearchSystem:
    def __init__(self, reranker_model_path="dmis-lab/biobert-base-cased-v1.1"):
        self.reranker = ReRanker(reranker_model_path)
    
    def search(self, query, max_results=20, rerank_top_k=10):
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'be', 'been', 'being'}
        clean_query = ' '.join(w for w in query.lower().split() if w not in stopwords)
        pmids = pubmed_search(clean_query, max_results=max_results)
        
        if not pmids:
            return []
        
        papers = pubmed_fetch(pmids)
        print(f"Found {len(papers)} papers")

        print("Reranking papers...")
        ranked_papers = self.reranker.rerank(query, papers, top_k=rerank_top_k)
        
        return ranked_papers