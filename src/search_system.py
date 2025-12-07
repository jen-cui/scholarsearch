import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from .knowledge_base import pubmed_search, pubmed_fetch
from .reranker import ReRanker
class PaperSearchSystem:
    """
    Complete search system combining PubMed retrieval with reranking.
    
    Attributes:
        reranker (ReRanker): ReRanker instance for scoring paper relevance.
    """
    def __init__(self, reranker_model_path="dmis-lab/biobert-base-cased-v1.1"):
        self.reranker = ReRanker(reranker_model_path)
    
    def search(self, query, max_results=20, rerank_top_k=10):
        """
        Search PubMed and rerank results by relevance.
        
        Args:
            query (str): The user's search query.
            max_results (int): Maximum number of papers to retrieve from PubMed. Default is 20.
            rerank_top_k (int): Number of top papers to return after reranking. Default is 10.
        
        Returns:
            list: List of top-k reranked papers with relevance scores, sorted by relevance.
                  List of dictionaries containing paper information with keys:
                    - title (str): Paper title
                    - abstract (str): Paper abstract text
                    - pmid (str): PubMed ID
                    - relevance_score (float): Relevance score from reranker (0-1)
        """
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