import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from src.search_system import PaperSearchSystem

class Agent:
    """
    Complete agent system integrating search, reranking, and LLM reasoning.
    
    Attributes:
        search_system (PaperSearchSystem): System for searching and reranking papers.
        llm: Language model for summarization.
    """
    def __init__(self, llm):
        self.search_system = PaperSearchSystem(reranker_model_path="./model")
        self.llm = llm

    def search_action(self, query: str):
        """
        Executes the search action.

        Args:
            query (str): The user's search query.
        
        Returns:
            list: list of top 10 papers
        """
        results = self.search_system.search(query, max_results=10, rerank_top_k=10)
        return results
    
    def summarize(self, paper):
        """
        Generates a summary for a given paper.
        
        Args:
            paper (dict): Paper dictionary containing 'abstract' key.
        
        Returns:
            str: Generated summary text.
        """
        text = f"{paper['abstract']}"
        summary = self.llm(text)[0]['summary_text']
        return summary
        
    def run(self, user_query: str):
        """
        Run the agent with multiple steps.
        
        Args:
            user_query (str): The user query.
        """
        papers = self.search_action(user_query)
        if papers:
            print("Top papers:")
            for i, paper in enumerate(papers, 1):
                print(f"  {i}. {paper['title']}")
                print(f"     Score: {paper['relevance_score']:.3f} PMID: {paper['pmid']}")
                print("SUMMARY:")
                summary = self.summarize(paper)
                print(summary)
                print("\n")
        
