import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from src.search_system import PaperSearchSystem

"""
Complete agent system integrating search, reranking, and LLM reasoning.
"""
class Agent:
    def __init__(self, llm):
        self.search_system = PaperSearchSystem(reranker_model_path="./model")
        self.llm = llm

    def search_action(self, query: str):
        """
        Executes the search action.
         - query (str): The user's search query.
        Returns a list of top 10 papers.
        """
        results = self.search_system.search(query, max_results=10, rerank_top_k=10)
        return results
    
    def summarize_action(self, papers):
        if not papers:
            return ""
    
        summaries = []
        for i, paper in enumerate(papers[:5], 1):
            text = f"{paper['abstract']}"
            summary = self.llm(text)[0]['summary_text']
            summaries.append(f"{i}. {summary}")
    
        return "\n\n".join(summaries)
        
    def run(self, user_query: str):
        """
        Run the agent with multiple steps.
         - user_query (str): The initial user query.
        Returns a list of steps taken by the agent.
        """
        papers = self.search_action(user_query)
        if papers:
            print("Top papers:")
            for i, paper in enumerate(papers, 1):
                print(f"  {i}. {paper['title']}")
                print(f"     Score: {paper['relevance_score']:.3f}, PMID: {paper['pmid']}")

        print("\n" + "="*60)
        print("SUMMARY:")
        summary = self.summarize_action(papers)
        print(summary)
        print("="*60 + "\n")
        
