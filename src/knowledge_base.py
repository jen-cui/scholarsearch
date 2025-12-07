import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from Bio import Entrez

Entrez.email = "cui.jen@northeastern.edu"  

def pubmed_search(query, max_results=10):
    """
    Search PubMed for papers matching the query.
        
    Args:
        query (str): Search query string.
        max_results (int): Maximum number of paper IDs to retrieve. Default is 10.
        
    Returns:
        list: List of PubMed IDs (PMIDs) as strings.
    """
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
        sort="relevance"
    )
    results = Entrez.read(handle)
    handle.close()
    return results["IdList"] 


def pubmed_fetch(pmid_list):
    """
    Fetch paper details from PubMed given a list of PMIDs.
    
    Args:
        pmid_list (list): List of PubMed IDs (PMIDs) as strings.
    
    Returns:
        list: List of dictionaries containing paper information with keys:
            - title (str): Paper title
            - abstract (str): Paper abstract text
            - pmid (str): PubMed ID
    """
    handle = Entrez.efetch(
        db="pubmed",
        id=",".join(pmid_list),
        rettype="abstract",
        retmode="xml"
    )
    papers = Entrez.read(handle)
    handle.close()

    extracted = []
    for paper in papers["PubmedArticle"]:
        info = paper["MedlineCitation"]["Article"]

        extracted.append({
            "title": info.get("ArticleTitle", ""),
            "abstract": " ".join(info.get("Abstract", {}).get("AbstractText", [])),
            "pmid": paper["MedlineCitation"]["PMID"],
        })

    return extracted
