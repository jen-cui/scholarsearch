import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from Bio import Entrez

Entrez.email = "jcui1738@gmail.com"  

"""

"""
def pubmed_search(query, max_results=10):
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
