import os
from dotenv import load_dotenv

load_dotenv()

from rich.console import Console
from opensearchpy import OpenSearch
from langchain_community.embeddings import HuggingFaceEmbeddings

CONNECTION_STRING  = os.getenv("OPENSEARCH_SERVICE_URI")
INDEX_NAME ="embedded_transcripts"
client = OpenSearch(CONNECTION_STRING, use_ssl=True, timeout=100)
client.info()


def match_based_search(query:str):
    res = client.search(index=INDEX_NAME, body={
        "_source": {},
        "query": {
           "match": {
               "content": {
                    "query": query
              }
            }
        },
        "highlight": {
            "pre_tags": "**",
            "post_tags": "**",
            "fields": {
                "content": {}
            }
        }
    })
    
    # number of results
    response = f"Number of results: {res['hits']['total']['value']}\n"
    
    for results in res["hits"]["hits"]:
        highlights = "\n".join(results["highlight"]["content"])
        response += f"""Title: {results["_source"]["title"]})
Results: {highlights}
"""

    return response

embeddings = HuggingFaceEmbeddings()

def knn_based_search(query:str):

    query_embedding = embeddings.embed_query(query)
    print(query_embedding)
    res = client.search(index=INDEX_NAME, body={
        "size": 50,
        "query": {
            "knn": {
                "content_vector": {
                    "vector": query_embedding,
                    "k": 10000
                }
            }
        }
    })
    
    # number of results
    print(f"Number of results: {res['hits']['total']['value']}")
    return res

if __name__ == "__main__":
    console = Console()
    match_results = match_based_search("What is Inbox Zero?")

    console.print(f"""Match-based search results for "What is Inbox Zero?":
{match_results}""", style="bold red")
    
    knn_results = knn_based_search("What is Inbox Zero?")
    console.print(f"""KNN-based search results for "What is Inbox Zero?":
{knn_results}""", style="bold green")
    