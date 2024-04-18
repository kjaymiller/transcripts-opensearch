import json
import uuid
import os
import pathlib
import frontmatter
from opensearchpy import OpenSearch, helpers
from rich.progress import track
from dotenv import load_dotenv

import arrow

from create_embeddings import create_embeddings

load_dotenv()

connection_string = os.getenv("OPENSEARCH_SERVICE_URI")
client = OpenSearch(connection_string, use_ssl=True, timeout=100)

index_mapping = {
    "properties": {
        "title": {"type": "text"},
        "description": {"type": "text"},
        "url": {"type": "keyword"},
        "content": {"type": "text"},
        "content_vector": {
            "type": "knn_vector",
            "dimension": 768,
            "method": {
                "name": "hnsw",
                "space_type": "l2",
                "engine": "faiss"
            },
        },
        "pub_date": {"type": "date"},
    }
}

fmt = r"MMMM[\s+]D[\w+,\s+]YYYY"


def _load_data(directory: pathlib.Path, index_name: str):
    for file in track(directory.iterdir(), description=f"Indexing transcripts:"):
        post = frontmatter.loads(file.read_text())

        yield {
            "_index": index_name,
            "_id": str(uuid.uuid4()),
            "title": post["title"],
            "description": post["description"],
            "url": post["url"],
            "content": post.content,
            "content_vector": create_embeddings(post.content)[0],
            "pub_date": arrow.get(post["pub_date"], fmt).date().isoformat(),
        }

def create_posts(
        input_directory: pathlib.Path,
        index_name:str,
    ) -> None:
    return [post for post in _load_data(input_directory, index_name)]
    
if __name__ == "__main__":
    index_name = "embedded_transcripts"
    client.indices.create(index=index_name, body={"mappings":index_mapping}, ignore=400)
    # posts = create_posts(
    #     input_directory=pathlib.Path("transcripts"),
    #     index_name=index_name,
    # )
    # pathlib.Path("embedded_posts.json").write_text(json.dumps(posts, indent=2))
    posts = json.loads(pathlib.Path("embedded_posts.json").read_text())
    response = helpers.bulk(client, posts)
    print(response)