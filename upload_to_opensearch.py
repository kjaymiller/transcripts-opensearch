import json
import os
import pathlib
import uuid

import arrow
import frontmatter
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from opensearchpy import OpenSearch, helpers
from rich.progress import track

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
            "method": {"name": "hnsw", "space_type": "l2", "engine": "faiss"},
        },
        "pub_date": {"type": "date"},
    }
}

fmt = r"MMMM[\s+]D[\w+,\s+]YYYY"


def create_embeddings(content: str) -> list[int]:
    """Generate embeddings for a document using HuggingFace's transformers library."""
    embeddings = HuggingFaceEmbeddings()
    return embeddings.embed_documents([content])


def _load_data(directory: pathlib.Path, index_name: str):
    for file in track(directory.iterdir(), description="Indexing transcripts:"):
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
    index_name: str,
) -> None:
    return [post for post in _load_data(input_directory, index_name)]


def posts_to_json(
    index_name: str,
    input_directory: pathlib.Path,
    output_file: pathlib.Path,
) -> None:
    posts = create_posts(
        input_directory=input_directory,
        index_name=index_name,
    )
    output_file.write_text(json.dumps(posts, indent=2))


def upload_to_opensearch(
    index_name: str,
    input_directory: pathlib.Path,
) -> None:
    posts = create_posts(
        input_directory=input_directory,
        index_name=index_name,
    )
    response = helpers.bulk(client, posts)
    return response


def upload_from_file(
    output_file: pathlib.Path,
) -> None:
    """ "Upload a json file to opensearch"""
    posts = json.loads(output_file.read_text())
    response = helpers.bulk(client, posts)
    return response
