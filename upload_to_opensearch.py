import json
import logging
import os
import pathlib
import uuid

import arrow
import frontmatter
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from opensearchpy import OpenSearch, helpers
from rich.progress import track
from rich.console import Console


load_dotenv()

# Log to file
logging.basicConfig(filename="updated_upload_to_opensearch.log", level=logging.WARNING)

connection_string = os.getenv("OPENSEARCH_SERVICE_URI")
client = OpenSearch(connection_string, use_ssl=True, timeout=100)
console = Console()

index_mapping = {
    "properties": {
        "title": {"type": "text"},
        "description": {"type": "text"},
        "url": {"type": "keyword"},
        "content": {"type": "text"},
        "content_vector": {
            "type": "knn_vector",
            "dimension": 768,
            "method": {"name": "hnsw", "space_type": "cosinemil", "engine": "nmslib"},
        },
        "pub_date": {"type": "date"},
    }
}

fmt = r"MMMM[\s+]D[\w+,\s+]YYYY"


def create_embeddings(content: str) -> list[int]:
    """Generate embeddings for a document using HuggingFace's transformers library."""
    embeddings = HuggingFaceEmbeddings()
    return embeddings.embed_documents([content])


def chunk_data(
        data:list[pathlib.Path],
        chunk_size:int=64,
        chunk_overlap:int=20,
        separators:list[str]=[".", "!", "?", "\n"]):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )

    return splitter.create_documents(data)

def load_data(directory: pathlib.Path, index_name: str):
    for file in track(directory.iterdir(), description="Indexing transcripts:"):
        
        post = frontmatter.loads(file.read_text())
        logging.warning(f"Indexing {post['title']}")
        base_data = {
                "_index": index_name,
                "title": post["title"],
                "description": post["description"],
                "url": post["url"],
                "pub_date": arrow.get(post["pub_date"], fmt).date().isoformat(),
            }
        
     
        posts = [{**base_data, **{
                "_id": str(uuid.uuid4()),
                "content": post_snippet.page_content,
                "content_vector": create_embeddings(post_snippet.page_content)[0],
                }} for post_snippet in chunk_data([post.content])]
        print(len(posts))
        response = helpers.bulk(client, posts)
        yield response


def create_posts(
    input_directory: pathlib.Path,
    index_name: str,
) -> None:
    return [post for post in load_data(input_directory, index_name)]


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


if __name__ == "__main__":
    index_name = "embedded_transcripts"
    client.indices.create(index=index_name, body={"mappings": index_mapping}, ignore=400)
    next(load_data(pathlib.Path("transcripts"), index_name))
