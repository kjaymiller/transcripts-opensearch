import os

import frontmatter
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

def create_embeddings(content: str) -> list[int]:
    """Generate embeddings for a document using HuggingFace's transformers library."""
    embeddings = HuggingFaceEmbeddings()
    return embeddings.embed_documents([content])


if __name__ == "__main__":
    # Create embeddings for each document in the transcripts directory
    post = frontmatter.loads(open("transcripts/0-episode-0-its-the-show-mp3.txt").read())
    embeddings = create_embeddings(post.content)
    post["content_vector"] = embeddings
    print(post.to_dict())
