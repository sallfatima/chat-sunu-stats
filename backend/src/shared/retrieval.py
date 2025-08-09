"""Manage the configuration of various retrievers.

This module provides functionality to create and manage retrievers for text documents only.
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever
from shared.configuration import BaseConfiguration
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

## Encoder constructors
def make_text_encoder(model: str) -> Embeddings:
    """Connect to the configured text encoder."""
    provider, model = model.split("/", maxsplit=1)
    match provider:
        case "openai":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model=model)
        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")


## Internal Pinecone utility
def _get_or_create_pinecone_vs(index_name: str, embedding_model: Embeddings) -> PineconeVectorStore:
    """Get or create a Pinecone vector store."""
    pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    if index_name not in pinecone_client.list_indexes().names():
        pinecone_client.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    return PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding_model
    )


@asynccontextmanager
async def make_retriever(
    config: RunnableConfig,
) -> AsyncGenerator[VectorStoreRetriever, None]:
    """Create a text retriever for the agent, based on the current configuration."""
    configuration = BaseConfiguration.from_runnable_config(config)
    embedding_model = make_text_encoder(configuration.embedding_model)

    match configuration.retriever_provider:
        case "pinecone":
            vectorstore = _get_or_create_pinecone_vs(
                os.environ["PINECONE_INDEX_NAME"], 
                embedding_model
            )
            retriever = vectorstore.as_retriever(search_kwargs=configuration.search_kwargs)
            yield retriever

        case _:
            raise ValueError(
                "❌ Unrecognized retriever_provider in configuration. "
                f"Expected one of: pinecone\n"
                f"Got: {configuration.retriever_provider}"
            )