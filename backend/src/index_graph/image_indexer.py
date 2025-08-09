import os
import requests
from typing import List, Dict, Any

import openai
import pandas as pd
import pinecone

from .configuration import Config


class ImageIndexer:
    def __init__(self) -> None:
        openai.api_key = Config.OPENAI_API_KEY
        pinecone.init(api_key=Config.PINECONE_API_KEY,
                     environment=Config.PINECONE_ENV)

        if Config.PINECONE_IMAGE_INDEX not in pinecone.list_indexes():
            pinecone.create_index(Config.PINECONE_IMAGE_INDEX, dimension=512)
        self._index = pinecone.Index(Config.PINECONE_IMAGE_INDEX)
        self._df = pd.read_csv(Config.CHART_INDEX_PATH)

    def _load_image(self, path: str) -> bytes:
        if path.startswith("http"):
            response = requests.get(path)
            response.raise_for_status()
            return response.content
        with open(path, "rb") as f:
            return f.read()

    def _embed_image(self, image_bytes: bytes) -> List[float]:
        result = openai.Embedding.create(
            input=image_bytes,
            model=Config.VISION_EMBEDDING_MODEL
        )
        return result["data"][0]["embedding"]  # type: ignore

    def index_all(self, batch_size: int = 32) -> None:
        vectors: List[tuple[str, List[float], Dict[str, Any]]] = []
        for idx, row in self._df.iterrows():
            path = row["image_path"]
            try:
                data = self._load_image(path)
                embedding = self._embed_image(data)
                metadata = {
                    "pdf_file": row["pdf_file"],
                    "page": int(row["page"]),
                    "image_index": int(row["image_index"]),
                    **({"caption": row["caption"]} if "caption" in row else {})
                }
                vectors.append((f"img-{idx}", embedding, metadata))
            except Exception as e:
                print(f"Erreur image '{path}': {e}")

            if len(vectors) >= batch_size:
                self._index.upsert(vectors)
                vectors.clear()

        if vectors:
            self._index.upsert(vectors)

    def query(self,
              text_query: str | None = None,
              image_path: str | None = None,
              top_k: int = 5
              ) -> List[Dict[str, Any]]:
        if image_path:
            data = self._load_image(image_path)
            vector = self._embed_image(data)
        elif text_query:
            resp = openai.Embedding.create(
                input=text_query,
                model=Config.TEXT_EMBEDDING_MODEL
            )
            vector = resp["data"][0]["embedding"]  # type: ignore
        else:
            raise ValueError("Fournir 'text_query' ou 'image_path'.")

        response = self._index.query(
            vector=vector,
            top_k=top_k,
          
        )
        return response.get("matches", [])

