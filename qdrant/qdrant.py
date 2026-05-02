
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    SparseVectorParams,
    SparseIndexParams,
    Prefetch,
    FusionQuery,
    Fusion
)

import os

from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")


# ---------------- SIMPLE (LlamaIndex) ----------------
def get_qdrant_vector_store(collection_name: str):

    client = QdrantClient(url=QDRANT_URL)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    return storage_context


# ---------------- HYBRID CLIENT ----------------
class QdrantHybridClient:

    def __init__(self):
        self.client = AsyncQdrantClient(url=QDRANT_URL)

    # -------- CREATE COLLECTION --------
    async def create_collection(self, collection_name: str, dim: int = 2560):

        collections = await self.client.get_collections()
        existing = {c.name for c in collections.collections}

        if collection_name in existing:
            return

        await self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "text_dense": VectorParams(
                    size=dim,
                    distance=Distance.COSINE
                )
            },
            sparse_vectors_config={
                "bm25_sparse": SparseVectorParams(
                    index=SparseIndexParams()
                )
            },
        )

    # -------- UPSERT --------
    async def upsert(self, collection_name, points):
        await self.client.upsert(
            collection_name=collection_name,
            points=points
        )

    # -------- HYBRID SEARCH (RRF) --------
    async def search(
        self,
        collection_name,
        query_dense,
        query_sparse,
        top_k=10
    ):

        results = await self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                Prefetch(query=query_dense, using="text_dense", limit=top_k * 2),
                Prefetch(query=query_sparse, using="bm25_sparse", limit=top_k * 2),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        return results.points