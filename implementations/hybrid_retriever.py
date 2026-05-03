"""
Hybrid Retrieval with Cross-Encoder Reranking + MMR Diversity
Production-ready search component for Finance RAG

Features:
- BM25 + Dense hybrid search via Qdrant RRF fusion
- Cross-encoder reranking for precision
- Maximum Marginal Relevance (MMR) for diversity
- Ticker-scoped search for targeted retrieval
- Persistent Qdrant backend

Usage:
    retriever = HybridRetriever(embed_model=my_embed_model)
    results = await retriever.search("Apple revenue 2024", "my_collection", k=5)
"""

import os
import loguru
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

from sentence_transformers import CrossEncoder
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Prefetch, FusionQuery, Fusion,
    Filter, FieldCondition, MatchValue,
)
from fastembed import SparseTextEmbedding


logger = loguru.logger

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")


@dataclass
class SearchResult:
    """Retrieved document with metadata"""
    text: str
    score: float
    rank: int
    metadata: Dict
    chunk_id: Optional[str] = None
    section: Optional[str] = None
    page: Optional[int] = None


class HybridRetriever:
    """
    Hybrid retrieval with cross-encoder reranking + MMR.

    Pipeline:
    1. Dense + BM25 sparse → Qdrant RRF fusion → k_candidates
    2. Cross-encoder reranking → top-k
    3. Optional MMR for diversity
    """

    def __init__(
        self,
        embed_model,
        cross_encoder_model: str = "BAAI/bge-reranker-v2-m3",
        qdrant_url: str = None,
    ):
        """
        Args:
            embed_model: LlamaIndex embedding model (used for dense query + MMR)
            cross_encoder_model: Cross-encoder model name for reranking
            qdrant_url: Persistent Qdrant URL
        """
        self.embed_model = embed_model
        self.qdrant_url = qdrant_url or QDRANT_URL
        self.client = AsyncQdrantClient(url=self.qdrant_url)

        # Sparse BM25 for query embedding
        logger.info("Loading sparse model (BM25)...")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

        # Cross-encoder for reranking
        logger.info(f"Loading cross-encoder: {cross_encoder_model}")
        self.cross_encoder = CrossEncoder(cross_encoder_model)

        logger.info("HybridRetriever initialized ✓")

    def _embed(self, text: str) -> list[float]:
        """Return a single dense embedding vector (fastembed API)."""
        vec = next(iter(self.embed_model.embed([text])))
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)

    # ------------------------------------------------------------------
    # Core search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        collection_name: str,
        k: int = 5,
        k_candidates: int = 20,
        apply_mmr: bool = False,
        mmr_diversity: float = 0.7,
    ) -> List[SearchResult]:
        """
        Full hybrid search pipeline.

        Args:
            query: Search query string
            collection_name: Qdrant collection to search
            k: Final number of results
            k_candidates: Candidates before reranking
            apply_mmr: Enable MMR diversity filtering
            mmr_diversity: λ for MMR (higher = more relevant, lower = more diverse)

        Returns:
            List of SearchResult ranked by relevance
        """
        # 1. Embed query (dense + sparse)
        query_dense = self._embed(query)

        sparse_obj = list(self.sparse_model.embed([query]))[0]
        query_sparse = {
            "indices": (sparse_obj.indices.tolist()
                        if hasattr(sparse_obj.indices, "tolist")
                        else sparse_obj.indices),
            "values": (sparse_obj.values.tolist()
                       if hasattr(sparse_obj.values, "tolist")
                       else sparse_obj.values),
        }

        # 2. Qdrant hybrid search with RRF fusion
        results = await self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                Prefetch(query=query_dense, using="text_dense", limit=k_candidates),
                Prefetch(query=query_sparse, using="bm25_sparse", limit=k_candidates),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=k_candidates,
            with_payload=True,
        )

        # 3. Convert to SearchResult
        candidates = []
        for rank, hit in enumerate(results.points):
            candidates.append(SearchResult(
                text=hit.payload.get("text", ""),
                score=hit.score,
                rank=rank + 1,
                metadata=hit.payload,
                chunk_id=hit.payload.get("chunk_id"),
                section=hit.payload.get("section"),
                page=hit.payload.get("page"),
            ))

        if not candidates:
            return []

        # 4. Cross-encoder reranking
        rerank_k = k * 2 if apply_mmr else k
        reranked = self._rerank(query, candidates, rerank_k)

        # 5. Optional MMR for diversity
        if apply_mmr and len(reranked) > k:
            reranked = self._apply_mmr(query, reranked, k, mmr_diversity)

        # Update final ranks
        for i, result in enumerate(reranked):
            result.rank = i + 1

        return reranked[:k]

    # ------------------------------------------------------------------
    # Cross-encoder reranking
    # ------------------------------------------------------------------

    def _rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        k: int,
    ) -> List[SearchResult]:
        """Rerank candidates using cross-encoder for precision."""

        if not candidates:
            return []

        pairs = [[query, r.text] for r in candidates]
        ce_scores = self.cross_encoder.predict(pairs)

        indexed_scores = sorted(
            enumerate(ce_scores), key=lambda x: x[1], reverse=True
        )

        reranked = []
        for idx, score in indexed_scores[:k]:
            candidates[idx].score = float(score)
            reranked.append(candidates[idx])

        return reranked

    # ------------------------------------------------------------------
    # Maximum Marginal Relevance
    # ------------------------------------------------------------------

    def _apply_mmr(
        self,
        query: str,
        results: List[SearchResult],
        k: int,
        diversity: float,
    ) -> List[SearchResult]:
        """
        MMR for diversity.

        MMR = argmax [ λ * relevance - (1-λ) * max_similarity_to_selected ]
        """

        if len(results) <= k:
            return results

        # Embed query and results for similarity computation
        query_emb = np.array(self._embed(query))
        result_embs = np.array([
            self._embed(r.text) for r in results
        ])

        # Normalize
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        norms = np.linalg.norm(result_embs, axis=1, keepdims=True) + 1e-10
        result_embs = result_embs / norms

        query_sims = result_embs @ query_emb

        selected = []
        remaining = list(range(len(results)))

        while len(selected) < k and remaining:
            best_mmr = -float('inf')
            best_idx = None

            for idx in remaining:
                relevance = query_sims[idx]

                if selected:
                    max_sim = max(
                        float(result_embs[idx] @ result_embs[s])
                        for s in selected
                    )
                else:
                    max_sim = 0

                mmr = diversity * relevance - (1 - diversity) * max_sim

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx

            selected.append(best_idx)
            remaining.remove(best_idx)

        return [results[i] for i in selected]


class FinanceHybridRetriever(HybridRetriever):
    """
    HybridRetriever with finance-specific optimizations:
    - Ticker-scoped search via Qdrant payload filtering
    """

    async def search_by_ticker(
        self,
        query: str,
        ticker: str,
        collection_name: str,
        k: int = 5,
    ) -> List[SearchResult]:
        """
        Search within documents for a specific company ticker.
        Uses Qdrant payload filtering for scoped retrieval.

        Usage:
            results = await retriever.search_by_ticker(
                "risk factors", "AAPL", "my_collection", k=5
            )
        """
        # Embed query
        query_dense = self._embed(query)

        sparse_obj = list(self.sparse_model.embed([query]))[0]
        query_sparse = {
            "indices": sparse_obj.indices.tolist()
                       if hasattr(sparse_obj.indices, "tolist")
                       else sparse_obj.indices,
            "values": sparse_obj.values.tolist()
                      if hasattr(sparse_obj.values, "tolist")
                      else sparse_obj.values,
        }

        # Qdrant payload filter for ticker
        ticker_filter = Filter(
            must=[FieldCondition(key="ticker", match=MatchValue(value=ticker))]
        )

        results = await self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                Prefetch(
                    query=query_dense, using="text_dense",
                    limit=k * 4, filter=ticker_filter,
                ),
                Prefetch(
                    query=query_sparse, using="bm25_sparse",
                    limit=k * 4, filter=ticker_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=k * 2,
            with_payload=True,
        )

        candidates = [
            SearchResult(
                text=hit.payload.get("text", ""),
                score=hit.score,
                rank=i + 1,
                metadata=hit.payload,
            )
            for i, hit in enumerate(results.points)
        ]

        reranked = self._rerank(query, candidates, k)
        for i, r in enumerate(reranked):
            r.rank = i + 1

        return reranked
