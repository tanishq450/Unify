import loguru
from pathlib import Path

from Model_loader.llm import ModelLoader
from utils.Data_ingestion import chunking, Docloader

from qdrant.qdrant import QdrantHybridClient
from qdrant_client.models import SparseVector
from fastembed import SparseTextEmbedding
from implementations.hybrid_retriever import HybridRetriever


# Sparse BM25 model (local, no API needed)
_sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

def compute_sparse_vectors(texts: list[str]):
    """Compute sparse vectors (BM25) using fastembed"""
    return list(_sparse_model.embed(texts))


class Rag_pipeline:
    def __init__(self):
        self.logger = loguru.logger

        # Core components
        self.model_loader = ModelLoader()
        self.model_loader.load_models()

        self.docloader = Docloader()
        self.chunker = chunking()

        # Qdrant hybrid client (for ingestion)
        self.qdrant = QdrantHybridClient()

        # Hybrid retriever (for search — reranking + MMR)
        # Pass model_loader directly — it exposes .embed() used by HybridRetriever
        self.retriever = HybridRetriever(
            embed_model=self.model_loader,
        )

    # ---------------- PREP ----------------
    def _prepare_documents(self, file_path):
        text = self.docloader.load_pdf(file_path)
        chunks = self.chunker.chunk_text(text)
        documents = self.chunker.convert_chunks(chunks)

        if not documents:
            raise RuntimeError("No documents produced")

        return documents

    # ---------------- EMBEDDINGS ----------------
    def _embed_documents(self, documents):
        texts = [doc.text for doc in documents]

        # Dense via MeshAPI (model_loader.embed), sparse via fastembed BM25
        dense = self.model_loader.embed(texts)
        sparse = compute_sparse_vectors(texts)

        return texts, dense, sparse

    # ---------------- BUILD POINTS ----------------

    def _build_points(self, texts, dense_embeddings, sparse_vectors):
        from qdrant_client.models import PointStruct

        points = []

        for i, (text, dense, sparse) in enumerate(zip(texts, dense_embeddings, sparse_vectors)):

            sparse_vector = SparseVector(
                indices=sparse.indices.tolist(),
                values=sparse.values.tolist()
            )

            points.append(
                PointStruct(
                    id=i,
                    vector={
                        "text_dense": dense,
                        "bm25_sparse": sparse_vector
                    },
                    payload={"text": text}
                )
            )

        return points

    # ---------------- INGEST ----------------
    async def ingest(self, file_path: str, persist_dir: str):

        try:
            collection_name = Path(persist_dir).name

            # 1. Prepare data
            documents = self._prepare_documents(file_path)

            # 2. Embeddings
            texts, dense, sparse = self._embed_documents(documents)

            # 3. Build Qdrant points
            points = self._build_points(texts, dense, sparse)

            # 4. Store in Qdrant
            await self.qdrant.create_collection(collection_name)
            await self.qdrant.upsert(collection_name, points)

            self.logger.info(f"Ingested → {collection_name}")

        except Exception:
            self.logger.exception("Ingestion failed")
            raise

    # ---------------- QUERY ----------------
    async def query(self, query: str, persist_dir: str):

        try:
            collection_name = Path(persist_dir).name

            # 1. Hybrid search + cross-encoder reranking (+ optional MMR)
            search_results = await self.retriever.search(
                query=query,
                collection_name=collection_name,
                k=5,
                k_candidates=20,
                apply_mmr=True,
                mmr_diversity=0.7,
            )

            if not search_results:
                return {"answer": "", "score": 0.0, "nodes": []}

            # 2. Convert to node dicts
            reranked_nodes = [
        {
            "text": r.text,
            "source": r.metadata.get("source"),
            "page": r.page,
            "chunk_id": r.chunk_id,
            "section": r.section,
        }
        for r in search_results
    ]

            # 3. Build context from top-3
            top_k_nodes = reranked_nodes[:3]
            context = "\n\n".join([n["text"] for n in top_k_nodes])

            # 4. LLM synthesis
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant. Answer only from the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer clearly and concisely:"},
            ]

            answer = self.model_loader.chat(messages)

            score = search_results[0].score if search_results else 0.0

            citations = [
            {
                "chunk_id": n["chunk_id"],
                "page": n["page"],
                "section": n["section"],
                "source": n["source"]
            }
            for n in top_k_nodes
        ]

            return {
             "answer": answer,
             "score": score,
             "nodes": reranked_nodes,
             "citations": citations,
}

        except Exception:
            self.logger.exception("Query failed")
            raise