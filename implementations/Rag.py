import loguru
from pathlib import Path

from Model_loader.llm import ModelLoader
from utils.Data_ingestion import chunking, Docloader

from llama_index.core.postprocessor import SentenceTransformerRerank

from qdrant.qdrant import QdrantHybridClient
from qdrant_client.models import SparseVector
from fastembed import SparseTextEmbedding
from types import SimpleNamespace
from llama_index.core.schema import NodeWithScore, TextNode

# Initialize sparse model globally to avoid reloading
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

def compute_sparse_vectors(texts: list[str]):
    """Compute sparse vectors (BM25) using fastembed"""
    embeddings = list(sparse_model.embed(texts))
    return embeddings


# ---------------- Reranker ----------------
reranker = SentenceTransformerRerank(
    model="BAAI/bge-reranker-v2-m3",
    top_n=5
)


class Rag_pipeline:
    def __init__(self):
        self.logger = loguru.logger

        # Core components
        self.model_loader = ModelLoader()
        self.docloader = Docloader()
        self.chunker = chunking()

        # Qdrant hybrid client
        self.qdrant = QdrantHybridClient()

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

        dense = self.model_loader.embed_model.get_text_embedding_batch(texts)
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
            self.model_loader.load_models()
            self.model_loader.set_settings()

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
            self.model_loader.load_models()
            self.model_loader.set_settings()

            collection_name = Path(persist_dir).name

            # 1. Query embeddings
            query_dense = self.model_loader.embed_model.get_text_embedding(query)
            if hasattr(query_dense, "tolist"):
                query_dense = query_dense.tolist()

            sparse_obj = compute_sparse_vectors([query])[0]
            query_sparse = {
                "indices": sparse_obj.indices.tolist() if hasattr(sparse_obj.indices, "tolist") else sparse_obj.indices,
                "values": sparse_obj.values.tolist() if hasattr(sparse_obj.values, "tolist") else sparse_obj.values,
            }

            # 2. Hybrid search
            results = await self.qdrant.search(
                collection_name,
                query_dense,
                query_sparse,
                top_k=10
            )

            # 3. Extract texts
            texts = [r.payload["text"] for r in results]

            # 4. Convert to Llama nodes
            llama_nodes = [
                NodeWithScore(node=TextNode(text=t), score=1.0)
                for t in texts
            ]

            # 5. Rerank
            reranked = reranker.postprocess_nodes(
                llama_nodes,
                query_str=query
            )

            # 6. Convert to dict
            reranked_nodes = [
                {"text": n.node.get_content()}
                for n in reranked
            ]

            # 7. Build context
            top_k_nodes = reranked_nodes[:3]
            context = "\n\n".join([n["text"] for n in top_k_nodes])

            # 8. LLM synthesis (REAL RAG)
            prompt = f"""
            You are a helpful AI assistant.

            Answer the question based ONLY on the context below.

            Context:
            {context}

            Question:
            {query}

            Answer clearly and concisely:
            """

            response = self.model_loader.llm.complete(prompt)

            answer = str(response)

            score = reranked[0].score if reranked else 0.0

            return {
                "answer": answer,
                "score": score,
                "nodes": reranked_nodes,  
            }

        except Exception:
            self.logger.exception("Query failed")
            raise