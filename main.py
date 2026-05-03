"""
Finance RAG Orchestrator — main.py

Central brain that connects:
  1. Intent Classification (AdaptiveRouter)
  2. Retrieval Engines (Simple RAG / Graph RAG / Table RAG)
  3. LLM Generation
  4. Hallucination Verification (FinGroundVerifier)

Usage:
    # Ingest a PDF
    python main.py ingest path/to/report.pdf my_collection

    # Query
    python main.py query "What was Apple's revenue in 2024?" my_collection

    # Interactive REPL
    python main.py interactive my_collection
"""

import asyncio
import sys
import os
import loguru
from dotenv import load_dotenv

load_dotenv()

from Model_loader.llm import ModelLoader
from implementations.intent_classifier import IntentClassifier, RAGStrategy
from implementations.Rag import Rag_pipeline
from implementations.Graph_rag import GRAPH_RAG
from implementations.hallucination_verifier import FinGroundVerifier
from utils.Data_ingestion import Docloader, unified_ingest
from langchain_anthropic import ChatAnthropic


logger = loguru.logger


# ============================================================================
# Orchestrator
# ============================================================================

class FinanceRAGOrchestrator:
    """
    Unified orchestrator for the Finance RAG pipeline.

    Flow:
        User Query
          → IntentClassifier (route to strategy)
          → Retrieval Engine (Simple RAG / Graph RAG / Table RAG)
          → LLM Draft Answer
          → FinGround Hallucination Verifier
          → Verified Answer
    """

    def __init__(self, collection_name: str = "default"):
        logger.info("Initializing Finance RAG Orchestrator...")

        self.collection_name = collection_name

        # --- Core components ---
        self.model_loader = ModelLoader()
        self.model_loader.load_models()

        # --- Intent classification ---
        self.classifier = IntentClassifier()

        # --- Retrieval engines ---
        self.rag_pipeline = Rag_pipeline()

        # Graph RAG — optional, only if Neo4j is reachable
        self.graph_rag = None
        self._init_graph_rag()

        # Table-aware RAG — optional, loaded on demand
        self.table_rag = None

        # --- Hallucination verifier ---
        self.verifier = FinGroundVerifier(
            llm_client=self.model_loader,
            embedding_model=self.model_loader
        )

        logger.info("Orchestrator ready ✓")

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_graph_rag(self):
        """Try to connect Graph RAG; skip silently if Neo4j is down."""
        try:
            self.graph_rag = GRAPH_RAG()
            # Graph RAG needs a LangChain-compatible LLM
            from langchain_openai import ChatOpenAI
            api_key = os.getenv("MESH_API_KEY")
            if not api_key:
                logger.warning("Graph RAG: MESH_API_KEY not found in environment. Graph ingestion will fail.")
            
            langchain_llm = ChatAnthropic(
                model="anthropic/claude-opus-4.1",
                temperature=0.1,
                max_tokens=4096,
                base_url=os.getenv("MESH_API_BASE", "https://api.meshapi.ai/v1"),
                api_key=api_key,
            )

            self.graph_rag.load_llm(langchain_llm)

            logger.info("Graph RAG connected ✓")
        except Exception as e:
            logger.warning(f"Graph RAG unavailable (Neo4j down?): {e}")
            self.graph_rag = None

    def _init_table_rag(self, pdf_path: str):
        """Lazy-load the multimodal table extractor for a specific PDF."""
        try:
            from implementations.multimodal_table_extractor import (
                UnifiedTableExtractor,
                TableAwareRAG,
            )

            extractor = UnifiedTableExtractor(prefer_local=False)
            self.table_rag = TableAwareRAG(extractor)
            self.table_rag.ingest(pdf_path)
            logger.info("Table-aware RAG loaded ✓")
        except Exception as e:
            logger.warning(f"Table RAG unavailable: {e}")
            self.table_rag = None

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    async def ingest(self, file_path: str, max_graph_chunks: int = 20):
        """
        Ingest a PDF into both Qdrant (hybrid RAG) and Neo4j (graph RAG).
        Also pre-loads table RAG for the document.
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return

        logger.info(f"Starting ingestion for: {file_path}")

        # Hybrid + Graph RAG ingestion
        await unified_ingest(file_path, self.collection_name, max_graph_chunks=max_graph_chunks)

        # Table RAG (pre-extract tables for multimodal queries)
        self._init_table_rag(file_path)

        logger.info("Ingestion complete ✓")

    # ------------------------------------------------------------------
    # Query — the main pipeline
    # ------------------------------------------------------------------

    async def query(self, user_query: str) -> dict:
        """
        Full pipeline: Route → Retrieve → Generate → Verify.

        Returns:
            {
                "answer": str,
                "verified_answer": str,
                "confidence": float,
                "intent": str,
                "strategy": str,
                "routing_reasoning": str,
                "claims": [...],
            }
        """
        # ---- Step 1: Classify intent ----
        routing = self.classifier.classify(user_query)
        logger.info(
            f"Intent: {routing.intent.value} | "
            f"Strategy: {routing.strategy.value} | "
            f"Confidence: {routing.confidence}"
        )

        # ---- Step 2: Retrieve context ----
        context_chunks = []
        raw_result = {}

        if routing.strategy == RAGStrategy.GRAPH_RAG and self.graph_rag:
            raw_result = await self._retrieve_graph(user_query)
            context_chunks = raw_result.get("context", [])

        elif routing.strategy == RAGStrategy.MULTIMODAL_RAG and self.table_rag:
            raw_result = self._retrieve_tables(user_query)
            context_chunks = raw_result.get("context", [])

        # Fallback / default: Simple hybrid RAG
        if not context_chunks:
            raw_result = await self._retrieve_simple(user_query)
            context_chunks = raw_result.get("context", [])

        if not context_chunks:
            return {
                "answer": "No relevant context found for your query.",
                "verified_answer": "No relevant context found for your query.",
                "confidence": 0.0,
                "intent": routing.intent.value,
                "strategy": routing.strategy.value,
                "routing_reasoning": routing.reasoning,
                "claims": [],
            }

        # ---- Step 3: Generate draft answer ----
        draft_answer = raw_result.get("answer", "")

        if not draft_answer:
            draft_answer = self._generate_answer(user_query, context_chunks)

        # ---- Step 4: Verify for hallucinations ----
        claims = self.verifier.decompose(draft_answer)
        verified_claims = self.verifier.verify(claims, context_chunks)

        verified_answer, confidence = self.verifier.regenerate_verified(
            verified_claims, user_query
        )

        # ---- Build response ----
        return {
            "answer": draft_answer,
            "verified_answer": verified_answer,
            "confidence": confidence,
            "intent": routing.intent.value,
            "strategy": routing.strategy.value,
            "routing_reasoning": routing.reasoning,
            "claims": [
                {
                    "text": c.text,
                    "type": c.claim_type.value,
                    "verified": c.verified,
                    "method": c.verification_method,
                    "evidence": c.supporting_evidence,
                }
                for c in verified_claims
            ],
        }

    # ------------------------------------------------------------------
    # Retrieval backends
    # ------------------------------------------------------------------

    async def _retrieve_simple(self, query: str) -> dict:
        """Hybrid RAG retrieval via Qdrant (BM25 + Dense + Rerank)."""
        try:
            result = await self.rag_pipeline.query(
                query=query,
                persist_dir=self.collection_name,
            )
            context = [node["text"] for node in result.get("nodes", [])]
            return {
                "answer": result.get("answer", ""),
                "context": context,
            }
        except Exception as e:
            logger.error(f"Simple RAG retrieval failed: {e}")
            return {"answer": "", "context": []}

    async def _retrieve_graph(self, query: str) -> dict:
        """Graph RAG retrieval via Neo4j Cypher."""
        try:
            response = self.graph_rag.query(query)
            # GraphCypherQAChain returns a dict with 'result' key
            answer = response.get("result", str(response)) if isinstance(response, dict) else str(response)
            return {
                "answer": answer,
                "context": [answer],
            }
        except Exception as e:
            logger.error(f"Graph RAG retrieval failed: {e}")
            logger.info("Falling back to Simple RAG...")
            return await self._retrieve_simple(query)

    def _retrieve_tables(self, query: str) -> dict:
        """Table-aware RAG retrieval for numerical / comparison queries."""
        try:
            results = self.table_rag.retrieve(query, k=5)
            context = [r["content"] for r in results if r.get("content")]
            return {
                "answer": "",  # Let the LLM synthesise from table context
                "context": context,
            }
        except Exception as e:
            logger.error(f"Table RAG retrieval failed: {e}")
            return {"answer": "", "context": []}

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate_answer(self, query: str, context_chunks: list) -> str:
        """Generate a draft answer using the LLM."""
        context = "\n\n".join(context_chunks)

        prompt = f"""You are a helpful financial analyst AI assistant.

Answer the question based ONLY on the context below. 
If the context does not contain enough information, say so explicitly.
Cite specific numbers and dates from the context.

Context:
{context}

Question:
{query}

Answer clearly and concisely:"""

        try:
            messages = [
                {"role": "system", "content": "You are a helpful financial analyst AI assistant. Answer only from the provided context. If context is insufficient, say so. Cite specific numbers and dates."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer clearly and concisely:"},
            ]
            return self.model_loader.chat(messages)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return ""


# ============================================================================
# CLI Entry Point
# ============================================================================

def print_result(result: dict):
    """Pretty-print a query result."""
    print("\n" + "=" * 70)
    print(f"  Intent:    {result['intent']}")
    print(f"  Strategy:  {result['strategy']}")
    print(f"  Reasoning: {result['routing_reasoning']}")
    print("=" * 70)

    print(f"\n📝 Draft Answer:\n{result['answer']}\n")

    print("-" * 70)
    print(f"✅ Verified Answer (confidence: {result['confidence']:.2f}):")
    print(f"{result['verified_answer']}\n")

    if result["claims"]:
        print("-" * 70)
        print("🔍 Claim Verification:")
        for i, claim in enumerate(result["claims"], 1):
            status = "✓" if claim["verified"] else "✗"
            print(f"  {status} [{claim['type']}] {claim['text']}")
            if claim.get("evidence"):
                print(f"    └─ {claim['evidence']}")
    print("=" * 70)


async def main():
    if len(sys.argv) < 2:
        print("""
Usage:
  python main.py ingest <pdf_path> <collection_name>
  python main.py query "<question>" <collection_name>
  python main.py interactive <collection_name>
  python main.py evaluate [args...]
        """)
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "ingest":
        if len(sys.argv) < 4:
            print("Usage: python main.py ingest <pdf_path> <collection_name>")
            sys.exit(1)

        pdf_path = sys.argv[2]
        collection = sys.argv[3]

        orchestrator = FinanceRAGOrchestrator(collection_name=collection)
        await orchestrator.ingest(pdf_path)
        print("✅ Ingestion complete!")

    elif command == "query":
        if len(sys.argv) < 4:
            print("Usage: python main.py query \"<question>\" <collection_name>")
            sys.exit(1)

        question = sys.argv[2]
        collection = sys.argv[3]

        orchestrator = FinanceRAGOrchestrator(collection_name=collection)
        result = await orchestrator.query(question)
        print_result(result)

    elif command == "interactive":
        collection = sys.argv[2] if len(sys.argv) > 2 else "default"

        orchestrator = FinanceRAGOrchestrator(collection_name=collection)

        print("\n🏦 Finance RAG — Interactive Mode")
        print("Type 'quit' to exit, 'ingest <path>' to add a PDF.\n")

        while True:
            try:
                user_input = input("You > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            # Handle inline ingest command
            if user_input.lower().startswith("ingest "):
                pdf_path = user_input[7:].strip()
                await orchestrator.ingest(pdf_path)
                print("✅ Ingestion complete!\n")
                continue

            # Query
            result = await orchestrator.query(user_input)
            print_result(result)

    elif command == "evaluate":
        import evaluation
        # Adjust sys.argv so evaluation.py's argparse works correctly
        sys.argv.pop(1)
        sys.argv[0] = "evaluation.py"
        evaluation.main()

    else:
        print(f"Unknown command: {command}")
        print("Available commands: ingest, query, interactive, evaluate")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
