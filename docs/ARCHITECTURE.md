# Finance RAG Architecture - Deep Dive

Complete guide to building production-grade financial RAG systems with implementation details.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [HDRR - Hybrid Document-Routed Retrieval](#hrrd-hybrid-document-routed-retrieval)
3. [Agentic RAG Architecture](#agentic-rag-architecture)
4. [GraphRAG Implementation](#graphrag-implementation)
5. [Hybrid Retrieval + Reranking](#hybrid-retrieval-reranking)
6. [Hallucination Prevention](#hallucination-prevention)
7. [Production Deployment Guide](#production-deployment)

---

## Architecture Overview

### State-of-the-Art Finance RAG Stack (2025-2026)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           QUERY LAYER                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │ Query Parser │  │ Intent Class │  │ Entity Extract│                  │
│  │              │  │ (ticker/year)│  │ (MCP Protocol)│                  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │
│         └─────────────────┴─────────────────┘                           │
│                          │                                              │
└──────────────────────────┼──────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         ROUTING LAYER                                    │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │              Document Router (HDRR Stage 1)                  │        │
│  │  • Extract: ticker, fiscal_year, filing_type, section        │        │
│  │  • Match: Against document registry (FAISS/SQL)              │        │
│  │  • Fallback: Full corpus search if confidence < threshold    │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                          │                                              │
└──────────────────────────┼──────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        RETRIEVAL LAYER                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │ Vector Search│  │ Hybrid Search│  │ Graph Traversal│                 │
│  │ (FAISS/Neo4j)│  │ (BM25+Dense) │  │ (Cypher Query) │                 │
│  │ k=50 initial │  │ λ=0.7 semantic│ │ 2-hop entities │                 │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │
│         └─────────────────┴─────────────────┘                           │
│                          │                                              │
│                          ▼                                              │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │              Cross-Encoder Reranker                          │        │
│  │  • Model: ms-marco-MiniLM-L-6-v2 / ms-marco-electra-base    │        │
│  │  • Input: k=50 → Output: k=5                                │        │
│  │  • MMR for diversity (λ=0.3)                                │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                          │                                              │
└──────────────────────────┼──────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      VERIFICATION LAYER (FinGround)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │ Atomic Claim │  │ Type-Router  │  │ Verification │                  │
│  │ Decomposition│  │ (6 types)    │  │ + Regenerate │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
│                          │                                              │
└──────────────────────────┼──────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       GENERATION LAYER                                   │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  LLM with Structured Output                                  │        │
│  │  • Model: GPT-4o / Claude-Opus / Llama-3-70B                 │        │
│  │  • Citations: Table-cell level with hash                     │        │
│  │  • Confidence Score + Refusal on low confidence              │        │
│  └─────────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## HDRR - Hybrid Document-Routed Retrieval

### Why HDRR?

Standard RAG fails on financial documents because:
- Similar sections across companies have near-identical embeddings
- "Risk Factors" from Apple vs. Microsoft → embedding collision
- Cross-document chunk confusion leads to hallucinations

**HDRR Solution**: Route to document FIRST, then retrieve chunks.

### Implementation

```python
from typing import Optional, List, Dict
from pydantic import BaseModel
from neo4j import GraphDatabase
import faiss
import numpy as np

class DocumentRoute(BaseModel):
    """Structured output from document router"""
    ticker: str
    fiscal_year: int
    filing_type: str  # "10-K", "10-Q", "8-K"
    sections: List[str]  # ["Item 1A", "Item 7", "MD&A"]
    confidence: float

class DocumentRouter:
    """
    HDRR Stage 1: Route query to specific document(s)
    """
    
    def __init__(self, neo4j_uri: str, embedding_model):
        self.driver = GraphDatabase.driver(neo4j_uri)
        self.embedding_model = embedding_model
        self.document_index = self._build_document_index()
        
    def route(self, query: str) -> DocumentRoute:
        """Extract document metadata from query"""
        
        # Step 1: LLM extraction with structured output
        system_prompt = """
        Extract the target document(s) from this financial query.
        
        Return JSON with:
        - ticker: Company ticker symbol (e.g., "AAPL", "MSFT")
        - fiscal_year: Fiscal year (e.g., 2024)
        - filing_type: Type of SEC filing (10-K, 10-Q, 8-K)
        - sections: Relevant sections (Item 1A, Item 7, MD&A, etc.)
        
        If multiple documents, return the PRIMARY target.
        If unclear, set confidence < 0.5 to trigger fallback.
        """
        
        # Use LLM with structured output
        route = self._extract_with_llm(query, system_prompt)
        
        # Step 2: Validate against document registry
        validated = self._validate_route(route)
        
        return validated
    
    def _validate_route(self, route: DocumentRoute) -> DocumentRoute:
        """Check if document exists in registry"""
        
        cypher = """
        MATCH (d:Document {ticker: $ticker, fiscal_year: $year, filing_type: $type})
        RETURN d.document_id AS id, d.available_sections AS sections
        """
        
        with self.driver.session() as session:
            result = session.run(
                cypher,
                ticker=route.ticker,
                year=route.fiscal_year,
                type=route.filing_type
            )
            doc = result.single()
            
            if not doc:
                route.confidence = 0.3  # Trigger fallback
            else:
                # Filter sections to available ones
                route.sections = [
                    s for s in route.sections 
                    if s in doc["sections"]
                ]
                
        return route
    
    def retrieve(self, route: DocumentRoute, query: str, k: int = 10):
        """
        HDRR Stage 2: Scoped retrieval within routed document(s)
        """
        
        if route.confidence < 0.5:
            # Fallback: Full corpus search
            return self._full_corpus_search(query, k)
        
        # Scoped search within document
        cypher = """
        MATCH (d:Document {
            ticker: $ticker, 
            fiscal_year: $year, 
            filing_type: $type
        })-[:HAS_CHUNK]->(c:Chunk)
        WHERE c.section IN $sections
        WITH c, vector.similarity.cosine(c.embedding, $query_embedding) AS score
        ORDER BY score DESC
        LIMIT $k
        RETURN c.text AS text, c.section AS section, score
        """
        
        query_embedding = self.embedding_model.encode(query)
        
        with self.driver.session() as session:
            results = session.run(
                cypher,
                ticker=route.ticker,
                year=route.fiscal_year,
                type=route.filing_type,
                sections=route.sections,
                query_embedding=query_embedding.tolist(),
                k=k * 2  # Retrieve more for reranking
            )
            
            return [(r["text"], r["section"], r["score"]) for r in results]
```

---

## Agentic RAG Architecture

### Multi-Agent Workflow for Complex Queries

For queries requiring multi-hop reasoning:
- "Compare Apple's 2024 risk factors to Microsoft's"
- "What caused the revenue change in Q3 2024?"

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator

class AgentState(TypedDict):
    """State passed between agents"""
    query: str
    plan: List[str]
    retrieved_context: List[dict]
    generated_answer: str
    verification_status: str
    confidence: float

class AgenticRAG:
    """
    Multi-agent orchestration for complex financial queries
    """
    
    def __init__(self):
        self.graph = self._build_graph()
        
    def _build_graph(self):
        """Build LangGraph workflow"""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes (agents)
        workflow.add_node("planner", self.planner_agent)
        workflow.add_node("researcher", self.researcher_agent)
        workflow.add_node("analyst", self.analyst_agent)
        workflow.add_node("verifier", self.verifier_agent)
        
        # Define edges
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "researcher")
        workflow.add_edge("researcher", "analyst")
        workflow.add_edge("analyst", "verifier")
        
        # Conditional edge: loop back if verification fails
        workflow.add_conditional_edges(
            "verifier",
            self._check_verification,
            {
                "pass": END,
                "retry": "researcher"
            }
        )
        
        return workflow.compile()
    
    def planner_agent(self, state: AgentState):
        """
        Break query into sub-questions (ReAct-style)
        """
        prompt = f"""
        Break this financial query into atomic sub-questions:
        
        Query: {state['query']}
        
        For each sub-question, specify:
        1. What information is needed
        2. Which document(s) to search
        3. What type of reasoning is required
        
        Return as JSON list.
        """
        
        plan = self._call_llm(prompt)
        return {"plan": plan}
    
    def researcher_agent(self, state: AgentState):
        """
        Execute retrieval for each sub-question
        Uses HDRR + Hybrid Search
        """
        contexts = []
        
        for sub_question in state['plan']:
            # Route to document
            route = self.document_router.route(sub_question)
            
            # Retrieve with hybrid search
            chunks = self.hybrid_retriever.search(
                query=sub_question,
                document_route=route,
                k=10
            )
            
            contexts.append({
                "question": sub_question,
                "chunks": chunks
            })
        
        return {"retrieved_context": contexts}
    
    def analyst_agent(self, state: AgentState):
        """
        Synthesize retrieved context into answer
        """
        context_text = self._format_context(state['retrieved_context'])
        
        prompt = f"""
        Answer the query using ONLY the provided context.
        
        Query: {state['query']}
        
        Context:
        {context_text}
        
        Requirements:
        - Cite specific sections (e.g., "AAPL 10-K 2024, Item 1A, page 23")
        - Include numerical values with units
        - Flag any missing information
        - Return confidence score (0-1)
        """
        
        answer = self._call_llm(prompt)
        confidence = self._extract_confidence(answer)
        
        return {
            "generated_answer": answer,
            "confidence": confidence
        }
    
    def verifier_agent(self, state: AgentState):
        """
        FinGround-style atomic claim verification
        """
        claims = self._decompose_claims(state['generated_answer'])
        verified_claims = []
        
        for claim in claims:
            verification = self._verify_claim(
                claim=claim,
                context=state['retrieved_context']
            )
            verified_claims.append(verification)
        
        # Regenerate with only verified claims
        verified_answer = self._regenerate_verified(verified_claims)
        
        all_verified = all(c["verified"] for c in verified_claims)
        
        return {
            "generated_answer": verified_answer,
            "verification_status": "pass" if all_verified else "retry",
        }
    
    def _check_verification(self, state: AgentState) -> str:
        """Decide whether to accept or retry"""
        if state['verification_status'] == "pass":
            return "pass"
        elif state.get('retry_count', 0) < 2:
            return "retry"
        else:
            return "pass"  # Force pass after 2 retries
    
    def execute(self, query: str) -> dict:
        """Run the full agentic workflow"""
        
        initial_state = {
            "query": query,
            "plan": [],
            "retrieved_context": [],
            "generated_answer": "",
            "verification_status": "",
            "confidence": 0.0
        }
        
        result = self.graph.invoke(initial_state)
        return result
```

---

## GraphRAG Implementation

### Two-Layer Knowledge Graph Architecture

Based on Neo4j's financial services implementation.

#### Step 1: Build Knowledge Graph

```python
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.kg_construction import SimpleKGPipeline
from neo4j import GraphDatabase

class FinancialKGBuilder:
    """
    Build knowledge graph from SEC filings
    """
    
    def __init__(self, neo4j_uri: str, llm: LLMInterface):
        self.driver = GraphDatabase.driver(neo4j_uri)
        self.llm = llm
        
    def build_graph(self, filing_text: str, metadata: dict):
        """
        Create two-layer graph:
        - Lexical layer: Chunk nodes
        - Semantic layer: Entity nodes
        """
        
        # Configure entity extraction
        schema = {
            "Company": {
                "properties": {
                    "ticker": "string (required)",
                    "name": "string",
                    "sector": "string",
                    "industry": "string"
                }
            },
            "FinancialMetric": {
                "properties": {
                    "name": "string (e.g., 'Revenue', 'Net Income')",
                    "value": "float",
                    "unit": "string (USD, percentage)",
                    "period": "string (Q1 2024, FY2024)"
                }
            },
            "RiskFactor": {
                "properties": {
                    "category": "string (Market, Credit, Operational, Legal)",
                    "severity": "string (High, Medium, Low)",
                    "description": "string"
                }
            },
            "Product": {
                "properties": {
                    "name": "string",
                    "category": "string"
                }
            }
        }
        
        # Build pipeline
        pipeline = SimpleKGPipeline(
            llm=self.llm,
            driver=self.driver,
            db_vector_index_name="chunk_embedding",
            from_llm_to_graph=True,
            schema=schema,
            chunk_size=512,
            chunk_overlap=64
        )
        
        # Process document
        pipeline.process_text(filing_text, metadata)
        
        # Create indexes
        self._create_indexes()
        
    def _create_indexes(self):
        """Create vector and fulltext indexes"""
        
        with self.driver.session() as session:
            # Vector index on Chunk embeddings
            session.run("""
                CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 1536,
                    `vector.similarity_function`: 'cosine'
                }}
            """)
            
            # Fulltext index for hybrid search
            session.run("""
                CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS
                FOR (c:Chunk) ON EACH [c.text]
            """)
            
            # Entity indexes
            session.run("CREATE INDEX company_ticker IF NOT EXISTS FOR (c:Company) ON (c.ticker)")
            session.run("CREATE INDEX metric_name IF NOT EXISTS FOR (m:FinancialMetric) ON (m.name)")
```

#### Step 2: Hybrid Cypher Retriever

```python
from neo4j_graphrag.retrievers import VectorRetriever, HybridCypherRetriever

class GraphRAGRetriever:
    """
    Retrieve from GraphRAG with multiple strategies
    """
    
    def __init__(self, driver, embedding_provider):
        self.driver = driver
        
        # Vector retriever (baseline)
        self.vector_retriever = VectorRetriever(
            driver=driver,
            index_name="chunk_embedding",
            embedding_provider=embedding_provider
        )
        
        # Hybrid retriever (vector + fulltext)
        self.hybrid_retriever = HybridCypherRetriever(
            driver=driver,
            index_name="chunk_embedding",
            fulltext_index_name="chunk_fulltext",
            embedding_provider=embedding_provider,
            return_properties=["text", "section", "page_number"]
        )
    
    def retrieve_basic(self, query: str, k: int = 5):
        """Simple vector retrieval"""
        return self.vector_retriever.search(query=query, top_k=k)
    
    def retrieve_hybrid(self, query: str, k: int = 10, alpha: float = 0.7):
        """
        Hybrid retrieval: vector + fulltext with reciprocal rank fusion
        
        alpha=0.7: 70% weight on semantic, 30% on keyword
        """
        return self.hybrid_retriever.search(
            query=query,
            top_k=k,
            alpha=alpha
        )
    
    def retrieve_with_cypher(self, query: str, custom_query: str, k: int = 10):
        """
        Custom Cypher retrieval for complex queries
        
        Example custom_query:
        MATCH (c:Company {ticker: 'AAPL'})-[:HAS_RISK]->(r:RiskFactor)
        MATCH (c)-[:HAS_CHUNK]->(chunk:Chunk)
        WHERE r.category = 'Market'
        RETURN chunk.text AS text, r.severity AS severity
        """
        
        embedding = self.embedding_provider.get_query_embedding(query)
        
        with self.driver.session() as session:
            result = session.run(
                custom_query,
                query_embedding=embedding,
                k=k
            )
            return [dict(r) for r in result]
    
    def retrieve_entity_context(self, ticker: str, entity_type: str = "RiskFactor"):
        """
        Retrieve all context around a specific entity
        
        Useful for: "What are all of Apple's market risks?"
        """
        
        cypher = f"""
        MATCH (c:Company {{ticker: $ticker}})-[:HAS_{entity_type.upper()}]->(e:{entity_type})
        MATCH (e)<-[:MENTIONS]-(chunk:Chunk)
        RETURN chunk.text AS text, 
               chunk.section AS section,
               chunk.page_number AS page,
               e.category AS category,
               e.severity AS severity
        ORDER BY chunk.page_number
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, ticker=ticker)
            return [dict(r) for r in result]
```

---

## Hybrid Retrieval + Reranking

### Production-Ready Implementation

```python
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
import numpy as np
from typing import List, Tuple

class HybridRetrieverWithReranking:
    """
    Hybrid retrieval with cross-encoder reranking
    
    Architecture:
    1. BM25 (keyword) + Dense (semantic) → k=50 candidates
    2. Reciprocal Rank Fusion for score combination
    3. Cross-encoder reranking → k=5 final
    4. MMR for diversity
    """
    
    def __init__(
        self,
        documents: List[str],
        dense_model: str = "BAAI/bge-large-en-v1.5",
        cross_encoder_model: str = "cross-encoder/ms-marco-electra-base"
    ):
        # Dense retrieval
        self.dense_model = SentenceTransformer(dense_model)
        self.doc_embeddings = self.dense_model.encode(documents)
        
        # BM25 retrieval
        tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Cross-encoder reranker
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        
        # Store documents
        self.documents = documents
        self.doc_scores_cache = {}
    
    def search(
        self, 
        query: str, 
        k_dense: int = 30, 
        k_bm25: int = 30,
        k_final: int = 5,
        alpha: float = 0.7,
        diversity: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Hybrid search with reranking
        
        Args:
            query: Search query
            k_dense: Number of dense retrieval candidates
            k_bm25: Number of BM25 candidates
            k_final: Final number of results after reranking
            alpha: Weight for dense scores (1-alpha for BM25)
            diversity: MMR diversity parameter (0=diverse, 1=relevant)
        
        Returns:
            List of (document, score) tuples
        """
        
        # Step 1: Dense retrieval
        query_embedding = self.dense_model.encode(query)
        dense_scores = np.dot(self.doc_embeddings, query_embedding)
        dense_indices = np.argsort(dense_scores)[::-1][:k_dense]
        
        # Step 2: BM25 retrieval
        query_tokens = query.split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_indices = np.argsort(bm25_scores)[::-1][:k_bm25]
        
        # Step 3: Reciprocal Rank Fusion
        fused_scores = self._reciprocal_rank_fusion(
            dense_indices, bm25_indices, alpha
        )
        
        # Get candidate documents
        candidate_indices = list(set(dense_indices) | set(bm25_indices))
        candidate_docs = [self.documents[i] for i in candidate_indices]
        
        # Step 4: Cross-encoder reranking
        pairs = [[query, doc] for doc in candidate_docs]
        cross_scores = self.cross_encoder.predict(pairs)
        
        # Rank by cross-encoder scores
        reranked_indices = np.argsort(cross_scores)[::-1]
        
        # Step 5: MMR for diversity
        final_indices = self._maximum_marginal_relevance(
            candidate_indices,
            query_embedding,
            k_final,
            diversity
        )
        
        # Return with scores
        results = [
            (self.documents[i], cross_scores[candidate_indices.index(i)])
            for i in final_indices
        ]
        
        return results
    
    def _reciprocal_rank_fusion(
        self, 
        dense_indices: np.ndarray, 
        bm25_indices: np.ndarray,
        alpha: float,
        k: int = 60
    ) -> dict:
        """
        Combine rankings using Reciprocal Rank Fusion (RRF)
        
        RRF score = sum(1 / (k + rank)) for each ranking
        """
        
        scores = {}
        
        for rank, idx in enumerate(dense_indices):
            scores[idx] = scores.get(idx, 0) + alpha / (k + rank + 1)
        
        for rank, idx in enumerate(bm25_indices):
            scores[idx] = scores.get(idx, 0) + (1 - alpha) / (k + rank + 1)
        
        return scores
    
    def _maximum_marginal_relevance(
        self,
        candidate_indices: List[int],
        query_embedding: np.ndarray,
        k: int,
        diversity: float
    ) -> List[int]:
        """
        Maximum Marginal Relevance for diversity
        
        MMR = argmax [ λ * relevance - (1-λ) * max_similarity_to_selected ]
        """
        
        selected = []
        remaining = candidate_indices.copy()
        
        # Get query relevance
        query_similarities = {
            idx: np.dot(self.doc_embeddings[idx], query_embedding)
            for idx in candidate_indices
        }
        
        while len(selected) < k and remaining:
            best_mmr = -float('inf')
            best_idx = None
            
            for idx in remaining:
                # Relevance component
                relevance = query_similarities[idx]
                
                # Diversity component
                if selected:
                    max_similarity = max(
                        np.dot(self.doc_embeddings[idx], self.doc_embeddings[s])
                        for s in selected
                    )
                else:
                    max_similarity = 0
                
                # MMR score
                mmr = diversity * relevance - (1 - diversity) * max_similarity
                
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx
            
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        return selected
```

---

## Hallucination Prevention

### FinGround-Style Atomic Claim Verification

```python
from enum import Enum
from typing import List, Dict, Optional
from pydantic import BaseModel

class ClaimType(str, Enum):
    """6 types of financial claims (FinGround taxonomy)"""
    NUMERICAL = "numerical"  # "Revenue was $383B"
    TEMPORAL = "temporal"    # "In Q3 2024..."
    ENTITY_ATTRIBUTE = "entity_attribute"  # "Apple is a technology company"
    COMPARATIVE = "comparative"  # "Revenue increased 5% YoY"
    REGULATORY = "regulatory"  # "Subject to SEC Rule 10-K"
    COMPUTATIONAL = "computational"  # "Gross margin = 44.1%"

class AtomicClaim(BaseModel):
    """Decomposed claim for verification"""
    text: str
    claim_type: ClaimType
    confidence: float
    supporting_evidence: Optional[str] = None
    verified: bool = False
    verification_method: Optional[str] = None

class FinGroundVerifier:
    """
    FinGround-style hallucination detection and prevention
    
    Achieves 78% hallucination reduction with 91.4% F1 detection
    """
    
    def __init__(self, llm, embedding_model):
        self.llm = llm
        self.embedding_model = embedding_model
    
    def decompose(self, answer: str) -> List[AtomicClaim]:
        """
        Decompose answer into atomic claims
        """
        
        prompt = f"""
        Decompose this financial answer into atomic claims.
        
        Answer: {answer}
        
        For each claim:
        1. Extract the minimal complete statement
        2. Classify the type (numerical, temporal, entity_attribute, comparative, regulatory, computational)
        3. Note any numbers, dates, or formulas
        
        Return as JSON list with fields:
        - text: The claim text
        - claim_type: One of the 6 types
        - has_numbers: Boolean
        - has_dates: Boolean
        - formula: If computational, the formula
        """
        
        claims_data = self._call_llm_json(prompt)
        
        claims = [
            AtomicClaim(**claim) 
            for claim in claims_data
        ]
        
        return claims
    
    def verify_claim(
        self, 
        claim: AtomicClaim, 
        context: List[str]
    ) -> AtomicClaim:
        """
        Verify a single claim against retrieved context
        """
        
        if claim.claim_type == ClaimType.NUMERICAL:
            verified, evidence = self._verify_numerical(claim, context)
        elif claim.claim_type == ClaimType.COMPARATIVE:
            verified, evidence = self._verify_comparative(claim, context)
        elif claim.claim_type == ClaimType.COMPUTATIONAL:
            verified, evidence = self._verify_computational(claim, context)
        else:
            verified, evidence = self._verify_generic(claim, context)
        
        claim.verified = verified
        claim.supporting_evidence = evidence
        claim.verification_method = f"verify_{claim.claim_type.value}"
        
        return claim
    
    def _verify_numerical(self, claim: AtomicClaim, context: List[str]):
        """
        Verify numerical claims with fuzzy matching
        
        Key: Values within ±5% may be hallucinations (hard to detect)
        """
        
        # Extract number from claim
        number = self._extract_number(claim.text)
        
        # Search context for matching numbers
        for ctx in context:
            ctx_numbers = self._extract_all_numbers(ctx)
            
            for ctx_num in ctx_numbers:
                # Exact match
                if abs(ctx_num - number) / number < 0.01:
                    return True, f"Found {ctx_num} in context"
                
                # Near match (flag for review)
                if abs(ctx_num - number) / number < 0.05:
                    return False, f"Near match: {ctx_num} vs {number}"
        
        return False, "No matching number found"
    
    def _verify_computational(self, claim: AtomicClaim, context: List[str]):
        """
        Verify computational claims by recomputing
        
        Example: "Gross margin = 44.1%"
        → Find revenue and COGS in context
        → Recompute: (Revenue - COGS) / Revenue
        """
        
        # Extract formula and claimed result
        formula = claim.formula or self._infer_formula(claim.text)
        claimed_value = self._extract_number(claim.text)
        
        # Find component values in context
        components = self._find_formula_components(formula, context)
        
        if components:
            # Recompute
            computed = self._evaluate_formula(formula, components)
            
            if abs(computed - claimed_value) / claimed_value < 0.01:
                return True, f"Computed: {computed}"
            else:
                return False, f"Computed: {computed}, claimed: {claimed_value}"
        
        return False, "Could not find formula components"
    
    def regenerate_verified(self, claims: List[AtomicClaim]) -> str:
        """
        Regenerate answer with only verified claims
        """
        
        verified_claims = [c.text for c in claims if c.verified]
        unverified_claims = [c.text for c in claims if not c.verified]
        
        if not verified_claims:
            return "I cannot verify any claims from the retrieved context. I cannot provide a reliable answer."
        
        prompt = f"""
        Regenerate the answer using ONLY these verified claims:
        
        Verified: {verified_claims}
        Unverified (exclude): {unverified_claims}
        
        Requirements:
        - Include citations for each claim
        - Flag any gaps from unverified claims
        - Return confidence score
        """
        
        return self.llm.generate(prompt)
```

---

## Production Deployment

### Complete System Architecture

```yaml
# docker-compose.yml for production RAG stack

version: '3.8'

services:
  # Neo4j with GraphRAG plugins
  neo4j:
    image: neo4j:5.23-enterprise
    environment:
      NEO4J_AUTH: neo4j/password
      NEO4J_PLUGINS: '["graph-data-science", "vector"]'
      NEO4J_dbms_security_procedures_unrestricted: 'vector.*'
    volumes:
      - neo4j_data:/data
    ports:
      - "7687:7687"
      - "7474:7474"
  
  # PostgreSQL with PGVector
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: rag_user
      POSTGRES_PASSWORD: secure_password
      POSTGRES_DB: rag_db
    volumes:
      - pg_data:/var/lib/postgresql/data
  
  # Redis for caching
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
  
  # API Server (FastAPI)
  api:
    build: ./api
    environment:
      NEO4J_URI: neo4j://neo4j:7687
      DATABASE_URL: postgresql://rag_user:secure_password@postgres/rag_db
      REDIS_URL: redis://redis:6379
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      RERANK_MODEL: cross-encoder/ms-marco-electra-base
    depends_on:
      - neo4j
      - postgres
      - redis
  
  # Document ingestion worker
  worker:
    build: ./worker
    environment:
      NEO4J_URI: neo4j://neo4j:7687
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    depends_on:
      - neo4j

volumes:
  neo4j_data:
  pg_data:
  redis_data:
```

### Monitoring & Evaluation

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

class RAGEvaluator:
    """
    Continuous evaluation pipeline
    """
    
    def __init__(self):
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
    
    def evaluate_batch(self, test_dataset):
        """
        Evaluate RAG system on test queries
        
        test_dataset: List of {query, ground_truth, contexts}
        """
        
        results = evaluate(
            test_dataset,
            metrics=self.metrics,
            llm="gpt-4o",
            embeddings="text-embedding-3-large"
        )
        
        return results
    
    def check_drift(self, current_metrics: dict, baseline_metrics: dict):
        """
        Detect performance drift
        
        Alert if any metric drops > 10% from baseline
        """
        
        alerts = []
        
        for metric_name in self.metrics:
            baseline = baseline_metrics[metric_name]
            current = current_metrics[metric_name]
            
            if (baseline - current) / baseline > 0.10:
                alerts.append({
                    "metric": metric_name,
                    "baseline": baseline,
                    "current": current,
                    "drop_pct": (baseline - current) / baseline * 100
                })
        
        return alerts
```

---

## Quick Start Implementation

### Minimal Working Example

```python
# Install dependencies
# pip install neo4j-graphrag[openai] sentence-transformers rank-bm25

from neo4j_graphrag.neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorRetriever
from sentence_transformers import SentenceTransformer

# Initialize
driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "password"))
embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Ingest document
filing_text = open("aapl_10k_2024.txt").read()

# Build index (one-time)
cypher = """
UNWIND $chunks AS chunk
MERGE (c:Chunk {id: chunk.id})
SET c.text = chunk.text,
    c.section = chunk.section,
    c.embedding = chunk.embedding
"""

# Chunk and embed
chunks = []
for i, chunk_text in enumerate(chunk_text(filing_text, 512, 64)):
    embedding = embedding_model.encode(chunk_text)
    chunks.append({
        "id": f"chunk_{i}",
        "text": chunk_text,
        "section": extract_section(chunk_text),
        "embedding": embedding.tolist()
    })

driver.execute_query(cypher, chunks=chunks)

# Query
retriever = VectorRetriever(driver, "chunk_embedding", embedding_model)
results = retriever.search(query="What are Apple's key risk factors?", top_k=5)

for result in results:
    print(result.text)
    print(f"Section: {result.section}")
    print("---")
```

---

## References

- [Neo4j GraphRAG for SEC Filings](https://github.com/neo4j-partners/sample-graphrag)
- [FinGround Hallucination Prevention](https://arxiv.org/html/2604.23588v1)
- [PwC RAG Architecture Study](https://arxiv.org/pdf/2511.18177)
- [FinRAG Hybrid Implementation](https://github.com/aatmaj28/FinRAG)
- [SEC Insights by LlamaIndex](https://github.com/run-llama/sec-insights)
