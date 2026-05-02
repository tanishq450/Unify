import os
import loguru
from typing import List, Optional, Any
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer

class GRAPH_RAG:
    """
    Graph RAG implementation using Neo4j and LangChain.
    Handles document ingestion into a knowledge graph and Cypher QA generation.
    """
    
    def __init__(self, llm: Optional[Any] = None):
        self.log = loguru.logger
        self.llm = llm
        self.graph = None
        self._init_graph()

    def _init_graph(self):
        """Initialize Neo4j graph connection"""
        try:
            # Fallbacks are used so it doesn't crash on import if env vars are missing
            self.graph = Neo4jGraph(
                url=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                username=os.environ.get("NEO4J_USERNAME", "neo4j"),
                password=os.environ.get("NEO4J_PASSWORD", "Tanishq(@)10"),
                enhanced_schema=True,   # Richer schema for Cypher generation
                sanitize=True,          # Clean unsafe characters
                refresh_schema=True,    # Always fetch latest schema 
            )
            self.log.info("Successfully connected to Neo4j Graph.")
        except Exception as e:
            self.log.error(f"Failed to connect to Neo4j Graph: {str(e)}")

    def load_llm(self, llm_instance: Any):
        """
        Load an LLM instance dynamically.
        Args:
            llm_instance: A LangChain compatible LLM instance.
        """
        try:
            self.llm = llm_instance
            self.log.info("MODEL LOADED SUCCESSFULLY")
        except Exception as e:
            self.log.error(f"PROBLEM LOADING MODEL: {str(e)}")
    
    def convert_docs(self, text: str) -> List[Document]:
        """
        Convert raw string text into LangChain Document format.
        """
        try:
            documents = [Document(page_content=text)]
            self.log.info("Documents successfully loaded.")
            return documents
        except Exception as e:
            self.log.error(f"Problem converting to Document: {str(e)}")
            return []

    def make_graph(self, documents: List[Document]):
        """
        Transform documents into graph nodes/relationships and ingest into Neo4j.
        """
        if not self.llm:
            self.log.error("LLM not loaded. Cannot transform documents.")
            return None
            
        if not self.graph:
            self.log.error("Graph not connected. Cannot ingest documents.")
            return None

        try:
            llm_transformer = LLMGraphTransformer(llm=self.llm)
            graph_documents = llm_transformer.convert_to_graph_documents(documents)
            self.graph.add_graph_documents(graph_documents)
            self.log.info("TRANSFORMATION AND GRAPH INGESTION COMPLETE")
            return graph_documents
        except Exception as e:
            self.log.error(f"ERROR DURING GRAPH TRANSFORMATION PROCESS: {str(e)}")
            return None

    def get_graph_chain(self) -> Optional[GraphCypherQAChain]:
        """
        Create and return the GraphCypherQAChain.
        """
        if not self.llm or not self.graph:
            self.log.error("LLM or Graph not initialized. Cannot create chain.")
            return None
            
        try:
            chain = GraphCypherQAChain.from_llm(
                llm=self.llm,
                graph=self.graph,
                verbose=True, 
                allow_dangerous_requests=True
            )
            self.log.info("CHAIN SUCCESSFULLY CREATED")
            return chain
        except Exception as e:
            self.log.error(f"ERROR OCCURRED CREATING CHAIN: {str(e)}")
            return None
            
    def query(self, question: str) -> str:
        """
        Utility method to directly query the graph RAG.
        """
        chain = self.get_graph_chain()
        if not chain:
            return "Unable to answer query. System not properly initialized."
            
        try:
            self.log.info(f"Querying graph RAG: {question}")
            response = chain.invoke({"query": question})
            return response
        except Exception as e:
            self.log.error(f"Error executing query: {str(e)}")
            return f"Error executing query: {str(e)}"
