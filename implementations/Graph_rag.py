import os
import re
import loguru
from typing import List, Optional, Any
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import json
import re
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration


def _strip_markdown_json(text: str) -> str:
    """
    Remove markdown code fences and isolate JSON payload.
    """
    # Remove markdown code fences
    stripped = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    stripped = stripped.replace("```", "")

    # Look for the start of a JSON object or array
    match = re.search(r"[\[\{]", stripped)
    if match:
        stripped = stripped[match.start():]
    else:
        # If no JSON structure is found, return original text
        return text.strip()

    # Look for the end of the JSON structure (last closing bracket/brace)
    match_end = list(re.finditer(r"[\]\}]", stripped))
    if match_end:
        last_match = match_end[-1]
        stripped = stripped[:last_match.end()]

    return stripped.strip()


def _normalize_graph_schema(text: str) -> str:
    """
    Normalize graph schema for LLMGraphTransformer compatibility.

    Fixes:
    - label -> type (nodes)
    - edges -> relationships
    - label -> type (relationships)
    """
    try:
        data = json.loads(text)

        # Normalize nodes
        if "nodes" in data:
            for node in data["nodes"]:
                if "type" not in node:
                    node["type"] = node.get("label", "Entity")

        # Normalize edges -> relationships
        if "edges" in data and "relationships" not in data:
            relationships = []

            for edge in data["edges"]:
                relationships.append({
                    "source": edge["source"],
                    "target": edge["target"],
                    "type": edge.get("label", "RELATED_TO")
                })

            data["relationships"] = relationships
            del data["edges"]

        # Normalize relationships if label exists
        if "relationships" in data:
            for rel in data["relationships"]:
                if "type" not in rel:
                    rel["type"] = rel.get("label", "RELATED_TO")

        return json.dumps(data)

    except Exception:
        return text


class _MarkdownStrippingLLM:
    """
    Wrapper that strips markdown and normalizes graph schema.
    """

    def __init__(self, llm):
        self._llm = llm

    def __getattr__(self, name):
        return getattr(self._llm, name)

    def _clean_result(self, result):
        """
        Clean and normalize output.
        """

        if isinstance(result, AIMessage):
            loguru.logger.debug(f"RAW LLM MESSAGE: {result.content[:500]}...")
            cleaned = _strip_markdown_json(result.content)
            cleaned = _normalize_graph_schema(cleaned)
            loguru.logger.debug(f"CLEANED LLM MESSAGE: {cleaned[:500]}...")

            return AIMessage(
                content=cleaned,
                additional_kwargs=result.additional_kwargs
            )

        if isinstance(result, str):
            cleaned = _strip_markdown_json(result)
            cleaned = _normalize_graph_schema(cleaned)
            return cleaned

        if hasattr(result, "generations"):
            cleaned_gens = []

            for gen in result.generations:
                if isinstance(gen, list):
                    cleaned_gens.append(
                        [self._clean_gen(g) for g in gen]
                    )
                else:
                    cleaned_gens.append(self._clean_gen(gen))

            result.generations = cleaned_gens

        return result

    def _clean_gen(self, gen):
        if isinstance(gen, ChatGeneration):
            cleaned_msg = self._clean_result(gen.message)

            return ChatGeneration(
                message=cleaned_msg,
                generation_info=gen.generation_info
            )

        return gen

    def invoke(self, input, *args, **kwargs):
        # Reinforce JSON requirement for models that tend to chat (like Jamba)
        if isinstance(input, list) and len(input) > 0:
            # If it's a list of messages, modify the last one or add a new one
            from langchain_core.messages import HumanMessage
            if hasattr(input[-1], "content"):
                input[-1].content += "\n\nIMPORTANT: Respond ONLY with the JSON object. Do not include any Markdown formatting, headers, or explanations. The response must start with '{' and end with '}'."
        elif isinstance(input, str):
            input += "\n\nIMPORTANT: Respond ONLY with the JSON object. Do not include any Markdown formatting, headers, or explanations. The response must start with '{' and end with '}'."

        result = self._llm.invoke(input, *args, **kwargs)
        return self._clean_result(result)

    def generate(self, prompts, *args, **kwargs):
        modified_prompts = [
            p + "\n\nIMPORTANT: Respond ONLY with the JSON object. Do not include Markdown."
            for p in prompts
        ]
        result = self._llm.generate(modified_prompts, *args, **kwargs)
        return self._clean_result(result)

    def predict(self, text, *args, **kwargs):
        modified_text = text + "\n\nIMPORTANT: Respond ONLY with the JSON object. Do not include Markdown."
        result = self._llm.predict(modified_text, *args, **kwargs)
        return self._clean_result(result)

    def __or__(self, other):
        return self._llm.__or__(other)

    def bind(self, **kwargs):
        return _MarkdownStrippingLLM(
            self._llm.bind(**kwargs)
        )

    def with_structured_output(self, *args, **kwargs):
        return self._llm.with_structured_output(*args, **kwargs)


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
            # Wrap the LLM to strip markdown fences from its JSON output.
            # Some models (jamba, etc.) wrap responses in ```json...``` blocks
            # which causes Pydantic JSON validation to fail.
            wrapped_llm = _MarkdownStrippingLLM(self.llm)
            llm_transformer = LLMGraphTransformer(llm=wrapped_llm,
                                                 strict_mode=True)
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
