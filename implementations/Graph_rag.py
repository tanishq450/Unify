import re
import json
import os
from typing import Any, Optional, List

import loguru
from langchain_core.runnables import RunnableSerializable, RunnableLambda
from langchain_core.messages import AIMessage
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from json_repair import repair_json




# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` fences from a string."""
    if not isinstance(text, str):
        return text
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _normalize_graph_schema(text: str) -> str:
    """
    Normalize Claude's JSON output to match DynamicGraph's Pydantic schema.

    Claude returns:
        {
            "entities": [{"id": "X", "type": "Y"}, ...],
            "relationships": [{"source": "X", "target": "Y", "type": "REL"}, ...]
        }

    DynamicGraph expects:
        {
            "nodes": [{"id": "X", "type": "Y"}, ...],
            "relationships": [
                {
                    "source_node_id": "X", "source_node_type": "Y",
                    "target_node_id": "A", "target_node_type": "B",
                    "type": "REL"
                }, ...
            ]
        }
    """
    try:
      

        text = repair_json(text)
        data = json.loads(text)

        # ── 1. entities -> nodes ──────────────────────────────────────
        if "entities" in data and "nodes" not in data:
            data["nodes"] = [
                {
                    "id":   e.get("id", e.get("name", "")),
                    "type": e.get("type", e.get("label", "Entity")),
                }
                for e in data["entities"]
            ]
            del data["entities"]

        # ── 2. fix existing nodes (label -> type, name -> id) ─────────
        for node in data.get("nodes", []):
            if "type" not in node and "label" in node:
                node["type"] = node["label"]
            if "id" not in node and "name" in node:
                node["id"] = node["name"]

        # ── 3. edges -> relationships ─────────────────────────────────
        if "edges" in data and "relationships" not in data:
            data["relationships"] = data.pop("edges")

        # ── 4. remap relationship fields to DynamicGraph schema ───────
        # Claude sends:  {"source": "X", "target": "Y", "type": "REL"}
        # DynamicGraph:  {"source_node_id": "X", "source_node_type": "?",
        #                 "target_node_id": "Y", "target_node_type": "?",
        #                 "type": "REL"}
        node_type_map = {
            n["id"]: n.get("type", "Entity")
            for n in data.get("nodes", [])
            if "id" in n
        }

        fixed_rels = []
        for rel in data.get("relationships", []):
            src      = rel.get("source",      rel.get("source_node_id", ""))
            tgt      = rel.get("target",      rel.get("target_node_id", ""))
            rel_type = rel.get("type",        rel.get("label", "RELATED_TO"))

            fixed_rels.append({
                "source_node_id":   src,
                "source_node_type": node_type_map.get(src, "Entity"),
                "target_node_id":   tgt,
                "target_node_type": node_type_map.get(tgt, "Entity"),
                "type":             rel_type,
            })

        data["relationships"] = fixed_rels

        return json.dumps(data)

    except Exception:
        return text


def _extract_content_from_claude(result: Any) -> Optional[str]:
    """
    Extract JSON string from whatever format Claude returns.
    Returns None if result is already a parsed Pydantic model.
    """
    # Case 1: plain string content
    if isinstance(result, AIMessage) and isinstance(result.content, str):
        return result.content

    # Case 2: Claude tool_use block — content is a list of dicts
    if isinstance(result, AIMessage) and isinstance(result.content, list):
        for block in result.content:
            if isinstance(block, dict):
                if block.get("type") == "tool_use" and "input" in block:
                    return json.dumps(block["input"])
                if block.get("type") == "text" and "text" in block:
                    return block["text"]

    # Case 3: already a plain dict
    if isinstance(result, dict):
        return json.dumps(result)

    # Case 4: Pydantic model — already parsed, signal with None
    # covers Pydantic v2 (model_fields) and v1 (__fields__)
    if hasattr(result, "model_fields") or hasattr(result, "__fields__"):
        return None

    return str(result)


def _clean_claude_output(result: Any) -> AIMessage:
    """
    Extract, strip fences, normalize schema, and wrap in AIMessage.
    Used by invoke() and ainvoke() which must always return an AIMessage.
    """
    content = _extract_content_from_claude(result)

    # None means result is already a Pydantic model — serialize it back
    # to a string so the rest of the pipeline gets a valid AIMessage
    if content is None:
        content = (
            result.model_dump_json()
            if hasattr(result, "model_dump_json")
            else str(result)
        )

    content = _strip_markdown_fences(content)
    content = _normalize_graph_schema(content)

    additional_kwargs = (
        result.additional_kwargs
        if hasattr(result, "additional_kwargs")
        else {}
    )
    return AIMessage(content=content, additional_kwargs=additional_kwargs)


# ---------------------------------------------------------------------------
# LLM Wrapper — Claude specific
# ---------------------------------------------------------------------------


from typing import Any
from langchain_core.runnables import RunnableSerializable, RunnableLambda


class MarkdownStrippingLLM(RunnableSerializable):
    """
    Claude-safe wrapper for LLMGraphTransformer.

    Fixes:
    - strips markdown fences BEFORE validation
    - normalizes graph schema
    - handles include_raw contract
    - avoids premature validation
    """

    llm: Any

    def invoke(self, input, config=None, **kwargs):
        result = self.llm.invoke(input, config=config, **kwargs)

        content = _extract_content_from_claude(result)

        if content is None:
            return str(result)

        content = _strip_markdown_fences(content)
        content = _normalize_graph_schema(content)

        return content

    async def ainvoke(self, input, config=None, **kwargs):
        result = await self.llm.ainvoke(input, config=config, **kwargs)

        content = _extract_content_from_claude(result)

        if content is None:
            return str(result)

        content = _strip_markdown_fences(content)
        content = _normalize_graph_schema(content)

        return content

    def with_structured_output(self, schema, **kwargs):
        """
        Manual structured parsing.
        Avoids native structured output because it validates too early.
        """

        include_raw = kwargs.get("include_raw", False)

        raw_llm = self.llm   # raw model only

        def clean_and_parse(result):
            raw_result = result

            # Extract content
            content = _extract_content_from_claude(raw_result)

            if content is None:
                raise ValueError(
                    "Failed to extract content from model output"
                )

            # CLEAN FIRST
            content = _strip_markdown_fences(content)

            # NORMALIZE SECOND
            content = _normalize_graph_schema(content)

            # VALIDATE LAST
            if hasattr(schema, "model_validate_json"):
                parsed = schema.model_validate_json(content)
            else:
                parsed = schema.parse_raw(content)

            if include_raw:
                return {
                    "raw": raw_result,
                    "parsed": parsed,
                    "parsing_error": None,
                }

            return parsed

        # IMPORTANT:
        # raw model → cleanup → validation
        return raw_llm | RunnableLambda(clean_and_parse)

    @property
    def InputType(self):
        return Any

    @property
    def OutputType(self):
        return Any

 
        
# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GRAPH_RAG:
    """
    Graph RAG using Neo4j + LangChain + Claude.

    Quick-start
    -----------
    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")

    rag = GRAPH_RAG()
    rag.load_llm(llm)

    docs = rag.convert_docs(your_long_text)
    rag.make_graph(docs)

    answer = rag.query("Who are the main entities?")
    print(answer)
    """

    def __init__(self, llm: Optional[Any] = None):
        self.log = loguru.logger
        self.llm = llm
        self.graph: Optional[Neo4jGraph] = None
        self._init_graph()

    def _init_graph(self):
        """Initialise Neo4j connection from environment variables."""
        try:
            self.graph = Neo4jGraph(
                url=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                username=os.environ.get("NEO4J_USERNAME", "neo4j"),
                password=os.environ.get("NEO4J_PASSWORD", ""),
                enhanced_schema=True,
                sanitize=True,
                refresh_schema=True,
            )
            self.log.info("Successfully connected to Neo4j.")
        except Exception as e:
            self.log.error(f"Neo4j connection failed: {e}")
            self.graph = None

    def load_llm(self, llm_instance: Any):
        """Attach an LLM instance (ChatAnthropic, ChatOpenAI, etc.)."""
        self.llm = llm_instance
        self.log.info("LLM loaded successfully.")

    def convert_docs(
        self,
        text: str,
        chunk_size: int = 4000,
        chunk_overlap: int = 400,
    ) -> List[Document]:
        """Split text into overlapping chunks and return LangChain Documents."""
        try:
            try:
                from langchain_text_splitters import RecursiveCharacterTextSplitter
            except ImportError:
                from langchain.text_splitter import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            docs = splitter.create_documents([text])
            self.log.info(f"Created {len(docs)} chunks.")
            return docs

        except Exception as e:
            self.log.error(f"Chunking failed: {e}")
            return []

    def make_graph(
        self,
        documents: List[Document],
        max_docs: int = 30,
    ) -> Optional[list]:
        """
        Extract entities & relationships from documents and store in Neo4j.
        """
        if not self.llm:
            self.log.error("LLM not loaded. Call load_llm() first.")
            return None

        if not self.graph:
            self.log.error("Neo4j not initialised.")
            return None

        if len(documents) > max_docs:
            self.log.warning(f"Truncating {len(documents)} docs to {max_docs}.")
            documents = documents[:max_docs]

        try:
            wrapped_llm = MarkdownStrippingLLM(llm=self.llm)

            transformer = LLMGraphTransformer(
                llm=wrapped_llm,
                strict_mode=False,
            )

            graph_documents = transformer.convert_to_graph_documents(documents)

            self.graph.add_graph_documents(
                graph_documents,
                baseEntityLabel=True,
                include_source=True,
            )

            self.log.info(
                f"Graph ingestion complete — {len(graph_documents)} graph doc(s) stored."
            )
            return graph_documents

        except Exception as e:
            self.log.error(f"ERROR DURING GRAPH TRANSFORMATION PROCESS: {e}")
            return None

    def get_graph_chain(self) -> Optional[GraphCypherQAChain]:
        """Build and return a GraphCypherQAChain."""
        if not self.llm or not self.graph:
            self.log.error("LLM or graph missing.")
            return None

        try:
            chain = GraphCypherQAChain.from_llm(
                llm=self.llm,
                graph=self.graph,
                verbose=True,
                allow_dangerous_requests=True,
            )
            self.log.info("Graph QA chain created.")
            return chain

        except Exception as e:
            self.log.error(f"Chain creation failed: {e}")
            return None

    def query(self, question: str) -> Any:
        """Run question against the graph and return the answer."""
        chain = self.get_graph_chain()
        if not chain:
            return "Unable to initialise graph chain."

        try:
            return chain.invoke({"query": question})
        except Exception as e:
            self.log.error(f"Query failed: {e}")
            return f"Query failed: {e}"