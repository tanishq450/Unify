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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_markdown_fences(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _normalize_graph_schema(text: str) -> str:
    try:
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

        # ── 2. fix existing nodes ─────────────────────────────────────
        for node in data.get("nodes", []):
            if "type" not in node and "label" in node:
                node["type"] = node["label"]
            if "id" not in node and "name" in node:
                node["id"] = node["name"]

        # ── 3. edges -> relationships ─────────────────────────────────
        if "edges" in data and "relationships" not in data:
            data["relationships"] = data.pop("edges")

        # ── 4. remap relationship fields to DynamicGraph schema ───────
        #
        #  Claude returns:
        #    {"source": "X", "target": "Y", "type": "REL"}
        #
        #  DynamicGraph expects:
        #    {"source_node_id": "X", "source_node_type": "?",
        #     "target_node_id": "Y", "target_node_type": "?",
        #     "type": "REL"}
        #
        # Build a node-type lookup from the nodes list
        node_type_map = {
            n["id"]: n.get("type", "Entity")
            for n in data.get("nodes", [])
        }

        fixed_rels = []
        for rel in data.get("relationships", []):
            src = rel.get("source", rel.get("source_node_id", ""))
            tgt = rel.get("target", rel.get("target_node_id", ""))
            rel_type = rel.get("type", rel.get("label", "RELATED_TO"))

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


def _extract_content_from_claude(result: Any) -> str:
    """
    Claude returns tool_use blocks inside AIMessage.content as a list.
    This extracts the actual JSON payload regardless of format.
    """
    # Case 1:
    if isinstance(result, AIMessage) and isinstance(result.content, str):
        return result.content

    # Case 2: Claude tool_use block — content is a list of dicts
    if isinstance(result, AIMessage) and isinstance(result.content, list):
        for block in result.content:
            # Tool use block has 'input' dict with the structured data
            if isinstance(block, dict):
                if block.get("type") == "tool_use" and "input" in block:
                    return json.dumps(block["input"])
                # Sometimes it's a text block
                if block.get("type") == "text" and "text" in block:
                    return block["text"]

    # Case 3: already a dict (structured output parsed by langchain)
    if isinstance(result, dict):
        return json.dumps(result)

    # Case 4: Pydantic model already parsed
    if hasattr(result, "model_dump"):
        return json.dumps(result.model_dump())

    return str(result)


def _clean_claude_output(result: Any) -> AIMessage:
    """Extract, clean and normalize Claude's response into a plain AIMessage."""
    content = _extract_content_from_claude(result)
    content = _strip_markdown_fences(content)
    content = _normalize_graph_schema(content)
    additional_kwargs = result.additional_kwargs if hasattr(result, "additional_kwargs") else {}
    return AIMessage(content=content, additional_kwargs=additional_kwargs)


# ---------------------------------------------------------------------------
# LLM Wrapper 
# ---------------------------------------------------------------------------

class MarkdownStrippingLLM(RunnableSerializable):
    """
    Claude-specific LLM wrapper for LLMGraphTransformer.

    Claude uses tool_use blocks for structured output, NOT plain JSON text.
    This wrapper intercepts at the right level so cleaning happens
    BEFORE Pydantic validation.
    """

    llm: Any

    def invoke(self, input, config=None, **kwargs):
        result = self.llm.invoke(input, config=config, **kwargs)
        return _clean_claude_output(result)

    async def ainvoke(self, input, config=None, **kwargs):
        result = await self.llm.ainvoke(input, config=config, **kwargs)
        return _clean_claude_output(result)

    def with_structured_output(self, schema, **kwargs):
        """
        CRITICAL for Claude:
        Do NOT call self.llm.with_structured_output() — Claude will use
        tool calling and bypass our cleaner entirely.

        Instead: call raw LLM → extract tool_use input → parse into schema.
        """

        def extract_and_parse(result):
            # Step 1: extract raw content (handles tool_use blocks)
            content = _extract_content_from_claude(result)

            # Step 2: strip any markdown fences
            content = _strip_markdown_fences(content)

            # Step 3: normalize schema
            content = _normalize_graph_schema(content)

            # Step 4: parse into Pydantic schema
            try:
                # Pydantic v2
                return schema.model_validate_json(content)
            except AttributeError:
                pass
            except Exception as e:
                raise ValueError(
                    f"Failed to parse into {schema.__name__}: {e}\n"
                    f"Content (first 500 chars): {content[:500]}"
                )

            try:
                # Pydantic v1
                return schema.parse_raw(content)
            except Exception as e:
                raise ValueError(
                    f"Pydantic v1 parse failed for {schema.__name__}: {e}\n"
                    f"Content (first 500 chars): {content[:500]}"
                )

        # Chain: this wrapper's invoke (which cleans) → parse into schema
        return self | RunnableLambda(extract_and_parse)

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

    def __init__(self, llm: Optional[Any] = None):
        self.log = loguru.logger
        self.llm = llm
        self.graph: Optional[Neo4jGraph] = None
        self._init_graph()

    def _init_graph(self):
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
        self.llm = llm_instance
        self.log.info("LLM loaded successfully.")

    def convert_docs(
        self,
        text: str,
        chunk_size: int = 4000,
        chunk_overlap: int = 400,
    ) -> List[Document]:
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
            # ── Claude-specific wrapper ──
            wrapped_llm = _MarkdownStrippingLLM(llm=self.llm)

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
        chain = self.get_graph_chain()
        if not chain:
            return "Unable to initialise graph chain."

        try:
            return chain.invoke({"query": question})
        except Exception as e:
            self.log.error(f"Query failed: {e}")
            return f"Query failed: {e}"