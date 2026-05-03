import asyncio
import os
import loguru
from dotenv import load_dotenv
from implementations.Graph_rag import GRAPH_RAG
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document

load_dotenv()

async def test_graph_rag():
    logger = loguru.logger
    logger.info("Starting Graph RAG test...")

    # 1. Initialize Graph RAG
    graph_rag = GRAPH_RAG()
    
    # 2. Setup LLM (using the same config as in main.py)
    api_key = os.getenv("MESH_API_KEY")
    if not api_key:
        logger.error("MESH_API_KEY not found. Cannot run test.")
        return

    llm = ChatAnthropic(
        model="anthropic/claude-3-5-sonnet-latest",
        temperature=0.1,
        max_tokens=4096,
        base_url=os.getenv("MESH_API_BASE", "https://api.meshapi.ai/v1"),
        api_key=api_key,
    )
    
    graph_rag.load_llm(llm)

    # 3. Test Ingestion
    test_text = """
    Apple Inc. is an American multinational technology company headquartered in Cupertino, California. 
    It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. 
    Tim Cook is the current CEO of Apple.
    """
    docs = [Document(page_content=test_text)]
    
    logger.info("Attempting graph ingestion...")
    result = graph_rag.make_graph(docs)
    
    if result:
        logger.info("Graph ingestion successful!")
    else:
        logger.error("Graph ingestion failed.")

    # 4. Test Query
    question = "Who founded Apple?"
    logger.info(f"Querying: {question}")
    response = graph_rag.query(question)
    
    logger.info(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(test_graph_rag())
