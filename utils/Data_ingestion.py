from llama_index.readers.file import PDFReader
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
import fitz
import os
import loguru
from chonkie import TokenChunker
from llama_index.core import Document
from pathlib import Path
from llama_index.core import load_index_from_storage
import loguru
from implementations.Rag import Rag_pipeline
from implementations.Graph_rag import GRAPH_RAG
from Model_loader.llm import ModelLoader




class Docloader:
    def __init__(self,output_dir="./data"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir,exist_ok=True)
        self.logger = loguru.logger
        
    def load_pdf(self,file_path:str):
        try:
            self.logger.info(f"Loading PDF from {file_path}")
            with fitz.open(file_path) as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
            self.logger.info(f"PDF loaded successfully")
            return text
        except Exception as e:
            self.logger.error(f"Error loading PDF: {e}")
            return None 

    def save_text(self,text:str,file_name:str):
        try:
            self.logger.info(f"Saving text to {file_name}")
            with open(os.path.join(self.output_dir,file_name),"w") as f:
                f.write(text)
            self.logger.info(f"Text saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving text: {e}")

    
    def is_encrypted(self,file_path:str):
        try:
            self.logger.info(f"Checking if PDF is encrypted: {file_path}")
            doc = fitz.open(file_path)
            if doc.is_encrypted:
                self.logger.info(f"PDF is encrypted")
                return True
            else:
                self.logger.info(f"PDF is not encrypted")
                return False
        except Exception as e:
            self.logger.error(f"Error checking if PDF is encrypted: {e}")
            return None 
        
        

  

    

class chunking:
    def __init__(self, chunk_size: int = 1000, output_dir: str = "./data",stride: int = 200):
        self.chunk_size = chunk_size
        self.output_dir = output_dir
        self.logger = loguru.logger
        self.stride = stride
        
    def chunk_text(self,text:str):
        try:
            self.logger.info(f"Chunking text")
            chunker = TokenChunker(chunk_size=self.chunk_size,chunk_overlap=self.stride)
            chunks = chunker(text)
            self.logger.info(f"Text chunked successfully")
            return chunks
        except Exception as e:
            self.logger.error(f"Error chunking text: {e}")
            return None 


    def save_chunks(self,chunks:list,file_name:str):
        try:
            self.logger.info(f"Saving chunks to {file_name}")
            with open(os.path.join(self.output_dir,file_name),"w") as f:
                for chunk in chunks:
                    f.write(chunk.text + "\n")
            self.logger.info(f"Chunks saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving chunks: {e}")
    

    def convert_chunks(self, chunks: list):
        if not chunks:
            raise ValueError("No chunks provided")

        self.logger.info("Converting chunks to Documents")

        documents = []
        for chunk in chunks:
            if isinstance(chunk, str):
                documents.append(Document(text=chunk))
            elif hasattr(chunk, "text"):
                documents.append(Document(text=chunk.text))
            else:
                raise TypeError(
                    f"Unsupported chunk type: {type(chunk)}"
                )

        self.logger.info("Chunks converted to Documents successfully")
        return documents


async def unified_ingest(file_path: str, collection_name: str):
    """
    Unified Ingestion Orchestrator:
    Pushes data to both Qdrant (Hybrid RAG) and Neo4j (Graph RAG).
    """
    
    logger = loguru.logger
    logger.info(f"Starting unified ingestion for {file_path} into collection '{collection_name}'")

    # ---------------- 1. INGEST TO QDRANT (HYBRID RAG) ----------------
    try:
        logger.info("Starting Qdrant ingestion...")
        rag = Rag_pipeline()
        await rag.ingest(file_path=file_path, persist_dir=collection_name)
        logger.info("✅ Qdrant ingestion complete!")
    except Exception as e:
        logger.error(f"❌ Qdrant ingestion failed: {e}")

    # ---------------- 2. INGEST TO NEO4J (GRAPH RAG) ----------------
    try:
        logger.info("Starting Neo4j ingestion...")
        loader = Docloader()
        text = loader.load_pdf(file_path)
        
        if text:
            graph_rag = GRAPH_RAG()
            
            # Use LangChain-compatible LLM (required by LLMGraphTransformer)
            langchain_llm = ModelLoader.get_langchain_llm()
            graph_rag.load_llm(langchain_llm)
            
            # Convert and make graph
            docs = graph_rag.convert_docs(text)
            graph_rag.make_graph(docs)
            logger.info("✅ Neo4j graph ingestion complete!")
        else:
            logger.error("No text could be extracted for graph ingestion.")
    except Exception as e:
        logger.error(f"❌ Neo4j ingestion failed: {e}")


if __name__ == "__main__":
    import asyncio
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python -m utils.Data_ingestion <path_to_pdf> <collection_name>")
        sys.exit(1)
        
    pdf_path = sys.argv[1]
    coll_name = sys.argv[2]
    
    asyncio.run(unified_ingest(pdf_path, coll_name))