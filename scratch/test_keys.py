import os
from dotenv import load_dotenv
from implementations.multimodal_table_extractor import UnifiedTableExtractor

load_dotenv()
key = os.getenv("LLAMA_API_KEY")
print(f"LLAMA_API_KEY in env: {key}")

extractor = UnifiedTableExtractor()
print(f"Extractor llama_parse_key: {extractor.llama_parse_key}")
