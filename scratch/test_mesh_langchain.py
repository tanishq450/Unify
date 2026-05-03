import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("MESH_API_KEY")
base_url = os.getenv("MESH_API_BASE", "https://api.meshapi.ai/v1")

print(f"Testing with MESH_API_KEY: {api_key[:5]}...")
print(f"Testing with MESH_API_BASE: {base_url}")

# This is what's currently in main.py
llm_old = ChatOpenAI(
    model="anthropic/claude-opus-4.1",
    temperature=0.1,
    openai_api_base=base_url,
    openai_api_key=api_key,
)

try:
    print("Testing old config (openai_api_base/openai_api_key)...")
    res = llm_old.invoke("Hi")
    print(f"Success! Response: {res.content[:50]}")
except Exception as e:
    print(f"Failed with old config: {e}")

# This is the recommended way for langchain-openai
llm_new = ChatOpenAI(
    model="anthropic/claude-opus-4.1",
    temperature=0.1,
    base_url=base_url,
    api_key=api_key,
)

try:
    print("\nTesting new config (base_url/api_key)...")
    res = llm_new.invoke("Hi")
    print(f"Success! Response: {res.content[:50]}")
except Exception as e:
    print(f"Failed with new config: {e}")
