import os
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("MESH_API_KEY")
base = os.getenv("MESH_API_BASE")

print(f"MESH_API_KEY: '{key}'")
print(f"MESH_API_BASE: '{base}'")
