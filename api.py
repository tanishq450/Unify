import os
import shutil
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import uvicorn
import loguru

# Import the orchestrator from main
from main import FinanceRAGOrchestrator

app = FastAPI(
    title="Unify Finance RAG API",
    description="API for the Unify multi-strategy RAG system.",
    version="1.0.0"
)

logger = loguru.logger

# ── Models ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    collection_name: str = "default"

class EvaluateRequest(BaseModel):
    component: str = "all"  # "intent", "verifier", "e2e", "all"

# ── Cache for Orchestrators ─────────────────────────────────────────────────
# Instantiating orchestrators (which loads models and connects to DBs) can be
# expensive, so we cache them by collection_name.
orchestrators = {}

def get_orchestrator(collection_name: str) -> FinanceRAGOrchestrator:
    if collection_name not in orchestrators:
        logger.info(f"Initializing orchestrator for collection: {collection_name}")
        orchestrators[collection_name] = FinanceRAGOrchestrator(collection_name=collection_name)
    return orchestrators[collection_name]

# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Unify Finance RAG API is running."
    }

@app.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    collection_name: str = Form("default")
):
    """
    Upload a financial PDF document to be ingested into the specified collection.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        # Save uploaded file to a temporary file
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        orchestrator = get_orchestrator(collection_name)
        
        logger.info(f"Ingesting file '{file.filename}' into '{collection_name}'")
        await orchestrator.ingest(temp_path)
        
        # Cleanup
        os.remove(temp_path)

        return {"status": "success", "message": f"Successfully ingested {file.filename} into collection '{collection_name}'"}

    except Exception as e:
        logger.exception("Failed during ingestion")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/query")
async def query_system(req: QueryRequest):
    """
    Query the RAG system. Routes the query, retrieves context, and verifies claims.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        orchestrator = get_orchestrator(req.collection_name)
        
        logger.info(f"Processing query: '{req.query}' on collection '{req.collection_name}'")
        result = await orchestrator.query(req.query)
        
        return {
            "status": "success",
            "data": result
        }

    except Exception as e:
        logger.exception("Failed during query processing")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/evaluate")
async def evaluate_system(req: EvaluateRequest):
    """
    Run the evaluation suite.
    Valid components: intent, verifier, e2e, all
    """
    import evaluation
    
    if req.component not in ["intent", "verifier", "e2e", "all"]:
        raise HTTPException(status_code=400, detail="Invalid component. Must be one of: intent, verifier, e2e, all")
        
    try:
        logger.info(f"Running evaluation for component: {req.component}")
        all_results = {}
        
        if req.component in ("intent", "all"):
            all_results["intent_classifier"] = evaluation.evaluate_intent_classifier(verbose=False)
            
        if req.component in ("verifier", "all"):
            all_results["hallucination_verifier"] = evaluation.evaluate_verifier(verbose=False)
            
        if req.component in ("e2e", "all"):
            all_results["end_to_end"] = evaluation.evaluate_e2e(verbose=False)
            
        return {
            "status": "success",
            "component": req.component,
            "data": all_results
        }
    except Exception as e:
        logger.exception("Failed during evaluation")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")



if __name__ == "__main__":
    logger.info("Starting Unify FastAPI Server...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
