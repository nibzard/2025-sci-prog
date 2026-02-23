# api/main.py
"""
FastAPI server for Laptop Battle comparison API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

from .compare import compare_laptops

app = FastAPI(
    title="Laptop Battle API",
    description="Compare laptops using Reddit sentiment analysis",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class CompareRequest(BaseModel):
    laptop1: str
    laptop2: str


class LaptopAnalysis(BaseModel):
    laptop_name: str
    sentiment_score: int
    pros: List[str]
    cons: List[str]
    key_themes: List[str]
    sentiment_explanation: str
    user_recommendation: str
    posts_analyzed: int
    error: Optional[str] = None


class CompareResponse(BaseModel):
    laptop1: LaptopAnalysis
    laptop2: LaptopAnalysis
    winner: str  # "laptop1", "laptop2", or "tie"
    score_difference: int


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Laptop Battle API is running"}


@app.post("/api/compare", response_model=CompareResponse)
async def compare_laptops_endpoint(request: CompareRequest):
    """
    Compare two laptops using Reddit sentiment analysis.
    
    This endpoint:
    1. Scrapes Reddit for posts about each laptop
    2. Stores content in ChromaDB
    3. Analyzes sentiment using Gemini LLM
    4. Returns comparison results with winner
    
    Note: First-time analysis may take 1-2 minutes per laptop.
    Subsequent requests use cached data and are faster.
    """
    if not request.laptop1 or not request.laptop2:
        raise HTTPException(status_code=400, detail="Both laptop names are required")
    
    if request.laptop1.lower() == request.laptop2.lower():
        raise HTTPException(status_code=400, detail="Please enter two different laptops")
    
    try:
        result = await compare_laptops(request.laptop1, request.laptop2)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/api/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "api_version": "1.0.0",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "Root health check"},
            {"path": "/api/compare", "method": "POST", "description": "Compare two laptops"},
            {"path": "/api/health", "method": "GET", "description": "Detailed health check"},
        ]
    }


if __name__ == "__main__":
    print("Starting Laptop Battle API server...")
    print("API docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
