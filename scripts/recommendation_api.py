import sys
sys.path.append('src')

import torch
import pandas as pd
import requests
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from scripts.generate_recommendation import OnlineRecommenderService


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Emotion-Aware Music Recommendation API", version="1.0.0")

class RecommendationRequest(BaseModel):
    text: str
    user_id: str
    top_k: int = 5

class SongRecommendation(BaseModel):
    song: str
    artist: str
    valence: float
    arousal: float
    score: float

class RecommendationResponse(BaseModel):
    query_text: str
    user_id: str
    inferred_va: dict
    recommendations: List[SongRecommendation]

try:
    user_top_artists = {
        "4fea8ee3745dada8a8fa2a2d26514bc1232c3a15": "the beatles"
    }

    recommender_service = OnlineRecommenderService(
        api_url="http://localhost:8000",
        song_parquet="data/processed/merged_data.parquet",
        user_emotion_path="data/processed/user_emotion_table.pkl",
        emotion_artist_path="data/processed/emotion_artist_table.pkl",
        user_top_artists=user_top_artists
    )
    logger.info("Recommender service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize recommender service: {e}")
    recommender_service = None

@app.get("/")
def root():
    return {
        "message": "Emotion-Aware Music Recommendation API",
        "endpoints": {
            "recommend": "POST /recommend - Generate song recommendations",
            "health": "GET /health - Check API health"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "recommender": "ready" if recommender_service else "not initialized",
        "songs_loaded": len(recommender_service.songs) if recommender_service else 0
    }

@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest):
    if not recommender_service:
        raise HTTPException(status_code=500, detail="Recommender service not initialized")

    if len(recommender_service.songs) == 0:
        raise HTTPException(status_code=400, detail="No songs in database. Please run build_spotify_va.py first.")

    try:
        query_va = recommender_service.infer_va(request.text)
        recs = recommender_service.recommend(query_va, request.user_id, request.top_k)

        recommendations = [
            SongRecommendation(
                song=r["song"],
                artist=r["artist"],
                valence=r["valence"],
                arousal=r["arousal"],
                score=r["score"]
            )
            for r in recs
        ]

        return RecommendationResponse(
            query_text=request.text,
            user_id=request.user_id,
            inferred_va={"valence": query_va[0].item(), "arousal": query_va[1].item()},
            recommendations=recommendations
        )
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Recommendation failed")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Recommendation API server...")
    uvicorn.run(app, host="0.0.0.0", port=8001)