import os
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np

# =========================
# Paths & Data
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

MOVIES_PKL = os.path.join(DATA_DIR, "movies.pkl")
SIM_PKL = os.path.join(DATA_DIR, "similarity.pkl")

movies = None
similarity = None


def load_artifacts():
    global movies, similarity
    if not os.path.exists(MOVIES_PKL):
        raise RuntimeError(f"❌ Missing {MOVIES_PKL}")
    if not os.path.exists(SIM_PKL):
        raise RuntimeError(f"❌ Missing {SIM_PKL}")

    with open(MOVIES_PKL, "rb") as f:
        movies = pickle.load(f)
    with open(SIM_PKL, "rb") as f:
        similarity = pickle.load(f)


# =========================
# FastAPI App
# =========================
app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    title: str
    top_n: int = 5


@app.get("/health")
def health():
    return {"status": "ok", "movies_loaded": movies is not None}


@app.post("/recommend")
def recommend(request: RecommendRequest):
    if movies is None or similarity is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    if request.title not in movies["title"].values:
        raise HTTPException(status_code=404, detail=f"Movie '{request.title}' not found")

    idx = movies[movies["title"] == request.title].index[0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1: request.top_n + 1]

    recommendations = []
    for i, score in sim_scores:
        recommendations.append({"title": movies.iloc[i].title, "score": float(score)})
    return recommendations


# =========================
# Serve Frontend
# =========================
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
os.makedirs(FRONTEND_DIR, exist_ok=True)

# Serve static files (index.html inside /frontend)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


@app.on_event("startup")
def startup_event():
    load_artifacts()
