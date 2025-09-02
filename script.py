# app/main.py
from fastapi import FastAPI, HTTPException, Query, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os, pickle
import numpy as np
import pandas as pd
from difflib import get_close_matches

APP_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(APP_DIR, "data")
MOVIES_PKL = os.path.join(DATA_DIR, "movies.pkl")
SIM_PKL = os.path.join(DATA_DIR, "similarity.pkl")   # or similarity.npy

app = FastAPI(title="Movie Recommender API")

# CORS: adjust allowed_origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # change to your frontend origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

movies: Optional[pd.DataFrame] = None
similarity = None
title_to_index = {}

def load_artifacts():
    global movies, similarity, title_to_index
    if not os.path.exists(MOVIES_PKL):
        raise RuntimeError(f"Missing {MOVIES_PKL}")
    movies = pd.read_pickle(MOVIES_PKL)

    # support either pickle or numpy .npy
    if os.path.exists(SIM_PKL):
        with open(SIM_PKL, "rb") as f:
            similarity = pickle.load(f)
    else:
        # try .npy fallback
        npy_path = os.path.join(DATA_DIR, "similarity.npy")
        if os.path.exists(npy_path):
            similarity = np.load(npy_path, mmap_mode='r')  # memory-map if large
        else:
            raise RuntimeError("Missing similarity artifact (similarity.pkl or similarity.npy)")

    # normalize types and build title index
    movies["title"] = movies["title"].astype(str)
    title_to_index = {t.lower(): i for i, t in enumerate(movies["title"].tolist())}

# Load at startup (synchronous)
load_artifacts()

class RecommendIn(BaseModel):
    title: str
    top_n: int = 10

class MovieOut(BaseModel):
    movie_id: int
    title: str
    score: float

@app.get("/health")
def health():
    return {"status": "ok", "num_movies": int(len(movies))}

@app.get("/movies", response_model=List[str])
def search_movies(q: str = Query("", description="search substring")):
    titles = movies["title"].tolist()
    if not q:
        return titles[:50]
    ql = q.lower().strip()
    contains = [t for t in titles if ql in t.lower()]
    if contains:
        return contains[:50]
    # fuzzy fallback
    return get_close_matches(q, titles, n=10, cutoff=0.6)

@app.post("/recommend", response_model=List[MovieOut])
def recommend(inp: RecommendIn):
    title = inp.title.strip()
    idx = title_to_index.get(title.lower())
    if idx is None:
        # fuzzy match fallback
        best = get_close_matches(title, movies["title"].tolist(), n=1, cutoff=0.6)
        if not best:
            raise HTTPException(status_code=404, detail="Title not found")
        idx = title_to_index[best[0].lower()]

    # similarity may be numpy memmap or array
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    out = []
    for i, s in scores[1: inp.top_n + 1]:  # skip self
        row = movies.iloc[i]
        out.append(MovieOut(movie_id=int(row["movie_id"]), title=str(row["title"]), score=float(s)))
    return out

