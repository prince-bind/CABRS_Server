from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import pickle
import os
import httpx
import random
from datetime import datetime, timedelta

app = FastAPI(title="CABRS AI: Matchup Engine v2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Load Data and Models
# -------------------------
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

try:
    df_over = pd.read_csv(os.path.join(BASE_PATH, "over_final.csv"), dtype={'season': str})
    df_over['venue'] = df_over['venue'].apply(lambda x: x.split(',')[0].strip())
    
    GLOBAL_MEAN = df_over['total_runs'].mean()
    bowler_avg = df_over.groupby('bowler')['total_runs'].mean().to_dict()
    venue_avg = df_over.groupby('venue')['total_runs'].mean().to_dict()
    vs_avg = df_over.groupby(['bowler','striker'])['total_runs'].mean().to_dict()

    with open(os.path.join(BASE_PATH, "model_runs.pkl"), "rb") as f:
        model_runs = pickle.load(f)
    with open(os.path.join(BASE_PATH, "model_wicket.pkl"), "rb") as f:
        model_wicket = pickle.load(f)
    with open(os.path.join(BASE_PATH, "encoder_runs.pkl"), "rb") as f:
        encoder_runs = pickle.load(f)
    with open(os.path.join(BASE_PATH, "encoder_wicket.pkl"), "rb") as f:
        encoder_wicket = pickle.load(f)
        
    print("✅ All Systems Operational: Models and CSV loaded.")
except Exception as e:
    print(f"❌ Critical System Failure: {e}")
    df_over = None

# Realistic Stats DB for Radar Chart
BOWLER_PROFILE_DB = {
    "JJ Bumrah": {"econ": 7.30, "sr": 18.5, "dot_pct": 45.2, "pressure_index": 9.8},
    "Mohammed Shami": {"econ": 8.44, "sr": 17.0, "dot_pct": 46.1, "pressure_index": 8.5},
    "Mohammed Siraj": {"econ": 8.55, "sr": 19.5, "dot_pct": 44.0, "pressure_index": 8.8},
    "YS Chahal": {"econ": 7.67, "sr": 16.5, "dot_pct": 41.5, "pressure_index": 8.2},
    "Rashid Khan": {"econ": 6.67, "sr": 17.0, "dot_pct": 47.8, "pressure_index": 9.5},
}

class MatchupRequest(BaseModel):
    venue: str
    striker: str
    non_striker: str
    over: int
    inning: int
    bowler_list: List[str]

@app.get("/get_metadata")
def get_metadata():
    if df_over is None: raise HTTPException(status_code=500, detail="Data Offline")
    return {
        "venues": sorted(df_over['venue'].unique().tolist()),
        "batters": sorted(df_over['striker'].unique().tolist()),
        "bowlers": sorted(df_over['bowler'].unique().tolist())
    }

@app.post("/predict")
def get_best_bowlers(request: MatchupRequest):
    if df_over is None: raise HTTPException(status_code=500, detail="Engine Offline")
    
    phase = 1 if request.over <= 6 else 2 if request.over <= 15 else 3
    results = []
    
    for b_name in request.bowler_list:
        try:
            row = {
                'inning': request.inning,
                'over': request.over - 1,
                'phase': int(phase),
                'bowler': b_name,
                'striker': request.striker,
                'non_striker': request.non_striker,
                'venue': request.venue,
                'bowler_avg_runs': bowler_avg.get(b_name, GLOBAL_MEAN),
                'venue_avg_runs': venue_avg.get(request.venue, GLOBAL_MEAN),
                'bowler_vs_striker_avg': vs_avg.get((b_name, request.striker), GLOBAL_MEAN)
            }
            input_df = pd.DataFrame([row])

            input_runs = encoder_runs.transform(input_df)
            pred_runs = model_runs.predict(input_runs)[0]
            
            input_wicket = encoder_wicket.transform(input_df)
            pred_wicket_prob = model_wicket.predict_proba(input_wicket)[0][1]
            
            final_score = float(pred_runs - (pred_wicket_prob * 6))
            profile = BOWLER_PROFILE_DB.get(b_name, {"econ": 8.0, "sr": 18.0, "dot_pct": 40.0, "pressure_index": 7.5})
            
            results.append({
                "bowler": b_name,
                "predicted_score": round(final_score, 3),
                "metrics": {
                    "econ": int(max(40, 100 - ((profile.get("econ", 8.0) - 6.5) * 20))),
                    "sr": int(max(40, 100 - ((profile.get("sr", 18.0) - 15.0) * 8.5))),
                    "dot": int(max(40, ((profile.get("dot_pct", 40) - 35) * 4))),
                    "pressure": int(profile.get("pressure_index", 7.5) * 10)
                },
                "ai_insights": [
                    f"Predicted Run Concession: {round(pred_runs, 2)}",
                    f"Wicket Probability: {round(pred_wicket_prob * 100, 1)}%"
                ]
            })
        except Exception as e:
            print(f"Prediction Error for {b_name}: {e}")
            continue

    results.sort(key=lambda x: x["predicted_score"])
    confidence = 85
    if len(results) > 1:
        gap = abs(results[1]["predicted_score"] - results[0]["predicted_score"])
        confidence = min(99, int(80 + (gap * 10)))

    return {
        "status": "success",
        "top_recommendation": results[0]["bowler"],
        "confidence": confidence,
        "predictions": results
    }

# --- RANDOMIZED TACTICAL BIO ENGINE (REPLACED GEMINI) ---

@app.get("/get_player_bio/{player_name}")
async def get_player_bio(player_name: str):
    insights = [
        f"{player_name} is an elite situational specialist known for high-release accuracy and cross-seam variations.",
        f"Tactical analysis suggests {player_name} excels at identifying batter weak zones and exploiting uneven bounce.",
        f"{player_name} provides consistent control in high-pressure phases with disciplined line and length adjustments.",
        f"Analysis indicates {player_name} uses deceptive pace changes to disrupt batter timing during transitional overs.",
        f"A strategic asset, {player_name} utilizes boundary dimensions effectively to force lower-percentage shots.",
        f"Data confirms {player_name} is a high-impact choice for creating dot-ball pressure in the current match phase."
    ]
    return {"bio": random.choice(insights)}


from datetime import datetime, timedelta
import httpx

CRICKET_API_KEY = "3343abea-2de7-4f13-865c-8aa41398e2da"

SCHEDULE_CACHE = {"data": None, "timestamp": None}
CACHE_DURATION = timedelta(minutes=30)

SERIES_ID = "f182390a-1144-42f2-95f2-95f2a890089e"


@app.get("/get_world_cup_schedule")
async def get_schedule():

    # ✅ Serve cached data
    if (
        SCHEDULE_CACHE["data"] is not None
        and datetime.now() - SCHEDULE_CACHE["timestamp"] < CACHE_DURATION
    ):
        print("⚡ Returning cached WC schedule")
        return SCHEDULE_CACHE["data"]

    async with httpx.AsyncClient(timeout=20) as client:
        try:
            # ---------- TRY SERIES FIXTURES ----------
            url = f"https://api.cricapi.com/v1/series_fixtures?apikey={CRICKET_API_KEY}&id={SERIES_ID}"
            response = await client.get(url)
            res_json = response.json()

            # ❌ If invalid API → fallback
            if res_json.get("status") == "failure":
                print("⚠ series_fixtures blocked, using fallback")

                fallback_url = f"https://api.cricapi.com/v1/currentMatches?apikey={CRICKET_API_KEY}&offset=0"
                fallback_res = await client.get(fallback_url)
                fallback_json = fallback_res.json()

                all_matches = fallback_json.get("data", [])

                # STRICT WORLD CUP FILTER
                matches = []
                for m in all_matches:
                    name = m.get("name", "").lower()

                    if (
                        ("world cup" in name or "t20wc" in name or "icc t20")
                        and "women" not in name
                        and "u19" not in name
                    ):
                        matches.append(m)

            else:
                data_obj = res_json.get("data", {})
                matches = (
                    data_obj.get("matchList", [])
                    if isinstance(data_obj, dict)
                    else data_obj or []
                )

            # ---------- NORMALIZE TEAM DATA ----------
            for match in matches:
                team_info = match.get("teamInfo", [])
                if len(team_info) >= 2:
                    match["teams"] = [
                        team_info[0].get("shortname", "TBC"),
                        team_info[1].get("shortname", "TBC"),
                    ]
                else:
                    match["teams"] = match.get("teams", ["TBC", "TBC"])

            # ---------- CACHE ----------
            SCHEDULE_CACHE["data"] = matches
            SCHEDULE_CACHE["timestamp"] = datetime.now()

            print(f"✅ Final WC Matches Count: {len(matches)}")

            return matches

        except Exception as e:
            print("❌ Schedule error:", e)
            return SCHEDULE_CACHE["data"] or []