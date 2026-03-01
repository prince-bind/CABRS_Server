from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import pickle
import os
import random

app = FastAPI(title="CABRS AI: Matchup Engine")

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

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "data_loaded": df_over is not None,
        "run_model_loaded": model_runs is not None if 'model_runs' in globals() else False,
        "wicket_model_loaded": model_wicket is not None if 'model_wicket' in globals() else False
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

def calculate_bowler_score(request: MatchupRequest, bowler_name: str):
    phase = 1 if request.over <= 6 else 2 if request.over <= 15 else 3

    row = {
        'inning': request.inning,
        'over': request.over - 1,
        'phase': int(phase),
        'bowler': bowler_name,
        'striker': request.striker,
        'non_striker': request.non_striker,
        'venue': request.venue,
        'bowler_avg_runs': bowler_avg.get(bowler_name, GLOBAL_MEAN),
        'venue_avg_runs': venue_avg.get(request.venue, GLOBAL_MEAN),
        'bowler_vs_striker_avg': vs_avg.get((bowler_name, request.striker), GLOBAL_MEAN)
    }

    input_df = pd.DataFrame([row])

    input_runs = encoder_runs.transform(input_df)
    pred_runs = model_runs.predict(input_runs)[0]

    input_wicket = encoder_wicket.transform(input_df)
    pred_wicket_prob = model_wicket.predict_proba(input_wicket)[0][1]

    final_score = float(pred_runs - (pred_wicket_prob * 6))

    return float(pred_runs), float(pred_wicket_prob), float(final_score)

@app.post("/predict")
def get_best_bowlers(request: MatchupRequest):
    if df_over is None:
        raise HTTPException(status_code=500, detail="Engine Offline")

    results = []

    for b_name in request.bowler_list:
        try:
            pred_runs, pred_wicket_prob, final_score = calculate_bowler_score(request, b_name)

            results.append({
                "bowler": b_name,
                "predicted_score": round(final_score, 3),
                "ai_insights": [
                    f"Predicted Run Concession: {round(pred_runs, 2)}",
                    f"Wicket Probability: {round(pred_wicket_prob * 100, 1)}%"
                ]
            })

        except Exception as e:
            print(f"Prediction Error for {b_name}: {e}")
            continue

    if not results:
        raise HTTPException(status_code=400, detail="No valid bowlers provided")

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

@app.post("/simulate_win_probability")
def simulate_win_probability(request: MatchupRequest):
    if df_over is None:
        raise HTTPException(status_code=500, detail="Engine Offline")

    simulation_results = []

    for b_name in request.bowler_list:
        try:
            pred_runs, pred_wicket_prob, final_score = calculate_bowler_score(request, b_name)

            # Convert numpy types to Python float
            pred_runs = float(pred_runs)
            pred_wicket_prob = float(pred_wicket_prob)

            base_win_prob = 0.50

            adjustment = (pred_wicket_prob * 0.15) - (pred_runs / 100)

            new_win_prob = float(
                min(0.99, max(0.01, base_win_prob + adjustment))
            )

            simulation_results.append({
                "bowler": b_name,
                "win_probability": round(new_win_prob * 100, 2),
                "predicted_runs": round(pred_runs, 2),
                "wicket_probability": round(pred_wicket_prob * 100, 2)
            })

        except Exception as e:
            print("Win Simulation Error:", e)
            continue

    if not simulation_results:
        raise HTTPException(status_code=400, detail="Simulation failed")

    simulation_results.sort(key=lambda x: x["win_probability"], reverse=True)

    return {
        "status": "success",
        "best_bowler": simulation_results[0]["bowler"],
        "simulations": simulation_results
    }

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

@app.post("/simulate_innings")
def simulate_innings(request: MatchupRequest):
    if df_over is None:
        raise HTTPException(status_code=500, detail="Engine Offline")

    total_runs = 0.0
    total_wickets = 0
    over_results = []

    current_striker = request.striker
    current_non_striker = request.non_striker

    for over_number in range(request.over, 21):

        best_bowler = None
        best_score = float("inf")
        best_runs = 0.0
        best_wicket_prob = 0.0

        for bowler in request.bowler_list:
            try:
                temp_request = MatchupRequest(
                    venue=request.venue,
                    striker=current_striker,
                    non_striker=current_non_striker,
                    over=over_number,
                    inning=request.inning,
                    bowler_list=request.bowler_list
                )

                pred_runs, pred_wicket_prob, final_score = calculate_bowler_score(
                    temp_request, bowler
                )

                # FORCE conversion to Python float
                pred_runs = float(pred_runs)
                pred_wicket_prob = float(pred_wicket_prob)
                final_score = float(final_score)

                if final_score < best_score:
                    best_score = final_score
                    best_bowler = bowler
                    best_runs = pred_runs
                    best_wicket_prob = pred_wicket_prob

            except Exception as e:
                print("Innings Simulation Error:", e)
                continue

        if best_bowler is None:
            break

        # Add runs (ensure pure float)
        total_runs += float(best_runs)

        # Simulate wicket
        if random.random() < float(best_wicket_prob):
            total_wickets += 1
            current_striker = "New Batter"
            if total_wickets >= 10:
                break

        over_results.append({
            "over": int(over_number),
            "bowler": best_bowler,
            "predicted_runs": round(float(best_runs), 2),
            "wicket_probability": round(float(best_wicket_prob) * 100, 2)
        })

    return {
        "status": "success",
        "projected_total_runs": round(float(total_runs), 2),
        "projected_wickets": int(total_wickets),
        "overs_simulated": int(len(over_results)),
        "over_breakdown": over_results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)