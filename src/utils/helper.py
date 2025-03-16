import os
import json
import shutil

def choose_and_save_best_model(results):
    """
    Given a list of (model_type, path, metrics_dict), pick the best model 
    based on R2 (or any other metric).
    Then copy or rename the saved file to something like 'models/best_model.*'.
    Return (best_model_type, best_model_path).
    """
    if not results:
        return None, None

    results_sorted = sorted(results, key=lambda x: x[2]["R2"], reverse=True)
    best_model_type, best_model_path, best_metrics = results_sorted[0]

    with open("models/best_model_info.json", "w") as f:
        json.dump({"best_model_type": best_model_type}, f)

    ext = ".pkl" if best_model_type == "xgboost" else ".pt"
    best_final_path = os.path.join("models", f"best_model{ext}")

    shutil.copy(best_model_path, best_final_path)

    return best_model_type, best_final_path