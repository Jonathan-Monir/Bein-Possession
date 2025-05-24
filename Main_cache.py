import os
import json
import time
from typing import List
from PIL import Image
import norfair
from tracker import process_video
from cluster_time_improve import main_multi_frame
from transformation import process_field_transformation
from possessionCalculation import CalculatePossession
import warnings
from videoDraw import draw_bounding_boxes_on_frames, save_video_from_frames
warnings.filterwarnings("ignore")

# Define the cache directory and create it if it doesn't exist
CACHE_DIR = "cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def load_cache(filename: str):
    """Attempt to load cached JSON data from the cache folder."""
    cache_path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(cache_path):
        print(f"Loading cache from {cache_path}")
        with open(cache_path, "r") as f:
            return json.load(f)
    return None

def save_cache(data, filename: str):
    """Save data as a JSON file in the cache folder."""
    cache_path = os.path.join(CACHE_DIR, filename)
    try:
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Saved cache to {cache_path}")
    except Exception as e:
        print(f"Error saving cache to {cache_path}: {e}")

def measure_time(func, cache_filename: str, *args, process_name="Process", use_cache=True):
    """
    Wrapper to measure execution time.
    Checks for a cached result before running the function.
    """
    if use_cache:
        cached_result = load_cache(cache_filename)
        if cached_result is not None:
            return cached_result

    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{process_name} completed in {elapsed_time:.4f} seconds")

    if use_cache:
        try:
            save_cache(result, cache_filename)
        except Exception as e:
            print(f"Error saving {process_name} output to cache: {e}")
    return result

# TRACKING
tracking_cache_file = "tracking.json"
results = measure_time(
    process_video, 
    tracking_cache_file, 
    r"resources\yolo8.pt", 
    r"resources\manc3.mp4", 
    20, 
    process_name="Tracking"
)

# CLUSTERING
clustering_cache_file = "clustering.json"
results_with_class_ids, team1_color, team2_color = measure_time(
    main_multi_frame, 
    clustering_cache_file, 
    results, 
    process_name="Clustering"
)

# Calibration configuration
calibrator_cfgs = {
    "cfg_path": r"Transformation/config/hrnetv2_w48.yaml",
    "cfg_line_path": r"Transformation/config/hrnetv2_w48_l.yaml",
    "kp_model_path": r"resources/SV_FT_TSWC_kp",
    "line_model_path": r"resources/SV_FT_TSWC_lines",
    "kp_threshold": 0.1486,
    "line_threshold": 0.3880
}

# FIELD TRANSFORMATION
field_transformation_cache_file = "field_transformation.json"
results = measure_time(
    process_field_transformation, 
    field_transformation_cache_file, 
    results_with_class_ids, 
    calibrator_cfgs, 
    process_name="Field Transformation"
)

# POSSESSION CALCULATION
# Define yard coordinates as given
yardTL, yardTR, yardBL, yardBR = [29.0, 17.0], [45.5, 17.0], [29.0, 26.0], [45.5, 26.0]
possession_cache_file = "possession_calculation.json"
poss, team_poss_list = measure_time(
    CalculatePossession, 
    possession_cache_file, 
    results, 
    yardTL, 
    yardTR, 
    yardBL, 
    yardBR, 
    process_name="Possession Calculation"
)

print("Possession Results:", poss)

# Visualize and export detection results (not cached as it's lighter-weight)
visualize = draw_bounding_boxes_on_frames(results_with_class_ids, team1_color, team2_color, team_poss_list)
save_video_from_frames(visualize, output_path="detection_results.mp4")
