import os

import json, cv2
from norfair import Detection
import numpy as np

def load_results_with_files(json_path, frames_dir="cache"):
    raw = json.load(open(json_path))
    out = []
    for entry in raw:
        frame = cv2.imread(os.path.join(frames_dir, entry["frame_file"]))
        balls = [Detection(points=np.array([[d["x1"],d["y1"]],
                                           [d["x2"],d["y2"]]]),
                           label=d["cls"],
                           data={"id":d["cls"]})
                 for d in entry["ball_detections"]]
        players = [Detection(points=np.array([[d["x1"],d["y1"]],
                                             [d["x2"],d["y2"]]]),
                             label=d["cls"],
                             data={"id":d["cls"]})
                   for d in entry["player_detections"]]
        out.append((frame, balls, players))
    return out

