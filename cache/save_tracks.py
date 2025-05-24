
import json
import os
import cv2

def det_to_dict(det):
    # Extract points → [[x1,y1],[x2,y2]]
    pts = det.points.tolist()
    x1, y1 = map(int, pts[0])
    x2, y2 = map(int, pts[1])

    # Try these in order: det.label, det.class_id, det.data["id"]
    cls = None
    if hasattr(det, "label") or det.label is not None:
        cls = det.label
    elif hasattr(det, "class_id") or det.class_id is not None:
        cls = det.class_id
    elif isinstance(det.data, dict) or "id" in det.data:
        cls = det.data["id"]
    else:
        cls = -1  # or 0, whatever your “unknown” sentinel is

    return {
        "cls": cls,
        "x1": x1, "y1": y1,
        "x2": x2, "y2": y2
    }

def serialize_results_with_files(
    results_tracking,
    frames_dir="cache/tracked_frames",
    json_path="cache/tracking_results.json"
):
    # 1) Make sure your frames folder and JSON folder exist
    os.makedirs(frames_dir, exist_ok=True)
    parent_json_dir = os.path.dirname(json_path) or "."
    os.makedirs(parent_json_dir, exist_ok=True)

    serializable = []
    for frame_idx, (frame, ball_dets, player_dets) in enumerate(results_tracking):
        # save frame to disk
        filename = f"frame_{frame_idx:05d}.png"
        full_frame_path = os.path.join(frames_dir, filename)
        cv2.imwrite(full_frame_path, frame)

        # pack detections
        entry = {
            "frame_idx": frame_idx,
            "frame_file": os.path.relpath(full_frame_path, parent_json_dir),
            "ball_detections":   [det_to_dict(d) for d in ball_dets],
            "player_detections": [det_to_dict(d) for d in player_dets],
        }
        serializable.append(entry)

    # 2) Dump JSON (will create the file, since the folder exists)
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)


# data_to_save = serialize_results(results_tracking)
# with open("cache/tracking_results.json", "w") as f:
#     import json
#     json.dump(data_to_save, f, indent=2)
# print("Dumped", len(data_to_save), "entries.")
# 
# from norfair import Detection
# import numpy as np
# 
# 
