def serialize_results(results_tracking):
    serializable = []
    for frame_idx, (frame, ball_dets, player_dets) in enumerate(results_tracking):
        def det_to_dict(det):
            # Extract points → [[x1,y1],[x2,y2]]
            pts = det.points.tolist()
            x1, y1 = map(int, pts[0])
            x2, y2 = map(int, pts[1])

            # Try these in order: det.label, det.class_id, det.data["id"]
            cls = None
            if hasattr(det, "label") and det.label is not None:
                cls = det.label
            elif hasattr(det, "class_id") and det.class_id is not None:
                cls = det.class_id
            elif isinstance(det.data, dict) and "id" in det.data:
                cls = det.data["id"]
            else:
                cls = -1  # or 0, whatever your “unknown” sentinel is

            return {
                "cls": int(cls),
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2
            }

        serializable.append({
            "frame_idx": frame_idx,
            "ball_detections":   [det_to_dict(d) for d in ball_dets],
            "player_detections": [det_to_dict(d) for d in player_dets],
        })

    return serializable

data_to_save = serialize_results(results_tracking)
with open("tracking_results.json", "w") as f:
    import json
    json.dump(data_to_save, f, indent=2)

from norfair import Detection
import numpy as np

def load_as_detections(json_path):
    raw = json.load(open(json_path))
    out = []
    for entry in raw:
        frame_idx = entry["frame_idx"]
        balls = [Detection(points=np.array([[d["x1"],d["y1"]],[d["x2"],d["y2"]]]),
                           label=d["cls"],
                           data={"id":d["cls"]})
                 for d in entry["ball_detections"]]
        players = [Detection(points=np.array([[d["x1"],d["y1"]],[d["x2"],d["y2"]]]),
                             label=d["cls"],
                             data={"id":d["cls"]})
                   for d in entry["player_detections"]]
        out.append((frame_idx, balls, players))
    return out
