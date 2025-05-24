from norfair import Detection
import numpy as np

def yolo_to_norfair_detections(yolo_detections, confidence_threshold=0.0):
    """
    Converts YOLO detections to Norfair Detection objects using corner points (like get_detections).

    Parameters
    ----------
    yolo_detections : List[List[float]]
        Each detection is [class_id, x1, y1, x2, y2, conf]
    confidence_threshold : float
        Minimum confidence to accept a detection

    Returns
    -------
    List[norfair.Detection]
        List of Norfair Detection objects with corner points.
    """
    detections = []

    for det in yolo_detections:
        if len(det) == 6:
            class_id, x1, y1, x2, y2, conf = det
        else:
            # If confidence isn't passed, assume it's high
            class_id, x1, y1, x2, y2 = det
            conf = 1.0

        if conf < confidence_threshold:
            continue

        # Use top-left and bottom-right as keypoints
        points = np.array([[x1, y1], [x2, y2]])

        data = {
            "class_id": int(class_id),
            "confidence": float(conf)
        }

        detection = Detection(points=points, data=data)
        detections.append(detection)

    return detections
