import cv2
import numpy as np
import os
import re
from collections import Counter
from sklearn.cluster import MiniBatchKMeans
from norfair import Detection
from make_vid import vid  # your video‑making function

# -------------------------
# Utility Functions
# -------------------------

def parse_yolo_labels(file_path):
    with open(file_path, 'r') as f:
        return [list(map(float, line.split())) for line in f]

_mask_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# -------------------------
# Color Extraction
# -------------------------

def apply_masking(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    player_mask = cv2.bitwise_not(mask)
    player_mask = cv2.morphologyEx(player_mask, cv2.MORPH_OPEN, _mask_kernel)
    player_mask = cv2.morphologyEx(player_mask, cv2.MORPH_CLOSE, _mask_kernel)
    pixels = crop.reshape(-1, 3)[player_mask.flatten() > 0]
    return pixels if pixels.size else None


def extract_dominant_colors(crop, n_colors=2):
    pixels = apply_masking(crop)
    if pixels is None or len(pixels) < 10:
        h, w = crop.shape[:2]
        pixels = crop[h//4:3*h//4, w//4:3*w//4].reshape(-1, 3)
    kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=22,
                             batch_size=512, max_iter=100)
    kmeans.fit(pixels)
    counts = np.bincount(kmeans.labels_)
    centers = kmeans.cluster_centers_
    top_idx = np.argsort(-counts)[:n_colors]
    return [tuple(map(int, centers[i])) for i in top_idx]

# -------------------------
# Clustering per Frame
# -------------------------

def multi_frame_cluster(results, n_teams=2):
    stream_kmeans = MiniBatchKMeans(n_clusters=n_teams, random_state=42,
                                    batch_size=512, max_iter=100)
    feature_queue = []
    info_queue = []  # list of (frame_idx, cls, (x1,y1,x2,y2))

    for idx, (frame, ball, plyr) in enumerate(results):
        for det in plyr:
            # unpack detection → cls, coords
            if hasattr(det, 'points') and len(det.points) == 2:
                (x1, y1), (x2, y2) = det.points
                cls = getattr(det, 'label', 1)
            elif isinstance(det, list) and len(det) == 5:
                cls, x1, y1, x2, y2 = det
            else:
                continue

            x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
            crop = frame[y1_i:y2_i, x1_i:x2_i]
            if crop.size == 0:
                continue

            feat = np.array(extract_dominant_colors(crop)).flatten()
            try:
                stream_kmeans.partial_fit([feat])
            except ValueError:
                continue

            feature_queue.append(feat)
            info_queue.append((idx, cls, (x1, y1, x2, y2)))

    if not feature_queue:
        return [
            (frame, [b if isinstance(b, list) else [
                            getattr(b, 'label', b.data.get('id',0)),
                            int(b.points[0][0]), int(b.points[0][1]),
                            int(b.points[1][0]), int(b.points[1][1])
                        ] for b in (ball + plyr)])
            for frame, ball, plyr in results
        ]

    labels = stream_kmeans.predict(feature_queue)
    per_frame = {i: [] for i in range(len(results))}
    for (idx, cls, box), lbl in zip(info_queue, labels):
        per_frame[idx].append([lbl + 1, *map(int, box)])

    final = []
    for i, (frame, ball, _) in enumerate(results):
        combined = per_frame.get(i, []) + (ball or [])
        cleaned = []
        for b in combined:
            if isinstance(b, Detection):
                (x1, y1), (x2, y2) = b.points
                size = b.scores[0] if b.scores else 50
                cid = b.data.get('id', 0)
                x1_i = int(x1 - size/2)
                y1_i = int(y1 - size/2)
                x2_i = x1_i + int(size)
                y2_i = y1_i + int(size)
                cleaned.append([cid, x1_i, y1_i, x2_i, y2_i])
            else:
                cleaned.append(b)
        cleaned.sort(key=lambda x: tuple(x))
        final.append((frame, cleaned))

    return final

# -------------------------
# Main Multi‑Frame Entry
# -------------------------

def main_multi_frame(results=None, debug=False):
    if results is None:
        frames_dir, labels_dir = 'frames', 'labels'
        files = sorted(os.listdir(frames_dir),
                       key=lambda f: int(re.search(r'\d+', f).group()))
        results = []
        for fname in files:
            img = cv2.imread(os.path.join(frames_dir, fname))
            if img is None:
                continue
            lbls = parse_yolo_labels(
                os.path.join(labels_dir, fname.replace('.jpg', '.txt'))
            )
            ball = [b for b in lbls if b[0] == 0]
            plyr = [b for b in lbls if b[0] == 1]
            results.append((img, ball, plyr))

    updated = multi_frame_cluster(results)

    # re‑extract colors for team‑color averaging
    t1, t2 = [], []
    for frame, dets in updated:
        for det in dets:
            cls, x1, y1, x2, y2 = det
            crop = frame[y1:y2, x1:x2]
            cols = extract_dominant_colors(crop)
            if cls == 1:
                t1.append(cols[0])
            else:
                t2.append(cols[0])

    team1_color = tuple(map(int, np.mean(t1, axis=0))) if t1 else (0, 0, 0)
    team2_color = tuple(map(int, np.mean(t2, axis=0))) if t2 else (0, 0, 0)

    return updated, team1_color, team2_color

# -------------------------
# New: Save Cropped Players
# -------------------------
def save_team_player_images(results, base_dir='team_players'):
    """
    Creates folders 'team1_players' and 'team2_players' inside base_dir
    and saves cropped player images accordingly.
    """
    team_dirs = {
        1: os.path.join(base_dir, 'team1_players'),
        2: os.path.join(base_dir, 'team2_players')
    }
    # make base and sub-directories
    os.makedirs(base_dir, exist_ok=True)
    for d in team_dirs.values():
        os.makedirs(d, exist_ok=True)

    # iterate over frames and detections
    for frame_idx, (frame, dets) in enumerate(results):
        for idx, det in enumerate(dets):
            cls, x1, y1, x2, y2 = det
            # ensure coordinates are ints and within frame bounds
            x1, y1, x2, y2 = map(lambda v: max(int(v), 0), [x1, y1, x2, y2])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            # save image
            fname = f'frame{frame_idx:04d}_player{idx}.jpg'
            out_path = os.path.join(team_dirs.get(cls, base_dir), fname)
            cv2.imwrite(out_path, crop)

# -------------------------
# If run as script:
# -------------------------

if __name__ == "__main__":
    updated, team1_color, team2_color = main_multi_frame()
    # save cropped player images by team
    save_team_player_images(updated)

    # flatten all detections into one list for vid(...)
    results_for_vid = [det
                       for _, dets in updated
                       for det in dets]

    _, frames = vid(
        results_for_vid,
        team1_color=team1_color,
        team2_color=team2_color
    )  # now `frames` is your annotated video frames
