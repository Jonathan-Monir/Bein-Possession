import cv2
import numpy as np
import os
import re
import time
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.cluster import MiniBatchKMeans  # Replacing KMeans with MiniBatchKMeans
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from concurrent.futures import ThreadPoolExecutor, as_completed
from norfair import Detection

def show_cropped_frame(crop, window_name='Cropped Frame'):
    """
    Displays a cropped image.
    
    :param crop: The cropped image/frame.
    :param window_name: Name of the display window (default: 'Cropped Frame').
    """
    cv2.imshow(window_name, crop)
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()  # Close the window

# -------------------------
# Feature Extraction
# -------------------------
def extract_deep_features(image, model):
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image)
    return features.flatten()

# -------------------------
# Cropping and Parsing
# -------------------------
def crop_players(image, boxes, image_shape):
    player_images = []
    img_h, img_w = image_shape[:2]
    for box in boxes:
        label, x_center, y_center, width, height = box
        if label == 1:  # Assuming label '1' corresponds to players
            x = int((x_center - width / 2) * img_w)
            y = int((y_center - height / 2) * img_h)
            w = int(width * img_w)
            h = int(height * img_h)
            player_crop = image[max(0, y):min(y + h, img_h), max(0, x):min(x + w, img_w)]
            player_images.append(player_crop)
    return player_images

def parse_yolo_labels(file_path):
    bounding_boxes = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = list(map(float, line.split()))
            cls, x1, x2, y1, y2 = parts
    return bounding_boxes

# -------------------------
# Color Extraction Functions
# -------------------------
def calculate_color_distance(color1, color2):
    hsv1 = cv2.cvtColor(np.uint8([[color1]]), cv2.COLOR_BGR2HSV)[0][0]
    hsv2 = cv2.cvtColor(np.uint8([[color2]]), cv2.COLOR_BGR2HSV)[0][0]
    rgb_dist = np.sqrt(np.sum((color1 - color2) ** 2))
    h_dist = min(abs(hsv1[0] - hsv2[0]), 180 - abs(hsv1[0] - hsv2[0])) / 90.0
    s_dist = abs(hsv1[1] - hsv2[1]) / 255.0
    v_dist = abs(hsv1[2] - hsv2[2]) / 255.0
    hsv_dist = np.sqrt(4 * h_dist ** 2 + s_dist ** 2 + v_dist ** 2)
    return 0.5 * rgb_dist + 0.5 * hsv_dist * 255

def apply_masking(crop):
    # convert to HSV once
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # define green range
    lower_green = np.array([35, 40, 40], dtype=np.uint8)
    upper_green = np.array([85, 255, 255], dtype=np.uint8)

    # mask out green
    green_mask  = cv2.inRange(hsv, lower_green, upper_green)
    player_mask = cv2.bitwise_not(green_mask)

    # optional: clean up small holes / noise
    kernel = np.ones((3,3), np.uint8)
    player_mask = cv2.morphologyEx(player_mask, cv2.MORPH_OPEN,  kernel)
    player_mask = cv2.morphologyEx(player_mask, cv2.MORPH_CLOSE, kernel)

    # apply and return only non-green pixels
    masked_crop = cv2.bitwise_and(crop, crop, mask=player_mask)
    return masked_crop.reshape(-1,3)[np.any(masked_crop.reshape(-1,3)!=0, axis=1)]

def extract_dominant_colors(crop, n_colors=3, debug=False):
    pixels = apply_masking(crop)
    if debug:
        print(f"Number of pixels after masking: {len(pixels)}")
    
    # Fallback if not enough pixels
    if len(pixels) < 10:
        h, w = crop.shape[:2]
        center_crop = crop[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        pixels = center_crop.reshape(-1, 3)
    
    # Use MiniBatchKMeans for speed
    kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=22, batch_size=512, max_iter=300)
    kmeans.fit(pixels)
    labels = kmeans.labels_
    unique_labels, counts = np.unique(labels, return_counts=True)
    centers = kmeans.cluster_centers_
    
    if debug:
        print("\nClustered Colors (BGR) and their counts:")
        for center, count in zip(centers, counts):
            print(f"  Color: {tuple(map(int, center))}, Count: {count}")
        visualize_colors(centers, counts)
    
    # Sort colors by frequency
    color_counts = list(zip(centers, counts))
    color_counts.sort(key=lambda x: x[1], reverse=True)
    dominant_colors = []
    for color, count in color_counts:
        color_tuple = tuple(map(int, color))
        if debug:
            print(f"Selected color: {color_tuple} with count {count}")
        dominant_colors.append(color_tuple)
        if len(dominant_colors) == 2:
            break
    while len(dominant_colors) < 2:
        dominant_colors.append((0, 0, 0))
    
    if debug:
        visualize_extraction(crop, dominant_colors)
    
    return dominant_colors

def visualize_colors(centers, counts):
    sorted_colors = [tuple(map(int, color)) for color, _ in sorted(zip(centers, counts), key=lambda x: x[1], reverse=True)]
    fig, ax = plt.subplots(figsize=(8, 2))
    color_bar = np.zeros((50, len(sorted_colors) * 50, 3), dtype=np.uint8)
    for i, color in enumerate(sorted_colors):
        color_bar[:, i * 50:(i + 1) * 50] = color
    plt.imshow(cv2.cvtColor(color_bar, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Clustered Colors")
    plt.show()

def visualize_extraction(crop, filtered_colors):
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 30, 30])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv_crop, lower_green, upper_green)
    masked_crop = cv2.bitwise_and(crop, crop, mask=cv2.bitwise_not(mask))
    masked_crop_rgb = cv2.cvtColor(masked_crop, cv2.COLOR_BGR2RGB)
    n_colors = len(filtered_colors)
    color_bar = np.zeros((50, n_colors * 50, 3), dtype=np.uint8)
    for i, color in enumerate(filtered_colors):
        color_bar[:, i * 50:(i + 1) * 50] = color
    color_bar_rgb = cv2.cvtColor(color_bar, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(crop_rgb)
    axes[0].axis("off")
    axes[0].set_title("Original Cropped Image")
    axes[1].imshow(masked_crop_rgb)
    axes[1].axis("off")
    axes[1].set_title("Masked Image")
    axes[2].imshow(color_bar_rgb)
    axes[2].axis("off")
    axes[2].set_title("Extracted Colors")
    plt.tight_layout()
    plt.show()

# -------------------------
# Clustering and Classification
# -------------------------
def classify_players_by_features(player_features, n_teams=2):
    start_time = time.perf_counter()
    kmeans = MiniBatchKMeans(n_clusters=n_teams, random_state=42, batch_size=512)
    labels = kmeans.fit_predict(player_features)
    end_time = time.perf_counter()
    clustering_time = end_time - start_time
    return labels

# -------------------------
# Visualization
# -------------------------
def visualize_results(vis_image, player_colors, team_labels, other_boxes, frame_index):
    COLORS = {
        0: (255, 255, 255),  # Ball - Gray
        1: (0, 0, 255),      # T1 - Red
        2: (255, 0, 0),      # T2 - Blue
        3: (0, 255, 0),      # REF - Green
        4: (255, 255, 0),    # GK1 - Cyan
        5: (0, 255, 255)     # GK2 - Yellow
    }
    LABELS = {
        0: "Ball",
        1: "T1",
        2: "T2",
        3: "REF",
        4: "GK1",
        5: "GK2"
    }
    
    cv2.putText(vis_image, f'Frame: {frame_index}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    for (color, box), team_label in zip(player_colors, team_labels):
        cls, x_center, y_center, width, height = box
        x1 = int(x_center)
        y1 = int(y_center)
        x2 = int(width)
        y2 = int(height)
        label_idx = int(team_label if team_label > 2 else team_label + 1)
        box_color = COLORS[label_idx]
        label_text = LABELS[label_idx]
        cv2.rectangle(vis_image, (x1, x2), (y1, y2), box_color, 2)
        cv2.putText(vis_image, label_text, (x1, y2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
    
    for box in other_boxes:
        cls, x1, x2, y1, y2 = box
        cls_idx = int(cls)
        box_color = COLORS.get(cls_idx, (128, 128, 128))
        label_text = LABELS.get(cls_idx, f"C{cls_idx}")
        cv2.rectangle(vis_image, (int(x1), int(x2)), (int(y1), int(y2)), box_color, 2)
        cv2.putText(vis_image, label_text, (int(x1), int(y2) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
    
    vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(vis_image_rgb)
    plt.axis('off')
    plt.title('Team Classification with Deep Features')
    plt.show()

# -------------------------
# Multi-Frame Clustering (No File Saving)
# -------------------------

def multi_frame_cluster(results, model=None, norfair=False):
    all_features = []   # Features for clustering
    features_info = []  # Mapping: (frame_index, original_box)
    
    for frame_idx, (frame, ball_detections, player_detections) in enumerate(results):
#         print(f"frame type: {type(frame)}")
#         print(f"ball_detections type: {type(ball_detections)}")
#         print(f"player_detections type: {type(player_detections)}")
# 
# 
#         print(f"ball_detection type: {type(ball_detections[0])}")
#         print(f"player_detection type: {type(player_detections[0])}")
        if model is not None:
            for box in player_detections:
                if norfair:
                    cls, x1, x2, y1, y2 = box
                else:
                    cls, x1, x2, y1, y2 = box

                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                crop = frame[y1:y2, x1:x2]
                if cls!=0:
                    show_cropped_frame(crop, "mltiframe")
                if crop.size == 0:
                    continue
                feat = extract_deep_features(crop, model)
                all_features.append(feat)
                features_info.append((frame_idx, box))
        else:
            player_colors, _ = extract_player_colors(frame, player_detections, norfair)
            for colors, box in player_colors:
                feat = np.array(colors).flatten()
                all_features.append(feat)
                features_info.append((frame_idx, box))
    
    all_features = np.array(all_features)
    if all_features.any():
        team_labels_global = classify_players_by_features(all_features, n_teams=2)
        
        new_player_detections = {i: [] for i in range(len(results))}
        for global_idx, label in enumerate(team_labels_global):
            frame_idx, original_box = features_info[global_idx]

            new_label = label + 1  # Map label to team IDs (0 reserved for ball)
            new_box = [new_label] + list(original_box[1:])
            new_player_detections[frame_idx].append(new_box)
        
        final_results = []
        for idx, (frame, ball_detections, _) in enumerate(results):
            updated_boxes = new_player_detections.get(idx, []) + ball_detections
            cleaned_boxes = []
            for box in updated_boxes:
                if isinstance(box, Detection):
                    class_id = box.data.get("id", 0)
                    
                    # Fallback size if scores are None
                    box_size = box.scores[0] if box.scores and len(box.scores) > 0 else 50

                    x_center, y_center = box.points[0]
                    x1 = int(x_center - box_size / 2)
                    y1 = int(y_center - box_size / 2)
                    x2 = int(x_center + box_size / 2)
                    y2 = int(y_center + box_size / 2)
                    crop = frame[y1:y2, x1:x2]

                    cleaned_boxes.append([class_id, x1, y1, x2, y2])
                else:
                    # Handle cases where box is a list (not a Detection)
                    if isinstance(box, list) and len(box) == 5:
                        cleaned_boxes.append(box)
                    else:
                        print(f"Skipping invalid box format: {box}")
            
            cleaned_boxes.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
            final_results.append((frame, ball_detections, cleaned_boxes))
    else:
        final_results = [(frame, None, None)]
    
    return final_results

# -------------------------
# Extract Player Colors
# -------------------------

def extract_player_colors(image, detections, norfair=True):
    player_colors = []
    other_boxes = []

    if image is None:
        print(" Error: Could not load image.")
        return [], []

    if detections is None or not isinstance(detections, list):
        print(" Warning: Detections is None or not a list.")
        return [], []

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for det in detections:

        if norfair and hasattr(det, 'points'):
            # Norfair Detection object
            if len(det.points[0]) != 2:
                print("Invalid detection points structure.")
                continue

            x1, y1 = det.points[0]
            x2, y2 = det.points[1]
#             print(f"xy: {x_center, y_center, width, height}")
#             x1 = int(x_center - width // 2)
#             y1 = int(y_center - height // 2)
#             x2 = int(x_center + width // 2)
#             y2 = int(y_center + height // 2)
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            cls = getattr(det, 'label', 1)

        elif isinstance(det, list) and len(det) == 5:
            # Standard YOLO-style list
            cls, x1, y1, x2, y2 = det
#             print(f"extr: {x1,x2,y1,y2}")
        else:
            print(f"Skipping unknown detection format: {det}")
            continue

        # Clamp to image boundaries
        x1, x2 = max(0, x1), min(image.shape[1], x2)
        y1, y2 = max(0, y1), min(image.shape[0], y2)

        if cls in range(1, 6):  # Players only
            crop = image_rgb[y1:y2, x1:x2]

            if crop.size == 0:
                print(f"Skipped zero-size crop for box: {x1},{y1},{x2},{y2}")
                continue
            dominant_colors = extract_dominant_colors(crop)
            player_colors.append((dominant_colors, (cls, x1, y1, x2, y2)))

        else:
            other_boxes.append((cls, x1, y1, x2, y2))

    return player_colors, other_boxes

# -------------------------
# Per-Frame Processing Function (Parallelized)
# -------------------------
def yolobbox2bbox(x,y,w,h):
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return x1, y1, x2, y2

def process_frame(frame_idx, frame, ball_detections, updated_detections, debug):
    start_frame_proc = time.perf_counter()

    if updated_detections is None:
        return frame_idx, [], [], []
    if debug:
        os.makedirs("team_1", exist_ok=True)
        os.makedirs("team_2", exist_ok=True)

        for det_idx, detection in enumerate(updated_detections):
            team_label = detection.data.get("team", None)
            if team_label in [1, 2]:
                # Assume the detection has 2 points: top-left and bottom-right
                if len(detection.points) == 2:
                    x1, y1 = detection.points[0]
                    x2, y2 = detection.points[1]
                else:
                    # fallback: use center point + fixed size box
                    x1, y1 = detection.points[0]
                    x2, y2 = detection.points[1]

                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                crop = frame[y1:y2, x1:x2]

                folder = "team_1" if team_label == 1 else "team_2"
                cv2.imwrite(os.path.join(folder, f"frame{frame_idx}_det{det_idx}.jpg"), crop)

    start_color = time.perf_counter()
    player_colors, other_boxes = extract_player_colors(frame, updated_detections)
    end_color = time.perf_counter()
    if debug:
        print(f"extract_player_colors for frame {frame_idx} took {end_color - start_color:.4f} seconds")

    # Map team labels for aggregation: team 1 → 0, team 2 → 1
    team_labels = []
    for detection in updated_detections:
        label = detection[0]
        if label in [1, 2]:
            team_labels.append(0 if label == 1 else 1)

    end_frame_proc = time.perf_counter()
    return frame_idx, player_colors, team_labels, other_boxes

# -------------------------
# Main Multi-Frame Function (with Parallel Processing)
# -------------------------
def main_multi_frame(results=None, debug=False):
    norfair = True
    if results is None:
        norfair = False
        start_total = time.perf_counter()
        results = []
        frames_dir = 'frames'   # Update this path if needed
        labels_dir = 'labels'   # Update this path if needed

        start_frames = time.perf_counter()
        frame_files = sorted(os.listdir(frames_dir), key=lambda f: int(re.findall(r'\d+', f)[0]))
        for frame_file in frame_files[7:10]:
            frame_path = os.path.join(frames_dir, frame_file)
            label_path = os.path.join(labels_dir, frame_file.replace('.jpg', '.txt'))
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            bounding_boxes = parse_yolo_labels(label_path)
            ball_detections = [box for box in bounding_boxes if box[0] == 0]
            player_detections = [box for box in bounding_boxes if box[0] == 1]
            results.append((frame, ball_detections, player_detections))
        end_frames = time.perf_counter()
        if debug:
            print(f"Frame reading and label parsing took {end_frames - start_frames:.4f} seconds")
    
    start_model = time.perf_counter()
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    model = Model(inputs=base_model.input, outputs=base_model.output)
    end_model = time.perf_counter()
    
    start_cluster = time.perf_counter()
    # Note: In this example we set model=None so that color features (via extract_dominant_colors) are used.
    updated_results = multi_frame_cluster(results, model=None, norfair=norfair)
    end_cluster = time.perf_counter()
    if debug:
        print(f"Multi-frame clustering took {end_cluster - start_cluster:.4f} seconds")
    
    # Process each frame in parallel and also collect team color information.
    processed_frames = {}
    with ThreadPoolExecutor() as executor:
        futures = []
        for frame_idx, (frame, ball_detections, updated_boxes) in enumerate(updated_results):
            futures.append(executor.submit(process_frame, frame_idx, frame, ball_detections, updated_boxes, debug))
        for future in as_completed(futures):
            frame_idx, player_colors, team_labels, other_boxes = future.result()
            processed_frames[frame_idx] = (player_colors, team_labels, other_boxes)
    
    # Aggregate dominant colors for each team from all frames.
    team1_colors = []
    team2_colors = []
    for frame_idx in sorted(processed_frames.keys()):
        player_colors, team_labels, _ = processed_frames[frame_idx]
        # Assume that player_colors list and team_labels list are aligned.
        for (colors, _), label in zip(player_colors, team_labels):
            # Use the most dominant color (first element of colors).
            if label == 0:
                team1_colors.append(colors[0])
            elif label == 1:
                team2_colors.append(colors[0])
    
    if team1_colors:
        team1_color = tuple(np.mean(team1_colors, axis=0).astype(int))
    else:
        team1_color = (0, 0, 0)
    if team2_colors:
        team2_color = tuple(np.mean(team2_colors, axis=0).astype(int))
    else:
        team2_color = (0, 0, 0)
    
    print(f"Team 1 color (BGR): {team1_color}")
    print(f"Team 2 color (BGR): {team2_color}")
    
    return updated_results, team1_color, team2_color

# Example usage:
if __name__ == '__main__':
    results_with_class_ids, team1_color, team2_color = main_multi_frame(debug=True)
    print("Multi-frame clustering complete. Processed results returned.")
    print("Team 1 Color (BGR):", team1_color)
    print("Team 2 Color (BGR):", team2_color)
