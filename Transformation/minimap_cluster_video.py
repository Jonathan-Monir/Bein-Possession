import cv2
import torch
import yaml
import numpy as np
import torchvision.transforms as T
from ultralytics import YOLO
import inference as inf
import argparse
from sklearn.cluster import KMeans
import torchvision.models as models
from torchvision import transforms

# Field line coordinates remain global for simplicity.
LINES_COORDS = [
    [[0., 54.16, 0.], [16.5, 54.16, 0.]],
    [[16.5, 13.84, 0.], [16.5, 54.16, 0.]],
    [[16.5, 13.84, 0.], [0., 13.84, 0.]],
    [[88.5, 54.16, 0.], [105., 54.16, 0.]],
    [[88.5, 13.84, 0.], [88.5, 54.16, 0.]],
    [[88.5, 13.84, 0.], [105., 13.84, 0.]],
    [[0., 37.66, -2.44], [0., 30.34, -2.44]],
    [[0., 37.66, 0.], [0., 37.66, -2.44]],
    [[0., 30.34, 0.], [0., 30.34, -2.44]],
    [[105., 37.66, -2.44], [105., 30.34, -2.44]],
    [[105., 30.34, 0.], [105., 30.34, -2.44]],
    [[105., 37.66, 0.], [105., 37.66, -2.44]],
    [[52.5, 0., 0.], [52.5, 68, 0.]],
    [[0., 68., 0.], [105., 68., 0.]],
    [[0., 0., 0.], [0., 68., 0.]],
    [[105., 0., 0.], [105., 68., 0.]],
    [[0., 0., 0.], [105., 0., 0.]],
    [[0., 43.16, 0.], [5.5, 43.16, 0.]],
    [[5.5, 43.16, 0.], [5.5, 24.84, 0.]],
    [[5.5, 24.84, 0.], [0., 24.84, 0.]],
    [[99.5, 43.16, 0.], [105., 43.16, 0.]],
    [[99.5, 43.16, 0.], [99.5, 24.84, 0.]],
    [[99.5, 24.84, 0.], [105., 24.84, 0.]]
]

class Detector:
    """Wraps the YOLO model for detecting players and ball."""
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def get_bounding_boxes(self, image_input):
        # image_input can be a file path or a numpy array (frame)
        results = self.model(image_input, imgsz=768)
        player_bboxes = []
        ball_bbox = None

        for result in results:
            for box in result.boxes:
                bbox = box.xyxy[0].tolist()  # [x_min, y_min, x_max, y_max]
                class_id = int(box.cls[0])
                if class_id == 1:  # Player class ID
                    player_bboxes.append([int(coord) for coord in bbox])
                elif class_id == 0:  # Ball class ID
                    ball_bbox = [int(coord) for coord in bbox]  # Take the last detected ball
        return player_bboxes, ball_bbox

class FieldTransformer:
    """
    Handles the transformation from image coordinates to field (world) coordinates.
    """
    def __init__(self, H_inv: np.ndarray, field_offset=(105/2, 68/2)):
        self.H_inv = H_inv
        self.field_offset = field_offset

    def image_to_field_point(self, bbox):
        if bbox is None:
            return None
        # Compute the center of the bottom edge of the bounding box
        x_center = (bbox[0] + bbox[2]) / 2
        y_bottom = bbox[3]
        pt_img = np.array([x_center, y_bottom, 1]).reshape(3, 1)
        # Transform to field coordinates using the inverse homography
        pt_field = self.H_inv @ pt_img
        pt_field /= pt_field[2]  # Normalize homogeneous coordinates

        # Adjust with the field offset
        X = pt_field[0, 0] + self.field_offset[0]
        Y = pt_field[1, 0] + self.field_offset[1]
        return (X, Y)

    @staticmethod
    def compute_homography(P: np.ndarray):
        H = np.array([
            [P[0, 0], P[0, 1], P[0, 3]],
            [P[1, 0], P[1, 1], P[1, 3]],
            [P[2, 0], P[2, 1], P[2, 3]]
        ])
        return H

class Minimap:
    """
    Creates and draws on a minimap based on the field parameters.
    """
    def __init__(self, field_width=105, field_length=68, scale=10, ss_factor=4,
                 margin_left=5, margin_right=5, margin_top=5, margin_bottom=5):
        self.field_width = field_width
        self.field_length = field_length
        self.scale = scale
        self.ss_factor = ss_factor
        self.margin_left = margin_left
        self.margin_right = margin_right
        self.margin_top = margin_top
        self.margin_bottom = margin_bottom
        self.minimap_hr, self.m_left, self.m_top = self.create_minimap()

    def create_minimap(self):
        total_width = self.field_width + self.margin_left + self.margin_right
        total_length = self.field_length + self.margin_top + self.margin_bottom
        hr_width = int(total_width * self.scale * self.ss_factor)
        hr_length = int(total_length * self.scale * self.ss_factor)
        minimap = np.full((hr_length, hr_width, 3), (31, 28, 23), dtype=np.uint8)
        thickness = 2 * self.ss_factor
        cv2.rectangle(minimap, (0, 0), (hr_width - 1, hr_length - 1), (96, 101, 104), thickness)
        return minimap, self.margin_left, self.margin_top

    def draw_field_lines(self, field_lines):
        effective_scale = self.scale * self.ss_factor
        thickness = 2 * self.ss_factor

        for line in field_lines:
            p1, p2 = line
            X1, Y1 = p1[0], p1[1]
            X2, Y2 = p2[0], p2[1]
            pt1 = (int((self.margin_left + X1) * effective_scale),
                   int((self.margin_top + (self.field_length - Y1)) * effective_scale))
            pt2 = (int((self.margin_left + X2) * effective_scale),
                   int((self.margin_top + (self.field_length - Y2)) * effective_scale))
            cv2.line(self.minimap_hr, pt1, pt2, (96, 101, 104), thickness)

        # Draw center circle
        center_field = (self.field_width / 2, self.field_length / 2)
        center_pixel = (int((self.margin_left + center_field[0]) * effective_scale),
                        int((self.margin_top + (self.field_length - center_field[1])) * effective_scale))
        center_circle_radius = 9.15  # meters
        radius_pixel = int(center_circle_radius * effective_scale)
        cv2.circle(self.minimap_hr, center_pixel, radius_pixel, (96, 101, 104), thickness, lineType=cv2.LINE_AA)

        # Draw penalty arcs
        penalty_arc_radius = 9.15
        penalty_spots = [(11, 34), (94, 34)]
        arc_angles = [(307, 360+53), (127, 233)]
        for i, (penalty_x, penalty_y) in enumerate(penalty_spots):
            center_arc = (int((self.margin_left + penalty_x) * effective_scale),
                          int((self.margin_top + (self.field_length - penalty_y)) * effective_scale))
            radius_arc_pixel = int(penalty_arc_radius * effective_scale)
            start_angle, end_angle = arc_angles[i]
            cv2.ellipse(self.minimap_hr, center_arc, (radius_arc_pixel, radius_arc_pixel),
                        0, start_angle, end_angle, (96, 101, 104), thickness, lineType=cv2.LINE_AA)
        return self.minimap_hr

    def draw_objects_with_team_info(self, player_detections, ball_field_position):
        """
        Draw players with team-specific colors on the minimap.
        Each element in player_detections should be a dict with keys:
          - 'field_position': (X, Y)
          - 'team': team id (1 or 2)
        The ball is drawn as before.
        """
        effective_scale = self.scale * self.ss_factor

        for det in player_detections:
            pos = det["field_position"]
            team = det["team"]
            # Adjust Y if needed (e.g., shifting for player height)
            X, Y = pos
            Y_adjusted = Y - 2  # adjust if necessary
            pt = (int((self.margin_left + X) * effective_scale),
                  int((self.margin_top + Y_adjusted) * effective_scale))
            # Team colors: team 1 = red (BGR: (0,0,255)), team 2 = blue (BGR: (255,0,0))
            color = (0, 0, 255) if team == 1 else (255, 0, 0)
            cv2.circle(self.minimap_hr, pt, radius=12 * self.ss_factor, color=color, thickness=-1)

        # Draw ball (yellow filled circle)
        if ball_field_position:
            X, Y = ball_field_position
            pt = (int((self.margin_left + X) * effective_scale),
                  int((self.margin_top + Y) * effective_scale))
            cv2.circle(self.minimap_hr, pt, radius=3 * self.ss_factor, color=(0, 255, 255), thickness=6 * self.ss_factor)
        return self.minimap_hr

    def get_final_minimap(self):
        target_width = int((self.field_width + self.margin_left + self.margin_right) * self.scale)
        target_height = int((self.field_length + self.margin_top + self.margin_bottom) * self.scale)
        minimap_final = cv2.resize(self.minimap_hr, (target_width, target_height), interpolation=cv2.INTER_AREA)
        return minimap_final

class CameraCalibrator:
    """
    Loads the camera calibration models and computes the projection matrix.
    """
    def __init__(self, cfg_path, cfg_line_path, kp_model_path, line_model_path):
        self.cfg = yaml.safe_load(open(cfg_path, 'r'))
        self.cfg_l = yaml.safe_load(open(cfg_line_path, 'r'))

        self.model_kp = inf.get_cls_net(self.cfg)
        loaded_state = torch.load(kp_model_path, map_location=torch.device('cpu'))
        self.model_kp.load_state_dict(loaded_state)
        self.model_kp.eval()

        self.model_line = inf.get_cls_net_l(self.cfg_l)
        loaded_state_l = torch.load(line_model_path, map_location=torch.device('cpu'))
        self.model_line.load_state_dict(loaded_state_l)
        self.model_line.eval()

    def calibrate(self, frame, cam, kp_threshold, line_threshold):
        final_params_dict = inf.inference(cam, frame, self.model_kp, self.model_line, kp_threshold, line_threshold)
        P = inf.projection_from_cam_params(final_params_dict)
        return P

def main():
    # You can adjust these parameters as needed.
    input_type = "image"  # "image" or "video"
    input_path = r"examples\tactical-cam-angle_3340085.jpg"  # or image path
    save_path = r"examples\output.jpg"  # leave as None if you do not wish to save

    kp_threshold = 0.1486
    line_threshold = 0.3880
    margin_left = 2
    margin_right = 2
    margin_top = 2
    margin_bottom = 2
    scale = 10
    ss_factor = 12  # supersampling factor

    # Initialize the detector with the YOLO model and camera calibrator.
    detector = Detector("yolov11y_0.797R.pt")
    calibrator = CameraCalibrator("config/hrnetv2_w48.yaml", "config/hrnetv2_w48_l.yaml", "SV_kp", "SV_lines")

    # Initialize the feature extractor and preprocessing pipeline (for clustering).
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_extractor = models.resnet50(pretrained=True)
    feature_extractor.fc = torch.nn.Identity()
    feature_extractor.eval()
    feature_extractor.to(device)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if input_type == "image":
        # Process a single image.
        frame = cv2.imread(input_path)
        if frame is None:
            print(f"Error loading image: {input_path}")
            return

        frame_height, frame_width = frame.shape[:2]
        cam = inf.FramebyFrameCalib(iwidth=frame_width, iheight=frame_height, denormalize=True)
        # Calibrate camera and compute the homography from the single image.
        P = calibrator.calibrate(frame, cam, kp_threshold, line_threshold)
        H = FieldTransformer.compute_homography(P)
        H_inv = np.linalg.inv(H)
        transformer = FieldTransformer(H_inv)

        # Detect players and ball.
        player_bboxes, ball_bbox = detector.get_bounding_boxes(frame)
        player_detections = []
        # For each player bbox, extract ROI and compute feature.
        for bbox in player_bboxes:
            roi = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            if roi.size == 0:
                continue
            with torch.no_grad():
                input_tensor = preprocess(roi).unsqueeze(0).to(device)
                feature = feature_extractor(input_tensor).squeeze().cpu().numpy()
            player_detections.append({'frame_idx': 0, 'bbox': bbox, 'feature': feature})

        if len(player_detections) == 0:
            print("No player detections found!")
            return

        # For a single image, clustering may be trivial.
        features = np.array([det['feature'] for det in player_detections])
        if len(features) < 2:
            # Only one player detected: assign team 1.
            team_labels = [0]
        else:
            kmeans = KMeans(n_clusters=2, random_state=42).fit(features)
            team_labels = kmeans.labels_
        for det, label in zip(player_detections, team_labels):
            det['team'] = int(label) + 1

        # Compute field positions.
        players_with_field = []
        for det in player_detections:
            field_pos = transformer.image_to_field_point(det['bbox'])
            if field_pos:
                det['field_position'] = field_pos
                players_with_field.append(det)
        ball_field_position = transformer.image_to_field_point(ball_bbox) if ball_bbox is not None else None

        # Create and draw the minimap.
        minimap_obj = Minimap(field_width=105, field_length=68, scale=scale, ss_factor=ss_factor,
                              margin_left=margin_left, margin_right=margin_right,
                              margin_top=margin_top, margin_bottom=margin_bottom)
        minimap_obj.draw_field_lines(LINES_COORDS)
        minimap_obj.draw_objects_with_team_info(players_with_field, ball_field_position)
        minimap_final = minimap_obj.get_final_minimap()

        cv2.imshow("Minimap", minimap_final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if save_path:
            cv2.imwrite(save_path, minimap_final)

    else:  # video input
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error opening video file: {input_path}")
            return

        # Read the first frame for calibration.
        ret, frame = cap.read()
        if not ret:
            print("Error reading first frame from video")
            cap.release()
            return

        frame_height, frame_width = frame.shape[:2]
        cam = inf.FramebyFrameCalib(iwidth=frame_width, iheight=frame_height, denormalize=True)
        P = calibrator.calibrate(frame, cam, kp_threshold, line_threshold)
        H = FieldTransformer.compute_homography(P)
        H_inv = np.linalg.inv(H)
        transformer = FieldTransformer(H_inv)

        # -------------------------
        # FIRST PASS: Detection & Feature Extraction for Clustering
        # -------------------------
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame
        player_detections = []  # List of dicts: each contains frame_idx, bbox, feature
        ball_detections = {}    # Mapping: frame_idx -> ball bbox
        frame_idx = 0

        print("Starting first pass: detection and feature extraction...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            player_bboxes, ball_bbox = detector.get_bounding_boxes(frame)
            for bbox in player_bboxes:
                # Extract ROI from the frame.
                roi = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                if roi.size == 0:
                    continue
                with torch.no_grad():
                    input_tensor = preprocess(roi).unsqueeze(0).to(device)
                    feature = feature_extractor(input_tensor).squeeze().cpu().numpy()
                player_detections.append({
                    'frame_idx': frame_idx,
                    'bbox': bbox,
                    'feature': feature
                })
            if ball_bbox is not None:
                ball_detections[frame_idx] = ball_bbox

            frame_idx += 1

        cap.release()
        print("First pass complete. Total player detections:", len(player_detections))

        if len(player_detections) == 0:
            print("No player detections found!")
            return

        # Cluster the player features into two teams.
        features = np.array([det['feature'] for det in player_detections])
        print("Clustering detections into 2 teams...")
        if len(features) < 2:
            team_labels = [0] * len(features)
        else:
            kmeans = KMeans(n_clusters=2, random_state=42).fit(features)
            team_labels = kmeans.labels_
        for det, label in zip(player_detections, team_labels):
            det['team'] = int(label) + 1

        # Build a mapping from frame index to list of player detections.
        frame_to_player_dets = {}
        for det in player_detections:
            frame_to_player_dets.setdefault(det['frame_idx'], []).append(det)

        # -------------------------
        # SECOND PASS: Draw Minimap with Team Colors
        # -------------------------
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        if save_path:
            # The minimap dimensions based on field size and margins.
            out_width = int((105 + margin_left + margin_right) * scale)
            out_height = int((68 + margin_top + margin_bottom) * scale)
            fps = cap.get(cv2.CAP_PROP_FPS)
            out = cv2.VideoWriter(save_path, fourcc, fps, (out_width, out_height))

        frame_idx = 0
        print("Starting second pass: drawing minimap...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get stored detections for the current frame.
            player_dets = frame_to_player_dets.get(frame_idx, [])
            players_with_field = []
            for det in player_dets:
                field_pos = transformer.image_to_field_point(det['bbox'])
                if field_pos:
                    det['field_position'] = field_pos
                    players_with_field.append(det)
            # Get ball detection for current frame.
            ball_bbox = ball_detections.get(frame_idx, None)
            ball_field_position = transformer.image_to_field_point(ball_bbox) if ball_bbox is not None else None

            # Create minimap and draw field lines.
            minimap_obj = Minimap(field_width=105, field_length=68, scale=scale, ss_factor=ss_factor,
                                  margin_left=margin_left, margin_right=margin_right,
                                  margin_top=margin_top, margin_bottom=margin_bottom)
            minimap_obj.draw_field_lines(LINES_COORDS)
            minimap_obj.draw_objects_with_team_info(players_with_field, ball_field_position)
            minimap_final = minimap_obj.get_final_minimap()

            cv2.imshow("Minimap", minimap_final)
            if out is not None:
                out.write(minimap_final)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1

        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
