import os
import cv2
import numpy as np
import PIL
from ultralytics import YOLO
from norfair import Tracker
from norfair.camera_motion import MotionEstimator
from norfair.distances import mean_euclidean
from preprosses import compute_noise, apply_nlm_denoising
import torch
from norfair import Tracker, Video
from tracking.inference.converter import Converter
# from tracking.inference import Converter
from tracking.soccer import Match, Player, Team
from tracking.soccer.draw import AbsolutePath
# from tracking.soccer.pass_event import Pass
from fill_miss_tracking import fill_results
import run_utils as ru

# Video and model paths
video_path = "manc.mp4"
fps = 10  # Target FPS for extraction


"""# Helpful functions"""


def delete_file(file_path):
    """
    Deletes the file at the specified file_path.

    Args:
        file_path (str): The path to the file to be deleted.
    """
    if os.path.isfile(file_path):
        try:
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        except Exception as e:
            print(f"An error occurred while deleting the file: {e}")
    else:
        print(f"File not found: {file_path}")





def process_video(yolo_path: str,
                  video_path: str,
                  target_fps: float,
                  start_second: float,
                  end_second: float):
    """
    Process a video within a specified time range.

    Args:
        yolo_path (str): Path to the YOLO model.
        video_path (str): Path to the input video file.
        target_fps (float): Desired processing frames per second. Set to -1 to match the video's original FPS.
        start_second (float): Start time in seconds.
        end_second (float): End time in seconds.

    Returns:
        results: List of tuples (frame, ball_tracks, player_tracks).
        motion_estimators: List of MotionEstimator states.
        coord_transformations: List of coordinate transformation matrices.
        video: Video writer object for the output video.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    coord_transformations = []
    motion_estimators = []

    # Initialize detectors and trackers
    yolo_detector = YOLO(yolo_path)
    yolo_detector.model.to(device)

    player_tracker = Tracker(distance_function=mean_euclidean,
                             distance_threshold=250,
                             initialization_delay=3,
                             hit_counter_max=90)
    ball_tracker = Tracker(distance_function=mean_euclidean,
                           distance_threshold=150,
                           initialization_delay=20,
                           hit_counter_max=2000)
    motion_estimator = MotionEstimator()

    # Open video
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # If target_fps is -1 or non-positive, match original FPS
    if target_fps <= 0:
        target_fps = orig_fps

    # Compute frame indices for the given time window
    start_frame = int(start_second * orig_fps)
    end_frame = int(end_second * orig_fps)
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame, min(end_frame, total_frames - 1))

    # Determine skip interval to achieve target_fps
    skip_interval = 1 if target_fps == orig_fps else int(round(orig_fps / target_fps))

    # Prepare video writer
    video = Video(input_path=video_path, output_path="new_vid.mp4")

    results = []
    frame_idx = 0

    # Iterate through the video frames
    for i, frame in enumerate(video):
        # Skip until start_frame
        if i < start_frame:
            continue
        # Stop after end_frame
        if i > end_frame:
            break

        # Skip frames to match target FPS
        if skip_interval > 1 and (i - start_frame) % skip_interval != 0:
            continue

        frame_idx += 1

        # Detect ball and players
        ball_detections = ru.get_detections(yolo_detector,
                                           frame,
                                           class_id=0,
                                           confidence_threshold=0.3)
        player_detections = ru.get_detections(yolo_detector,
                                             frame,
                                             class_id=1,
                                             confidence_threshold=0.35)

        detections = ball_detections + player_detections
        try:
            coord_transformation = ru.update_motion_estimator(
                motion_estimator=motion_estimator,
                detections=detections,
                frame=frame
            )
        except Exception:
            coord_transformation = None

        # Update trackers
        player_track_objects = player_tracker.update(
            detections=player_detections,
            coord_transformations=coord_transformation
        )
        ball_track_objects = ball_tracker.update(
            detections=ball_detections,
            coord_transformations=coord_transformation
        )

        # Convert tracked objects to detection format
        player_tracks = Converter.TrackedObjects_to_Detections_nor(
            player_track_objects,
            cls=1
        )
        ball_tracks = Converter.TrackedObjects_to_Detections_nor(
            ball_track_objects,
            cls=0
        )



        # Fallback if ball not detected
        if not(ball_tracks) and not(ball_detections):
            continue
        elif not(ball_tracks):
            ball_tracks = ball_detections


        if not(player_tracks) and not(player_detections):
            continue
        elif not(player_tracks):
            player_tracks = player_detections
        elif len(player_detections)<5:
            continue


        # Collect results
        results.append((frame, ball_tracks, player_tracks))
        coord_transformations.append(coord_transformation)
        motion_estimators.append(motion_estimator)

    return results, motion_estimators, coord_transformations, video



if __name__ == "__main__":
    process_video("yolo8.pt", "jooooooooo.mp4", 30)
