import cv2
import numpy as np
import random
from typing import List, Tuple

class BallTracker:
    def __init__(self, max_history=50, trail_color=(0, 255, 0), trail_thickness=2):
        """
        Initialize ball tracker with trail settings.
        
        Args:
            max_history: Maximum number of positions to remember for trail
            trail_color: BGR color tuple for the trail
            trail_thickness: Thickness of the trail lines
        """
        self.previous_positions = []  # Stores ball positions in image coordinates
        self.max_history = max_history
        self.trail_color = trail_color
        self.trail_thickness = trail_thickness

    def update_tracking(self, frame: np.ndarray, ball_detections: List[Tuple], team1_color, team2_color, team_poss) -> np.ndarray:
        """
        Update ball tracking and draw trails between positions using YOLO detections.
        
        Args:
            frame: Current video frame (OpenCV format)
            ball_detections: List of ball detections in YOLO format [class_id, x1, y1, x2, y2]
            
        Returns:
            Frame with tracking lines drawn
        """
        if team_poss == 1:
            color = sanitize_color(team1_color)
        elif team_poss == 2:
            color = sanitize_color(team2_color)
        else:
            pass
        color = (int(color[2]), int(color[1]), int(color[0]))

        if not ball_detections:
            # No ball detected, maintain existing trail but don't add new points
            if len(self.previous_positions) > 0:
                # Optionally fade out trail when ball is lost
                pass
            return frame

        # Take first detection if multiple exist
        _, x1, y1, x2, y2 = ball_detections[0]
        
        # Convert coordinates to integers
        current_position = (int((x1 + x2) // 2), int((y1 + y2) // 2))  # Ball center as integers

        # Draw trail if we have previous positions
        if len(self.previous_positions) > 0:
            # Draw lines between all previous positions for a smooth trail
            for i in range(len(self.previous_positions) - 1):
                cv2.line(
                    frame,
                    self.previous_positions[i],
                    self.previous_positions[i+1],
                    color,
                    self.trail_thickness
                )

            # Draw arrow from last position to current position
            if len(self.previous_positions) >= 2:
                cv2.arrowedLine(
                    frame,
                    self.previous_positions[-2],  # Start from second-to-last position
                    self.previous_positions[-1],  # Point to last position
                    (0, 0, 255),  # Red color for arrow
                    3,
                    tipLength=0.3
                )

        # Draw current ball position (ensure coordinates are integers)
        cv2.circle(frame, current_position, 5, (0, 255, 255), -1)  # Yellow circle

        # Update previous positions
        self.previous_positions.append(current_position)
        if len(self.previous_positions) > self.max_history:
            self.previous_positions.pop(0)

        return frame


def sanitize_color(color):
    # Ensure the color tuple is a 3-element tuple of plain ints.
    if isinstance(color, (list, tuple)) and len(color) == 3:
        return tuple(int(c) for c in color)
    return (0, 0, 0)

def draw_bounding_boxes_on_frames(results_with_class_ids, team1_color, team2_color, team_poss_list):
    """Draw bounding boxes on frames using OpenCV with ball tracking."""
    class_id_colors = {}  # Store unique colors for each class ID

    # Initialize ball tracker
    ball_tracker = BallTracker(
        max_history=20, 
        trail_color=(0, 255, 0),  # Green trail
        trail_thickness=2
    )
    
    processed_frames = []
    
    for i, (frame, ball_detections, updated_boxes) in enumerate(results_with_class_ids):
        # Ensure frame is in OpenCV format (NumPy array)
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
            
        # Draw ball trajectory (using only YOLO detections)
        if ball_detections:
            frame = ball_tracker.update_tracking(frame, ball_detections,team1_color, team2_color,team_poss_list[i])
        
        # Draw bounding boxes for objects (players, etc.)
        for box in updated_boxes:
            class_id, x1, y1, x2, y2 = box
            # Use provided team colors for player boxes.
            if class_id == 1:
                color = sanitize_color(team1_color)
            elif class_id == 2:
                color = sanitize_color(team2_color)
            else:
                continue
            
            color = (int(color[2]), int(color[1]), int(color[0]))
            # Draw the rectangle using the determined color
            cv2.rectangle(
                frame, 
                (int(x1), int(y1)), 
                (int(x2), int(y2)), 
                color, 
                2
            )
            
        processed_frames.append((frame, ball_detections, updated_boxes))
    
    return processed_frames

def save_video_from_frames(results_with_class_ids, output_path="output.mp4", fps=20):
    """Creates and saves a video from processed frames."""
    if not results_with_class_ids:
        print("No frames to save!")
        return

    first_frame = results_with_class_ids[0][0]
    height, width, _ = first_frame.shape

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame, _, _ in results_with_class_ids:
        out.write(frame)

    out.release()
    print(f"Video saved as {output_path}")
