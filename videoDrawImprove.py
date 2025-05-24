
from PIL import Image
import PIL
import cv2
import numpy as np
import random
from typing import List, Tuple

from tracking.soccer.draw import AbsolutePath
from tracking.soccer.match import Match
from tracking.soccer.player import Player
from tracking.soccer.team import Team
from run_utils import get_main_ball
from norfair import Video



def draw_bounding_boxes_on_frames(results_with_class_ids, team1_color, team2_color, team_poss_list, motion_estimator, coord_transformations, video):
# motion_estimator = MotionEstimator()
# coord_transformations = None


# Convert RGB to BGR
    team1_color_bgr = team1_color[::-1]
    team2_color_bgr = team2_color[::-1]

    team1 = Team(
        name="Chelsea",
        abbreviation="CHE",
        color=team1_color_bgr
    )

    team2 = Team(name="Man City", abbreviation="MNC", color=team2_color_bgr)
    teams = [team1, team2]

# Paths
    path = AbsolutePath()

# Get Counter img

    for i, (frame, ball_detections, players_detections) in enumerate(results_with_class_ids):


        # Draw
        frame = PIL.Image.fromarray(frame)

        ball = get_main_ball(ball_detections)
        if players_detections:
            players = Player.from_detections(detections=players_detections, teams=teams)

            if True:
                frame = Player.draw_players(
                    players=players, frame=frame, confidence=False, id=True, teams=teams
                )

                frame = path.draw(
                    img=frame,
                    detection=ball.detection,
                    coord_transformations=coord_transformations[i],
                    poss=1,
                    teams=teams,
                )

#             frame = match.draw_possession_counter(
#                 frame, counter_background=possession_background, debug=False
#             )

                if ball:
                    frame = ball.draw(frame)

            if False:
                pass_list = match.passes

                frame = Pass.draw_pass_list(
                    img=frame, passes=pass_list, coord_transformations=coord_transformations
                )

                frame = match.draw_passes_counter(
                    frame, counter_background=passes_background, debug=False
                )

        frame = np.array(frame)

        # Write video
        video.write(frame)
        print(f"video is finished")


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
