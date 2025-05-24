

from norfair import Detection
import cv2
import norfair
import numpy as np



class Ball:
    def __init__(self, detection: norfair.Detection):
        """
        Initialize Ball

        Parameters
        ----------
        detection : norfair.Detection
            norfair.Detection containing the ball
        """
        self.detection = detection
        self.color = None

    def set_color(self, match: "Match"):
        """
        Sets the color of the ball to the team color with the ball possession in the match.

        Parameters
        ----------
        match : Match
            Match object
        """
        if match.team_possession is None:
            return

        self.color = match.team_possession.color

        if self.detection:
            self.detection.data["color"] = match.team_possession.color

    def get_center(self, points: np.array) -> tuple:
        """
        Returns the center of the points

        Parameters
        ----------
        points : np.array
            2D points

        Returns
        -------
        tuple
            (x, y) coordinates of the center
        """
        x1, y1 = points[0]
        x2, y2 = points[1]

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        return (center_x, center_y)

    @property
    def center(self) -> tuple:
        """
        Returns the center of the ball

        Returns
        -------
        tuple
            Center of the ball (x, y)
        """
        if self.detection is None:
            return None

        center = self.get_center(self.detection.points)
        round_center = np.round_(center)

        return round_center

    @property
    def center_abs(self) -> tuple:
        """
        Returns the center of the ball in absolute coordinates

        Returns
        -------
        tuple
            Center of the ball (x, y)
        """
        if self.detection is None:
            return None

        center = self.get_center(self.detection.absolute_points)
        round_center = np.round_(center)

        return round_center

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw the ball on the frame

        Parameters
        ----------
        frame : np.ndarray
            Frame to draw on

        Returns
        -------
        np.ndarray
            Frame with ball drawn
        """
        if self.detection is None:
            return frame

        return Draw.draw_detection(self.detection, frame)

    def __str__(self):
        return f"Ball: {self.center}"


from math import sqrt
from typing import List

import norfair
import numpy as np
import PIL


def get_text_size(draw, text, font):
    # Get bounding box of text: returns (left, top, right, bottom)
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width, height

class Draw:
    @staticmethod
    def draw_rectangle(
        img: PIL.Image.Image,
        origin: tuple,
        width: int,
        height: int,
        color: tuple,
        thickness: int = 2,
    ) -> PIL.Image.Image:
        """
        Draw a rectangle on the image

        Parameters
        ----------
        img : PIL.Image.Image
            Image
        origin : tuple
            Origin of the rectangle (x, y)
        width : int
            Width of the rectangle
        height : int
            Height of the rectangle
        color : tuple
            Color of the rectangle (BGR)
        thickness : int, optional
            Thickness of the rectangle, by default 2

        Returns
        -------
        PIL.Image.Image
            Image with the rectangle drawn
        """

        draw = PIL.ImageDraw.Draw(img)
        draw.rectangle(
            [origin, (origin[0] + width, origin[1] + height)],
            fill=color,
            width=thickness,
        )
        return img

    @staticmethod
    def draw_text(
        img: PIL.Image.Image,
        origin: tuple,
        text: str,
        font: PIL.ImageFont = None,
        color: tuple = (255, 255, 255),
    ) -> PIL.Image.Image:
        """
        Draw text on the image

        Parameters
        ----------
        img : PIL.Image.Image
            Image
        origin : tuple
            Origin of the text (x, y)
        text : str
            Text to draw
        font : PIL.ImageFont
            Font to use
        color : tuple, optional
            Color of the text (RGB), by default (255, 255, 255)

        Returns
        -------
        PIL.Image.Image
        """
        draw = PIL.ImageDraw.Draw(img)

        if font is None:
            font = PIL.ImageFont.truetype("fonts/Gidole-Regular.ttf", size=20)

        draw.text(
            origin,
            text,
            font=font,
            fill=color,
        )

        return img

    @staticmethod
    def draw_bounding_box(
        img: PIL.Image.Image, rectangle: tuple, color: tuple, thickness: int = 3
    ) -> PIL.Image.Image:
        """

        Draw a bounding box on the image

        Parameters
        ----------
        img : PIL.Image.Image
            Image
        rectangle : tuple
            Rectangle to draw ( (xmin, ymin), (xmax, ymax) )
        color : tuple
            Color of the rectangle (BGR)
        thickness : int, optional
            Thickness of the rectangle, by default 2

        Returns
        -------
        PIL.Image.Image
            Image with the bounding box drawn
        """

        rectangle = rectangle[0:2]

        draw = PIL.ImageDraw.Draw(img)
        rectangle = [tuple(x) for x in rectangle]
        # draw.rectangle(rectangle, outline=color, width=thickness)
        draw.rounded_rectangle(rectangle, radius=7, outline=color, width=thickness)

        return img

    @staticmethod
    def draw_detection(
        detection: norfair.Detection,
        img: PIL.Image.Image,
        confidence: bool = False,
        id: bool = False,
    ) -> PIL.Image.Image:
        """
        Draw a bounding box on the image from a norfair.Detection

        Parameters
        ----------
        detection : norfair.Detection
            Detection to draw
        img : PIL.Image.Image
            Image
        confidence : bool, optional
            Whether to draw confidence in the box, by default False
        id : bool, optional
            Whether to draw id in the box, by default False

        Returns
        -------
        PIL.Image.Image
            Image with the bounding box drawn
        """

        if detection is None:
            return img

        x1, y1 = detection.points[0]
        x2, y2 = detection.points[1]

        color = (0, 0, 0)
        if "color" in detection.data:
            color = detection.data["color"] + (255,)

        img = Draw.draw_bounding_box(img=img, rectangle=detection.points, color=color)

        if "label" in detection.data:
            label = detection.data["label"]
            img = Draw.draw_text(
                img=img,
                origin=(x1, y1 - 20),
                text=label,
                color=color,
            )

        if "id" in detection.data and id is True:
            id = detection.data["id"]
            img = Draw.draw_text(
                img=img,
                origin=(x2, y1 - 20),
                text=f"ID: {id}",
                color=color,
            )

        if confidence:
            img = Draw.draw_text(
                img=img,
                origin=(x1, y2),
                text=str(round(detection.data["p"], 2)),
                color=color,
            )

        return img

    @staticmethod
    def draw_pointer(
        detection: norfair.Detection, img: PIL.Image.Image, color: tuple = (0, 255, 0)
    ) -> PIL.Image.Image:
        """

        Draw a pointer on the image from a norfair.Detection bounding box

        Parameters
        ----------
        detection : norfair.Detection
            Detection to draw
        img : PIL.Image.Image
            Image
        color : tuple, optional
            Pointer color, by default (0, 255, 0)

        Returns
        -------
        PIL.Image.Image
            Image with the pointer drawn
        """
        if detection is None:
            return

        if color is None:
            color = (0, 0, 0)

        x1, y1 = detection.points[0]
        x2, y2 = detection.points[1]

        draw = PIL.ImageDraw.Draw(img)

        # (t_x1, t_y1)        (t_x2, t_y2)
        #   \                  /
        #    \                /
        #     \              /
        #      \            /
        #       \          /
        #        \        /
        #         \      /
        #          \    /
        #           \  /
        #       (t_x3, t_y3)

        width = 20
        height = 20
        vertical_space_from_bbox = 7

        t_x3 = 0.5 * x1 + 0.5 * x2
        t_x1 = t_x3 - width / 2
        t_x2 = t_x3 + width / 2

        t_y1 = y1 - vertical_space_from_bbox - height
        t_y2 = t_y1
        t_y3 = y1 - vertical_space_from_bbox

        draw.polygon(
            [
                (t_x1, t_y1),
                (t_x2, t_y2),
                (t_x3, t_y3),
            ],
            fill=color,
        )

        draw.line(
            [
                (t_x1, t_y1),
                (t_x2, t_y2),
                (t_x3, t_y3),
                (t_x1, t_y1),
            ],
            fill="black",
            width=2,
        )

        return img

    @staticmethod
    def rounded_rectangle(
        img: PIL.Image.Image, rectangle: tuple, color: tuple, radius: int = 15
    ) -> PIL.Image.Image:
        """
        Draw a rounded rectangle on the image

        Parameters
        ----------
        img : PIL.Image.Image
            Image
        rectangle : tuple
            Rectangle to draw ( (xmin, ymin), (xmax, ymax) )
        color : tuple
            Color of the rectangle (BGR)
        radius : int, optional
            Radius of the corners, by default 15

        Returns
        -------
        PIL.Image.Image
            Image with the rounded rectangle drawn
        """

        overlay = img.copy()
        draw = PIL.ImageDraw.Draw(overlay, "RGBA")
        draw.rounded_rectangle(rectangle, radius, fill=color)
        return overlay

    @staticmethod
    def half_rounded_rectangle(
        img: PIL.Image.Image,
        rectangle: tuple,
        color: tuple,
        radius: int = 15,
        left: bool = False,
    ) -> PIL.Image.Image:
        """

        Draw a half rounded rectangle on the image

        Parameters
        ----------
        img : PIL.Image.Image
            Image
        rectangle : tuple
            Rectangle to draw ( (xmin, ymin), (xmax, ymax) )
        color : tuple
            Color of the rectangle (BGR)
        radius : int, optional
            Radius of the rounded borders, by default 15
        left : bool, optional
            Whether the flat side is the left side, by default False

        Returns
        -------
        PIL.Image.Image
            Image with the half rounded rectangle drawn
        """
        overlay = img.copy()
        draw = PIL.ImageDraw.Draw(overlay, "RGBA")
        draw.rounded_rectangle(rectangle, radius, fill=color)

        height = rectangle[1][1] - rectangle[0][1]
        stop_width = 13

        if left:
            draw.rectangle(
                (
                    rectangle[0][0] + 0,
                    rectangle[1][1] - height,
                    rectangle[0][0] + stop_width,
                    rectangle[1][1],
                ),
                fill=color,
            )
        else:
            draw.rectangle(
                (
                    rectangle[1][0] - stop_width,
                    rectangle[1][1] - height,
                    rectangle[1][0],
                    rectangle[1][1],
                ),
                fill=color,
            )
        return overlay

    @staticmethod
    
    def text_in_middle_rectangle(
            img: PIL.Image.Image,
            origin: tuple,
            width: int,
            height: int,
            text: str,
            font: PIL.ImageFont = None,
            color=(255, 255, 255),
        ) -> PIL.Image.Image:
        """
        Draw text in the middle of a rectangle.
        """
        draw = PIL.ImageDraw.Draw(img)

        if font is None:
            font = PIL.ImageFont.truetype("fonts/Gidole-Regular.ttf", size=24)

        # Use our helper function instead of draw.textsize
        w, h = get_text_size(draw, text, font)
        text_origin = (
            origin[0] + width / 2 - w / 2,
            origin[1] + height / 2 - h / 2,
        )

        draw.text(text_origin, text, font=font, fill=color)
        return img

    @staticmethod
    def add_alpha(img: PIL.Image.Image, alpha: int = 100) -> PIL.Image.Image:
        """
        Add an alpha channel to an image

        Parameters
        ----------
        img : PIL.Image.Image
            Image
        alpha : int, optional
            Alpha value, by default 100

        Returns
        -------
        PIL.Image.Image
            Image with alpha channel
        """
        data = img.getdata()
        newData = []
        for old_pixel in data:

            # Don't change transparency of transparent pixels
            if old_pixel[3] != 0:
                pixel_with_alpha = old_pixel[:3] + (alpha,)
                newData.append(pixel_with_alpha)
            else:
                newData.append(old_pixel)

        img.putdata(newData)
        return img


class PathPoint:
    def __init__(
        self, id: int, center: tuple, color: tuple = (255, 255, 255), alpha: float = 1
    ):
        """
        Path point

        Parameters
        ----------
        id : int
            Id of the point
        center : tuple
            Center of the point (x, y)
        color : tuple, optional
            Color of the point, by default (255, 255, 255)
        alpha : float, optional
            Alpha value of the point, by default 1
        """
        self.id = id
        self.center = center
        self.color = color
        self.alpha = alpha

    def __str__(self) -> str:
        return str(self.id)

    @property
    def color_with_alpha(self) -> tuple:
        return (self.color[0], self.color[1], self.color[2], int(self.alpha * 255))

    @staticmethod
    def get_center_from_bounding_box(bounding_box: np.ndarray) -> tuple:
        """
        Get the center of a bounding box

        Parameters
        ----------
        bounding_box : np.ndarray
            Bounding box [[xmin, ymin], [xmax, ymax]]

        Returns
        -------
        tuple
            Center of the bounding box (x, y)
        """
        return (
            int((bounding_box[0][0] + bounding_box[1][0]) / 2),
            int((bounding_box[0][1] + bounding_box[1][1]) / 2),
        )

    @staticmethod
    def from_abs_bbox(
        id: int,
        abs_point: np.ndarray,
        coord_transformations,
        color: tuple = None,
        alpha: float = None,
    ) -> "PathPoint":
        """
        Create a PathPoint from an absolute bounding box.
        It converts the absolute bounding box to a relative one and then to a center point

        Parameters
        ----------
        id : int
            Id of the point
        abs_point : np.ndarray
            Absolute bounding box
        coord_transformations : "CoordTransformations"
            Coordinate transformations
        color : tuple, optional
            Color of the point, by default None
        alpha : float, optional
            Alpha value of the point, by default None

        Returns
        -------
        PathPoint
            PathPoint
        """

        rel_point = coord_transformations.abs_to_rel(abs_point)
        center = PathPoint.get_center_from_bounding_box(rel_point)

        return PathPoint(id=id, center=center, color=color, alpha=alpha)


class AbsolutePath:
    def __init__(self) -> None:
        self.past_points = []
        self.color_by_index = {}

    def center(self, points: np.ndarray) -> tuple:
        """
        Get the center of a Norfair Bounding Box Detection point

        Parameters
        ----------
        points : np.ndarray
            Norfair Bounding Box Detection point

        Returns
        -------
        tuple
            Center of the point (x, y)
        """
        return (
            int((points[0][0] + points[1][0]) / 2),
            int((points[0][1] + points[1][1]) / 2),
        )

    @property
    def path_length(self) -> int:
        return len(self.past_points)

    def draw_path_slow(
        self,
        img: PIL.Image.Image,
        path: List[PathPoint],
        thickness: int = 4,
    ) -> PIL.Image.Image:
        """
        Draw a path with alpha

        Parameters
        ----------
        img : PIL.Image.Image
            Image
        path : List[PathPoint]
            List of points to draw
        thickness : int, optional
            Thickness of the path, by default 4

        Returns
        -------
        PIL.Image.Image
            Image with the path drawn
        """
        draw = PIL.ImageDraw.Draw(img, "RGBA")

        for i in range(len(path) - 1):
            draw.line(
                [path[i].center, path[i + 1].center],
                fill=path[i].color_with_alpha,
                width=thickness,
            )
        return img

    def draw_arrow_head(
        self,
        img: PIL.Image.Image,
        start: tuple,
        end: tuple,
        color: tuple = (255, 255, 255),
        length: int = 10,
        height: int = 6,
        thickness: int = 4,
        alpha: int = 255,
    ) -> PIL.Image.Image:

        # https://stackoverflow.com/questions/43527894/drawing-arrowheads-which-follow-the-direction-of-the-line-in-pygame
        draw = PIL.ImageDraw.Draw(img, "RGBA")

        dX = end[0] - start[0]
        dY = end[1] - start[1]

        # vector length
        Len = sqrt(dX * dX + dY * dY)  # use Hypot if available

        if Len == 0:
            return img

        # normalized direction vector components
        udX = dX / Len
        udY = dY / Len

        # perpendicular vector
        perpX = -udY
        perpY = udX

        # points forming arrowhead
        # with length L and half-width H
        arrowend = end

        leftX = end[0] - length * udX + height * perpX
        leftY = end[1] - length * udY + height * perpY

        rightX = end[0] - length * udX - height * perpX
        rightY = end[1] - length * udY - height * perpY

        if len(color) <= 3:
            color += (alpha,)

        draw.line(
            [(leftX, leftY), arrowend],
            fill=color,
            width=thickness,
        )

        draw.line(
            [(rightX, rightY), arrowend],
            fill=color,
            width=thickness,
        )

        return img

    def draw_path_arrows(
        self,
        img: PIL.Image.Image,
        path: List[PathPoint],
        thickness: int = 4,
        frame_frequency: int = 30,
    ) -> PIL.Image.Image:
        """
        Draw a path with arrows every 30 points

        Parameters
        ----------
        img : PIL.Image.Image
            Image
        path : List[PathPoint]
            Path
        thickness : int, optional
            Thickness of the path, by default 4

        Returns
        -------
        PIL.Image.Image
            Image with the arrows drawn
        """

        for i, point in enumerate(path):

            if i < 4 or i % frame_frequency:
                continue

            end = path[i]
            start = path[i - 4]

            img = self.draw_arrow_head(
                img=img,
                start=start.center,
                end=end.center,
                color=start.color_with_alpha,
                thickness=thickness,
            )

        return img

    def draw_path_fast(
        self,
        img: PIL.Image.Image,
        path: List[PathPoint],
        color: tuple,
        width: int = 2,
        alpha: int = 255,
    ) -> PIL.Image.Image:
        """
        Draw a path without alpha (faster)

        Parameters
        ----------
        img : PIL.Image.Image
            Image
        path : List[PathPoint]
            Path
        color : tuple
            Color of the path
        with : int
            Width of the line
        alpha : int
            Color alpha (0-255)

        Returns
        -------
        PIL.Image.Image
            Image with the path drawn
        """
        draw = PIL.ImageDraw.Draw(img, "RGBA")

        path_list = [point.center for point in path]

        color += (alpha,)

        draw.line(
            path_list,
            fill=color,
            width=width,
        )

        return img

    def draw_arrow(
        self,
        img: PIL.Image.Image,
        points: List[PathPoint],
        color: tuple,
        width: int,
        alpha: int = 255,
    ) -> PIL.Image.Image:
        """Draw arrow between two points

        Parameters
        ----------
        img : PIL.Image.Image
            image to draw
        points : List[PathPoint]
            start and end points
        color : tuple
            color of the arrow
        width : int
            width of the arrow
        alpha : int, optional
            color alpha (0-255), by default 255

        Returns
        -------
        PIL.Image.Image
            Image with the arrow
        """

        img = self.draw_path_fast(
            img=img, path=points, color=color, width=width, alpha=alpha
        )
        img = self.draw_arrow_head(
            img=img,
            start=points[0].center,
            end=points[1].center,
            color=color,
            length=30,
            height=15,
            alpha=alpha,
        )

        return img

    def add_new_point(
        self, detection: norfair.Detection, color: tuple = (255, 255, 255)
    ) -> None:
        """
        Add a new point to the path

        Parameters
        ----------
        detection : norfair.Detection
            Detection
        color : tuple, optional
            Color of the point, by default (255, 255, 255)
        """

        if detection is None:
            return

        self.past_points.append(detection.absolute_points)

        self.color_by_index[len(self.past_points) - 1] = color

    def filter_points_outside_frame(
        self, path: List[PathPoint], width: int, height: int, margin: int = 0
    ) -> List[PathPoint]:
        """
        Filter points outside the frame with a margin

        Parameters
        ----------
        path : List[PathPoint]
            List of points
        width : int
            Width of the frame
        height : int
            Height of the frame
        margin : int, optional
            Margin, by default 0

        Returns
        -------
        List[PathPoint]
            List of points inside the frame with the margin
        """

        return [
            point
            for point in path
            if point.center[0] > 0 - margin
            and point.center[1] > 0 - margin
            and point.center[0] < width + margin
            and point.center[1] < height + margin
        ]

    def draw(
        self,
        img: PIL.Image.Image,
        detection: norfair.Detection,
        coord_transformations,
        color: tuple = (255, 255, 255),
    ) -> PIL.Image.Image:
        """
        Draw the path

        Parameters
        ----------
        img : PIL.Image.Image
            Image
        detection : norfair.Detection
            Detection
        coord_transformations : _type_
            Coordinate transformations
        color : tuple, optional
            Color of the path, by default (255, 255, 255)

        Returns
        -------
        PIL.Image.Image
            Image with the path drawn
        """

        self.add_new_point(detection=detection, color=color)

        if len(self.past_points) < 2:
            return img

        path = [
            PathPoint.from_abs_bbox(
                id=i,
                abs_point=point,
                coord_transformations=coord_transformations,
                alpha=i / (1.2 * self.path_length),
                color=self.color_by_index[i],
            )
            for i, point in enumerate(self.past_points)
        ]

        path_filtered = self.filter_points_outside_frame(
            path=path,
            width=img.size[0],
            height=img.size[0],
            margin=250,
        )

        img = self.draw_path_slow(img=img, path=path_filtered)
        img = self.draw_path_arrows(img=img, path=path)

        return img


class Team:
    def __init__(
        self,
        name: str,
        color: tuple = (0, 0, 0),
        abbreviation: str = "NNN",
        board_color: tuple = None,
        text_color: tuple = (0, 0, 0),
    ):
        """
        Initialize Team

        Parameters
        ----------
        name : str
            Team name
        color : tuple, optional
            Team color, by default (0, 0, 0)
        abbreviation : str, optional
            Team abbreviation, by default "NNN"

        Raises
        ------
        ValueError
            If abbreviation is not 3 characters long or not uppercase
        """
        self.name = name
        self.possession = 0
        self.passes = []
        self.color = color
        self.abbreviation = abbreviation
        self.text_color = text_color

        if board_color is None:
            self.board_color = color
        else:
            self.board_color = board_color

        if len(abbreviation) != 3 or not abbreviation.isupper():
            raise ValueError("abbreviation must be length 3 and uppercase")

    def get_percentage_possession(self, duration: int) -> float:
        """
        Return team possession in percentage

        Parameters
        ----------
        duration : int
            Match duration in frames

        Returns
        -------
        float
            Team possession in percentage
        """
        if duration == 0:
            return 0
        return round(self.possession / duration, 2)

    def get_time_possession(self, fps: int) -> str:
        """
        Return team possession in time format

        Parameters
        ----------
        fps : int
            Frames per second

        Returns
        -------
        str
            Team possession in time format (mm:ss)
        """

        seconds = round(self.possession / fps)
        minutes = seconds // 60
        seconds = seconds % 60

        # express seconds in 2 digits
        seconds = str(seconds)
        if len(seconds) == 1:
            seconds = "0" + seconds

        # express minutes in 2 digits
        minutes = str(minutes)
        if len(minutes) == 1:
            minutes = "0" + minutes

        return f"{minutes}:{seconds}"

    def __str__(self):
        return self.name

    def __eq__(self, other: "Team") -> bool:
        if isinstance(self, Team) == False or isinstance(other, Team) == False:
            return False

        return self.name == other.name

    @staticmethod
    def from_name(teams: List["Team"], name: str) -> "Team":
        """
        Return team object from name

        Parameters
        ----------
        teams : List[Team]
            List of Team objects
        name : str
            Team name

        Returns
        -------
        Team
            Team object
        """
        for team in teams:
            if team.name == name:
                return team
        return None


class Player:
    def __init__(self, detection: Detection):
        """

        Initialize Player

        Parameters
        ----------
        detection : Detection
            Detection containing the player
        """
        self.detection = detection

        self.team = None

        if detection:
            if "team" in detection.data:
                self.team = detection.data["team"]

    def get_left_foot(self, points: np.array):
        x1, y1 = points[0]
        x2, y2 = points[1]

        return [x1, y2]

    def get_right_foot(self, points: np.array):
        return points[1]

    @property
    def left_foot(self):
        points = self.detection.points
        left_foot = self.get_left_foot(points)

        return left_foot

    @property
    def right_foot(self):
        points = self.detection.points
        right_foot = self.get_right_foot(points)

        return right_foot

    @property
    def left_foot_abs(self):
        points = self.detection.absolute_points
        left_foot_abs = self.get_left_foot(points)

        return left_foot_abs

    @property
    def right_foot_abs(self):
        points = self.detection.absolute_points
        right_foot_abs = self.get_right_foot(points)

        return right_foot_abs

    @property
    def feet(self) -> np.ndarray:
        return np.array([self.left_foot, self.right_foot])

    def distance_to_ball(self, ball: Ball) -> float:
        """
        Returns the distance between the player closest foot and the ball

        Parameters
        ----------
        ball : Ball
            Ball object

        Returns
        -------
        float
            Distance between the player closest foot and the ball
        """

        if self.detection is None or ball.center is None:
            return None

        left_foot_distance = np.linalg.norm(ball.center - self.left_foot)
        right_foot_distance = np.linalg.norm(ball.center - self.right_foot)

        return min(left_foot_distance, right_foot_distance)

    def closest_foot_to_ball(self, ball: Ball) -> np.ndarray:
        """

        Returns the closest foot to the ball

        Parameters
        ----------
        ball : Ball
            Ball object

        Returns
        -------
        np.ndarray
            Closest foot to the ball (x, y)
        """

        if self.detection is None or ball.center is None:
            return None

        left_foot_distance = np.linalg.norm(ball.center - self.left_foot)
        right_foot_distance = np.linalg.norm(ball.center - self.right_foot)

        if left_foot_distance < right_foot_distance:
            return self.left_foot

        return self.right_foot

    def closest_foot_to_ball_abs(self, ball: Ball) -> np.ndarray:
        """

        Returns the closest foot to the ball

        Parameters
        ----------
        ball : Ball
            Ball object

        Returns
        -------
        np.ndarray
            Closest foot to the ball (x, y)
        """

        if self.detection is None or ball.center_abs is None:
            return None

        left_foot_distance = np.linalg.norm(ball.center_abs - self.left_foot_abs)
        right_foot_distance = np.linalg.norm(ball.center_abs - self.right_foot_abs)

        if left_foot_distance < right_foot_distance:
            return self.left_foot_abs

        return self.right_foot_abs

    def draw(
        self, frame: PIL.Image.Image, confidence: bool = False, id: bool = False
    ) -> PIL.Image.Image:
        """
        Draw the player on the frame

        Parameters
        ----------
        frame : PIL.Image.Image
            Frame to draw on
        confidence : bool, optional
            Whether to draw confidence text in bounding box, by default False
        id : bool, optional
            Whether to draw id text in bounding box, by default False

        Returns
        -------
        PIL.Image.Image
            Frame with player drawn
        """
        if self.detection is None:
            return frame

        if self.team is not None:
            self.detection.data["color"] = self.team.color

        return Draw.draw_detection(self.detection, frame, confidence=confidence, id=id)

    def draw_pointer(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw a pointer above the player

        Parameters
        ----------
        frame : np.ndarray
            Frame to draw on

        Returns
        -------
        np.ndarray
            Frame with pointer drawn
        """
        if self.detection is None:
            return frame

        color = None

        if self.team:
            color = self.team.color

        return Draw.draw_pointer(detection=self.detection, img=frame, color=color)

    def __str__(self):
        return f"Player: {self.feet}, team: {self.team}"

    def __eq__(self, other: "Player") -> bool:
        if isinstance(self, Player) == False or isinstance(other, Player) == False:
            return False

        self_id = self.detection.data["id"]
        other_id = other.detection.data["id"]

        return self_id == other_id

    @staticmethod
    def have_same_id(player1: "Player", player2: "Player") -> bool:
        """
        Check if player1 and player2 have the same ids

        Parameters
        ----------
        player1 : Player
            One player
        player2 : Player
            Another player

        Returns
        -------
        bool
            True if they have the same id
        """
        if not player1 or not player2:
            return False
        if "id" not in player1.detection.data or "id" not in player2.detection.data:
            return False
        return player1 == player2

    @staticmethod
    def draw_players(
        players: List["Player"],
        frame: PIL.Image.Image,
        confidence: bool = False,
        id: bool = False,
    ) -> PIL.Image.Image:
        """
        Draw all players on the frame

        Parameters
        ----------
        players : List[Player]
            List of Player objects
        frame : PIL.Image.Image
            Frame to draw on
        confidence : bool, optional
            Whether to draw confidence text in bounding box, by default False
        id : bool, optional
            Whether to draw id text in bounding box, by default False

        Returns
        -------
        PIL.Image.Image
            Frame with players drawn
        """
        for player in players:
            frame = player.draw(frame, confidence=confidence, id=id)

        return frame

    @staticmethod
    def from_detections(
        detections: List[Detection], teams=List[Team]
    ) -> List["Player"]:
        """
        Create a list of Player objects from a list of detections and a list of teams.

        It reads the classification string field of the detection, converts it to a
        Team object and assigns it to the player.

        Parameters
        ----------
        detections : List[Detection]
            List of detections
        teams : List[Team], optional
            List of teams, by default List[Team]

        Returns
        -------
        List[Player]
            List of Player objects
        """
        players = []

        for detection in detections:
            if detection is None:
                continue

            if "classification" in detection.data:
                team_name = detection.data["classification"]
                team = Team.from_name(teams=teams, name=team_name)
                detection.data["team"] = team

            player = Player(detection=detection)

            players.append(player)

        return players

class Match:
    def __init__(self, home: Team, away: Team, fps: int = 30):
        """

        Initialize Match

        Parameters
        ----------
        home : Team
            Home team
        away : Team
            Away team
        fps : int, optional
            Fps, by default 30
        """
        self.duration = 0
        self.home = home
        self.away = away
        self.team_possession = self.home
        self.current_team = self.home
        self.possession_counter = 0
        self.closest_player = None
        self.ball = None
        # Amount of consecutive frames new team has to have the ball in order to change possession
        self.possesion_counter_threshold = 20
        # Distance in pixels from player to ball in order to consider a player has the ball
        self.ball_distance_threshold = 45
        self.fps = fps
        # Pass detection
        self.pass_event = PassEvent()

    def update(self, players: List[Player], ball: Ball):
        """

        Update match possession and closest player

        Parameters
        ----------
        players : List[Player]
            List of players
        ball : Ball
            Ball
        """

        self.update_possession()

        if ball is None or ball.detection is None:
            self.closest_player = None
            return

        self.ball = ball

        closest_player = min(players, key=lambda player: player.distance_to_ball(ball))

        self.closest_player = closest_player

        ball_distance = closest_player.distance_to_ball(ball)

        if ball_distance > self.ball_distance_threshold:
            self.closest_player = None
            return

        # Reset counter if team changed
        if closest_player.team != self.current_team:
            self.possession_counter = 0
            self.current_team = closest_player.team

        self.possession_counter += 1

        if (
            self.possession_counter >= self.possesion_counter_threshold
            and closest_player.team is not None
        ):
            self.change_team(self.current_team)

        # Pass detection
        self.pass_event.update(closest_player=closest_player, ball=ball)

        self.pass_event.process_pass()

    def change_team(self, team: Team):
        """

        Change team possession

        Parameters
        ----------
        team : Team, optional
            New team in possession
        """
        self.team_possession = team

    def update_possession(self):
        """
        Updates match duration and possession counter of team in possession
        """
        if self.team_possession is None:
            return

        self.team_possession.possession += 1
        self.duration += 1

    @property
    def home_possession_str(self) -> str:
        return f"{self.home.abbreviation}: {self.home.get_time_possession(self.fps)}"

    @property
    def away_possession_str(self) -> str:
        return f"{self.away.abbreviation}: {self.away.get_time_possession(self.fps)}"

    def __str__(self) -> str:
        return f"{self.home_possession_str} | {self.away_possession_str}"

    @property
    def time_possessions(self) -> str:
        return f"{self.home.name}: {self.home.get_time_possession(self.fps)} | {self.away.name}: {self.away.get_time_possession(self.fps)}"

    @property
    def passes(self) -> List["Pass"]:
        home_passes = self.home.passes
        away_passes = self.away.passes

        return home_passes + away_passes

    def possession_bar(self, frame: PIL.Image.Image, origin: tuple) -> PIL.Image.Image:
        """
        Draw possession bar

        Parameters
        ----------
        frame : PIL.Image.Image
            Frame
        origin : tuple
            Origin (x, y)

        Returns
        -------
        PIL.Image.Image
            Frame with possession bar
        """

        bar_x = origin[0]
        bar_y = origin[1]
        bar_height = 29
        bar_width = 310

        ratio = self.home.get_percentage_possession(self.duration)

        # Protect against too small rectangles
        if ratio < 0.07:
            ratio = 0.07

        if ratio > 0.93:
            ratio = 0.93

        left_rectangle = (
            origin,
            [int(bar_x + ratio * bar_width), int(bar_y + bar_height)],
        )

        right_rectangle = (
            [int(bar_x + ratio * bar_width), bar_y],
            [int(bar_x + bar_width), int(bar_y + bar_height)],
        )

        left_color = self.home.board_color
        right_color = self.away.board_color

        frame = self.draw_counter_rectangle(
            frame=frame,
            ratio=ratio,
            left_rectangle=left_rectangle,
            left_color=left_color,
            right_rectangle=right_rectangle,
            right_color=right_color,
        )

        # Draw home text
        if ratio > 0.15:
            home_text = (
                f"{int(self.home.get_percentage_possession(self.duration) * 100)}%"
            )

            frame = Draw.text_in_middle_rectangle(
                img=frame,
                origin=left_rectangle[0],
                width=left_rectangle[1][0] - left_rectangle[0][0],
                height=left_rectangle[1][1] - left_rectangle[0][1],
                text=home_text,
                color=self.home.text_color,
            )

        # Draw away text
        if ratio < 0.85:
            away_text = (
                f"{int(self.away.get_percentage_possession(self.duration) * 100)}%"
            )

            frame = Draw.text_in_middle_rectangle(
                img=frame,
                origin=right_rectangle[0],
                width=right_rectangle[1][0] - right_rectangle[0][0],
                height=right_rectangle[1][1] - right_rectangle[0][1],
                text=away_text,
                color=self.away.text_color,
            )

        return frame

    def draw_counter_rectangle(
        self,
        frame: PIL.Image.Image,
        ratio: float,
        left_rectangle: tuple,
        left_color: tuple,
        right_rectangle: tuple,
        right_color: tuple,
    ) -> PIL.Image.Image:
        """Draw counter rectangle for both teams

        Parameters
        ----------
        frame : PIL.Image.Image
            Video frame
        ratio : float
            counter proportion
        left_rectangle : tuple
            rectangle for the left team in counter
        left_color : tuple
            color for the left team in counter
        right_rectangle : tuple
            rectangle for the right team in counter
        right_color : tuple
            color for the right team in counter

        Returns
        -------
        PIL.Image.Image
            Drawed video frame
        """

        # Draw first one rectangle or another in orther to make the
        # rectangle bigger for better rounded corners

        if ratio < 0.15:
            left_rectangle[1][0] += 20

            frame = Draw.half_rounded_rectangle(
                frame,
                rectangle=left_rectangle,
                color=left_color,
                radius=15,
            )

            frame = Draw.half_rounded_rectangle(
                frame,
                rectangle=right_rectangle,
                color=right_color,
                left=True,
                radius=15,
            )
        else:
            right_rectangle[0][0] -= 20

            frame = Draw.half_rounded_rectangle(
                frame,
                rectangle=right_rectangle,
                color=right_color,
                left=True,
                radius=15,
            )

            frame = Draw.half_rounded_rectangle(
                frame,
                rectangle=left_rectangle,
                color=left_color,
                radius=15,
            )

        return frame

    def passes_bar(self, frame: PIL.Image.Image, origin: tuple) -> PIL.Image.Image:
        """
        Draw passes bar

        Parameters
        ----------
        frame : PIL.Image.Image
            Frame
        origin : tuple
            Origin (x, y)

        Returns
        -------
        PIL.Image.Image
            Frame with passes bar
        """

        bar_x = origin[0]
        bar_y = origin[1]
        bar_height = 29
        bar_width = 310

        home_passes = len(self.home.passes)
        away_passes = len(self.away.passes)
        total_passes = home_passes + away_passes

        if total_passes == 0:
            home_ratio = 0
            away_ratio = 0
        else:
            home_ratio = home_passes / total_passes
            away_ratio = away_passes / total_passes

        ratio = home_ratio

        # Protect against too small rectangles
        if ratio < 0.07:
            ratio = 0.07

        if ratio > 0.93:
            ratio = 0.93

        left_rectangle = (
            origin,
            [int(bar_x + ratio * bar_width), int(bar_y + bar_height)],
        )

        right_rectangle = (
            [int(bar_x + ratio * bar_width), bar_y],
            [int(bar_x + bar_width), int(bar_y + bar_height)],
        )

        left_color = self.home.board_color
        right_color = self.away.board_color

        # Draw first one rectangle or another in orther to make the
        # rectangle bigger for better rounded corners
        frame = self.draw_counter_rectangle(
            frame=frame,
            ratio=ratio,
            left_rectangle=left_rectangle,
            left_color=left_color,
            right_rectangle=right_rectangle,
            right_color=right_color,
        )

        # Draw home text
        if ratio > 0.15:
            home_text = f"{int(home_ratio * 100)}%"

            frame = Draw.text_in_middle_rectangle(
                img=frame,
                origin=left_rectangle[0],
                width=left_rectangle[1][0] - left_rectangle[0][0],
                height=left_rectangle[1][1] - left_rectangle[0][1],
                text=home_text,
                color=self.home.text_color,
            )

        # Draw away text
        if ratio < 0.85:
            away_text = f"{int(away_ratio * 100)}%"

            frame = Draw.text_in_middle_rectangle(
                img=frame,
                origin=right_rectangle[0],
                width=right_rectangle[1][0] - right_rectangle[0][0],
                height=right_rectangle[1][1] - right_rectangle[0][1],
                text=away_text,
                color=self.away.text_color,
            )

        return frame

    def get_possession_background(
        self,
    ) -> PIL.Image.Image:
        """
        Get possession counter background

        Returns
        -------
        PIL.Image.Image
            Counter background
        """

        counter = PIL.Image.open("./images/possession_board.png").convert("RGBA")
        counter = Draw.add_alpha(counter, 210)
        counter = np.array(counter)
        red, green, blue, alpha = counter.T
        counter = np.array([blue, green, red, alpha])
        counter = counter.transpose()
        counter = PIL.Image.fromarray(counter)
        counter = counter.resize((int(315 * 1.2), int(210 * 1.2)))
        return counter

    def get_passes_background(self) -> PIL.Image.Image:
        """
        Get passes counter background

        Returns
        -------
        PIL.Image.Image
            Counter background
        """

        counter = PIL.Image.open("./images/passes_board.png").convert("RGBA")
        counter = Draw.add_alpha(counter, 210)
        counter = np.array(counter)
        red, green, blue, alpha = counter.T
        counter = np.array([blue, green, red, alpha])
        counter = counter.transpose()
        counter = PIL.Image.fromarray(counter)
        counter = counter.resize((int(315 * 1.2), int(210 * 1.2)))
        return counter

    def draw_counter_background(
        self,
        frame: PIL.Image.Image,
        origin: tuple,
        counter_background: PIL.Image.Image,
    ) -> PIL.Image.Image:
        """
        Draw counter background

        Parameters
        ----------
        frame : PIL.Image.Image
            Frame
        origin : tuple
            Origin (x, y)
        counter_background : PIL.Image.Image
            Counter background

        Returns
        -------
        PIL.Image.Image
            Frame with counter background
        """
        frame.paste(counter_background, origin, counter_background)
        return frame

    def draw_counter(
        self,
        frame: PIL.Image.Image,
        text: str,
        counter_text: str,
        origin: tuple,
        color: tuple,
        text_color: tuple,
        height: int = 27,
        width: int = 120,
    ) -> PIL.Image.Image:
        """
        Draw counter

        Parameters
        ----------
        frame : PIL.Image.Image
            Frame
        text : str
            Text in left-side of counter
        counter_text : str
            Text in right-side of counter
        origin : tuple
            Origin (x, y)
        color : tuple
            Color
        text_color : tuple
            Color of text
        height : int, optional
            Height, by default 27
        width : int, optional
            Width, by default 120

        Returns
        -------
        PIL.Image.Image
            Frame with counter
        """

        team_begin = origin
        team_width_ratio = 0.417
        team_width = width * team_width_ratio

        team_rectangle = (
            team_begin,
            (team_begin[0] + team_width, team_begin[1] + height),
        )

        time_begin = (origin[0] + team_width, origin[1])
        time_width = width * (1 - team_width_ratio)

        time_rectangle = (
            time_begin,
            (time_begin[0] + time_width, time_begin[1] + height),
        )

        frame = Draw.half_rounded_rectangle(
            img=frame,
            rectangle=team_rectangle,
            color=color,
            radius=20,
        )

        frame = Draw.half_rounded_rectangle(
            img=frame,
            rectangle=time_rectangle,
            color=(239, 234, 229),
            radius=20,
            left=True,
        )

        frame = Draw.text_in_middle_rectangle(
            img=frame,
            origin=team_rectangle[0],
            height=height,
            width=team_width,
            text=text,
            color=text_color,
        )

        frame = Draw.text_in_middle_rectangle(
            img=frame,
            origin=time_rectangle[0],
            height=height,
            width=time_width,
            text=counter_text,
            color="black",
        )

        return frame

    def draw_debug(self, frame: PIL.Image.Image) -> PIL.Image.Image:
        """Draw line from closest player feet to ball

        Parameters
        ----------
        frame : PIL.Image.Image
            Video frame

        Returns
        -------
        PIL.Image.Image
            Drawed video frame
        """
        if self.closest_player and self.ball:
            closest_foot = self.closest_player.closest_foot_to_ball(self.ball)

            color = (0, 0, 0)
            # Change line color if its greater than threshold
            distance = self.closest_player.distance_to_ball(self.ball)
            if distance > self.ball_distance_threshold:
                color = (255, 255, 255)

            draw = PIL.ImageDraw.Draw(frame)
            draw.line(
                [
                    tuple(closest_foot),
                    tuple(self.ball.center),
                ],
                fill=color,
                width=2,
            )

    def draw_possession_counter(
        self,
        frame: PIL.Image.Image,
        counter_background: PIL.Image.Image,
        debug: bool = False,
    ) -> PIL.Image.Image:
        """

        Draw elements of the possession in frame

        Parameters
        ----------
        frame : PIL.Image.Image
            Frame
        counter_background : PIL.Image.Image
            Counter background
        debug : bool, optional
            Whether to draw extra debug information, by default False

        Returns
        -------
        PIL.Image.Image
            Frame with elements of the match
        """

        # get width of PIL.Image
        frame_width = frame.size[0]
        counter_origin = (frame_width - 540, 40)

        frame = self.draw_counter_background(
            frame,
            origin=counter_origin,
            counter_background=counter_background,
        )

        frame = self.draw_counter(
            frame,
            origin=(counter_origin[0] + 35, counter_origin[1] + 130),
            text=self.home.abbreviation,
            counter_text=self.home.get_time_possession(self.fps),
            color=self.home.board_color,
            text_color=self.home.text_color,
            height=31,
            width=150,
        )
        frame = self.draw_counter(
            frame,
            origin=(counter_origin[0] + 35 + 150 + 10, counter_origin[1] + 130),
            text=self.away.abbreviation,
            counter_text=self.away.get_time_possession(self.fps),
            color=self.away.board_color,
            text_color=self.away.text_color,
            height=31,
            width=150,
        )
        frame = self.possession_bar(
            frame, origin=(counter_origin[0] + 35, counter_origin[1] + 195)
        )

        if self.closest_player:
            frame = self.closest_player.draw_pointer(frame)

        if debug:
            frame = self.draw_debug(frame=frame)

        return frame

    def draw_passes_counter(
        self,
        frame: PIL.Image.Image,
        counter_background: PIL.Image.Image,
        debug: bool = False,
    ) -> PIL.Image.Image:
        """

        Draw elements of the passes in frame

        Parameters
        ----------
        frame : PIL.Image.Image
            Frame
        counter_background : PIL.Image.Image
            Counter background
        debug : bool, optional
            Whether to draw extra debug information, by default False

        Returns
        -------
        PIL.Image.Image
            Frame with elements of the match
        """

        # get width of PIL.Image
        frame_width = frame.size[0]
        counter_origin = (frame_width - 540, 40)

        frame = self.draw_counter_background(
            frame,
            origin=counter_origin,
            counter_background=counter_background,
        )

        frame = self.draw_counter(
            frame,
            origin=(counter_origin[0] + 35, counter_origin[1] + 130),
            text=self.home.abbreviation,
            counter_text=str(len(self.home.passes)),
            color=self.home.board_color,
            text_color=self.home.text_color,
            height=31,
            width=150,
        )
        frame = self.draw_counter(
            frame,
            origin=(counter_origin[0] + 35 + 150 + 10, counter_origin[1] + 130),
            text=self.away.abbreviation,
            counter_text=str(len(self.away.passes)),
            color=self.away.board_color,
            text_color=self.away.text_color,
            height=31,
            width=150,
        )
        frame = self.passes_bar(
            frame, origin=(counter_origin[0] + 35, counter_origin[1] + 195)
        )

        if self.closest_player:
            frame = self.closest_player.draw_pointer(frame)

        if debug:
            frame = self.draw_debug(frame=frame)

        return frame


from typing import Iterable, List

class Pass:
    def __init__(
        self, start_ball_bbox: np.ndarray, end_ball_bbox: np.ndarray, team: Team
    ) -> None:
        # Abs coordinates
        self.start_ball_bbox = start_ball_bbox
        self.end_ball_bbox = end_ball_bbox
        self.team = team
        self.draw_abs = AbsolutePath()

    def draw(
        self, img: PIL.Image.Image, coord_transformations: "CoordinatesTransformation"
    ) -> PIL.Image.Image:
        """Draw a pass

        Parameters
        ----------
        img : PIL.Image.Image
            Video frame
        coord_transformations : CoordinatesTransformation
            coordinates transformation

        Returns
        -------
        PIL.Image.Image
            frame with the new pass
        """
        rel_point_start = PathPoint.from_abs_bbox(
            id=0,
            abs_point=self.start_ball_bbox,
            coord_transformations=coord_transformations,
            color=self.team.color,
        )
        rel_point_end = PathPoint.from_abs_bbox(
            id=1,
            abs_point=self.end_ball_bbox,
            coord_transformations=coord_transformations,
            color=self.team.color,
        )

        new_pass = [rel_point_start, rel_point_end]

        pass_filtered = self.draw_abs.filter_points_outside_frame(
            path=new_pass,
            width=img.size[0],
            height=img.size[0],
            margin=3000,
        )

        if len(pass_filtered) == 2:
            img = self.draw_abs.draw_arrow(
                img=img, points=pass_filtered, color=self.team.color, width=6, alpha=150
            )

        return img

    @staticmethod
    def draw_pass_list(
        img: PIL.Image.Image,
        passes: List["Pass"],
        coord_transformations: "CoordinatesTransformation",
    ) -> PIL.Image.Image:
        """Draw all the passes

        Parameters
        ----------
        img : PIL.Image.Image
            Video frame
        passes : List[Pass]
            Passes list to draw
        coord_transformations : CoordinatesTransformation
            Coordinate transformation for the current frame

        Returns
        -------
        PIL.Image.Image
            Drawed frame
        """
        for pass_ in passes:
            img = pass_.draw(img=img, coord_transformations=coord_transformations)

        return img

    def get_relative_coordinates(
        self, coord_transformations: "CoordinatesTransformation"
    ) -> tuple:
        """
        Print the relative coordinates of a pass

        Parameters
        ----------
        coord_transformations : CoordinatesTransformation
            Coordinates transformation

        Returns
        -------
        tuple
            (start, end) of the pass with relative coordinates
        """
        relative_start = coord_transformations.abs_to_rel(self.start_ball_bbox)
        relative_end = coord_transformations.abs_to_rel(self.end_ball_bbox)

        return (relative_start, relative_end)

    def get_center(self, points: np.array) -> tuple:
        """
        Returns the center of the points

        Parameters
        ----------
        points : np.array
            2D points

        Returns
        -------
        tuple
            (x, y) coordinates of the center
        """
        x1, y1 = points[0]
        x2, y2 = points[1]

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        return (center_x, center_y)

    def round_iterable(self, iterable: Iterable) -> Iterable:
        """
        Round all entries from one Iterable object

        Parameters
        ----------
        iterable : Iterable
            Iterable to round

        Returns
        -------
        Iterable
            Rounded Iterable
        """
        return [round(item) for item in iterable]

    def generate_output_pass(
        self, start: np.ndarray, end: np.ndarray, team_name: str
    ) -> str:
        """
        Generate a string with the pass information

        Parameters
        ----------
        start : np.ndarray
            The start point of the pass
        end : np.ndarray
            The end point of the pass
        team_name : str
            The team that did this pass

        Returns
        -------
        str
            String with the pass information
        """
        relative_start_point = self.get_center(start)
        relative_end_point = self.get_center(end)

        relative_start_round = self.round_iterable(relative_start_point)
        relative_end_round = self.round_iterable(relative_end_point)

        return f"Start: {relative_start_round}, End: {relative_end_round}, Team: {team_name}"

    def tostring(self, coord_transformations: "CoordinatesTransformation") -> str:
        """
        Get a string with the relative coordinates of this pass

        Parameters
        ----------
        coord_transformations : CoordinatesTransformation
            Coordinates transformation

        Returns
        -------
        str
            string with the relative coordinates
        """
        relative_start, relative_end = self.get_relative_coordinates(
            coord_transformations
        )

        return self.generate_output_pass(relative_start, relative_end, self.team.name)

    def __str__(self):
        return self.generate_output_pass(
            self.start_ball_bbox, self.end_ball_bbox, self.team.name
        )


class PassEvent:
    def __init__(self) -> None:
        self.ball = None
        self.closest_player = None
        self.init_player_with_ball = None
        self.last_player_with_ball = None
        self.player_with_ball_counter = 0
        self.player_with_ball_threshold = 3
        self.player_with_ball_threshold_dif_team = 4

    def update(self, closest_player: Player, ball: Ball) -> None:
        """
        Updates the player with the ball counter

        Parameters
        ----------
        closest_player : Player
            The closest player to the ball
        ball : Ball
            Ball class
        """
        self.ball = ball
        self.closest_player = closest_player

        same_id = Player.have_same_id(self.init_player_with_ball, closest_player)

        if same_id:
            self.player_with_ball_counter += 1
        elif not same_id:
            self.player_with_ball_counter = 0

        self.init_player_with_ball = closest_player

    def validate_pass(self, start_player: Player, end_player: Player) -> bool:
        """
        Check if there is a pass between two players of the same team

        Parameters
        ----------
        start_player : Player
            Player that originates the pass
        end_player : Player
            Destination player of the pass

        Returns
        -------
        bool
            Valid pass occurred
        """
        if Player.have_same_id(start_player, end_player):
            return False
        if start_player.team != end_player.team:
            return False

        return True

    def generate_pass(
        self, team: Team, start_pass: np.ndarray, end_pass: np.ndarray
    ) -> Pass:
        """
        Generate a new pass

        Parameters
        ----------
        team : Team
            Pass team
        start_pass : np.ndarray
            Pass start point
        end_pass : np.ndarray
            Pass end point

        Returns
        -------
        Pass
            The generated instance of the Pass class
        """
        start_pass_bbox = [start_pass, start_pass]

        new_pass = Pass(
            start_ball_bbox=start_pass_bbox,
            end_ball_bbox=end_pass,
            team=team,
        )

        return new_pass

    def process_pass(self) -> None:
        """
        Check if a new pass was generated and in the positive case save the new pass into de right team
        """
        if self.player_with_ball_counter >= self.player_with_ball_threshold:
            # init the last player with ball
            if self.last_player_with_ball is None:
                self.last_player_with_ball = self.init_player_with_ball

            valid_pass = self.validate_pass(
                start_player=self.last_player_with_ball,
                end_player=self.closest_player,
            )

            if valid_pass:
                # Generate new pass
                team = self.closest_player.team
                start_pass = self.last_player_with_ball.closest_foot_to_ball_abs(
                    self.ball
                )
                end_pass = self.ball.detection.absolute_points

                new_pass = self.generate_pass(
                    team=team, start_pass=start_pass, end_pass=end_pass
                )
                team.passes.append(new_pass)
            else:
                if (
                    self.player_with_ball_counter
                    < self.player_with_ball_threshold_dif_team
                ):
                    return

            self.last_player_with_ball = self.closest_player



