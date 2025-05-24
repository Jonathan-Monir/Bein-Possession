from typing import List

import norfair
import numpy as np
import pandas as pd


class Converter:
    @staticmethod
    def Boxes_to_Detections(boxes) -> List[norfair.Detection]:
        """
        Converts YOLOv11 detection boxes to a list of norfair.Detection objects.

        Parameters
        ----------
        boxes : List
            List of detection boxes from YOLOv11, each with attributes 'cls', 'conf', and 'xyxy'.

        Returns
        -------
        List[norfair.Detection]
            List of norfair.Detection objects.
        """
        detections = []
        for box in boxes:
            # Remove extra dimensions from box.xyxy
            coords = box.xyxy.squeeze()  # now should be a tensor with shape (4,)
            xmin, ymin, xmax, ymax = coords.tolist()

            points = np.array([[xmin, ymin], [xmax, ymax]])

            # Extract class ID and confidence
            class_id = int(box.cls)
            confidence = float(box.conf)

            # Create detection data dictionary
            data = {
                "class_id": class_id,
                "confidence": confidence,
            }

            # Create Norfair Detection object
            detection = norfair.Detection(points=points, data=data, label=class_id)
            detections.append(detection)
        return detections

    @staticmethod
    def DataFrame_to_Detections(df: pd.DataFrame) -> List[norfair.Detection]:
        """
        Converts a DataFrame to a list of norfair.Detection

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the bounding boxes

        Returns
        -------
        List[norfair.Detection]
            List of norfair.Detection
        """

        detections = []

        for index, row in df.iterrows():
            # get the bounding box coordinates
            xmin = round(row["xmin"])
            ymin = round(row["ymin"])
            xmax = round(row["xmax"])
            ymax = round(row["ymax"])

            box = np.array(
                [
                    [xmin, ymin],
                    [xmax, ymax],
                ]
            )

            # get the predicted class
            name = row["name"]
            confidence = row["confidence"]

            data = {
                "name": name,
                "p": confidence,
            }

            if "color" in row:
                data["color"] = row["color"]

            if "label" in row:
                data["label"] = row["label"]

            if "classification" in row:
                data["classification"] = row["classification"]

            detection = norfair.Detection(
                points=box,
                data=data,
            )

            detections.append(detection)

        return detections

    @staticmethod
    def Detections_to_DataFrame(detections: List[norfair.Detection]) -> pd.DataFrame:
        """
        Converts a list of norfair.Detection objects to a DataFrame.
        
        Parameters
        ----------
        detections : List[norfair.Detection]
            List of Norfair Detection objects.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing the bounding boxes and associated detection data.
        """
        import pandas as pd

        df = pd.DataFrame()
        # Mapping from class_id to names (adjust as necessary)
        class_mapping = {0: "ball", 1: "player"}
        
        for detection in detections:
            xmin = detection.points[0][0]
            ymin = detection.points[0][1]
            xmax = detection.points[1][0]
            ymax = detection.points[1][1]

            # Try to get "name" from detection.data; if missing, use class_id to get a name.
            name = detection.data.get("name")
            if name is None:
                class_id = detection.data.get("class_id")
                name = class_mapping.get(class_id, "unknown") if class_id is not None else "unknown"

            # Confidence might be stored under "p" or "confidence"
            confidence = detection.data.get("p", detection.data.get("confidence", 0))

            data = {
                "xmin": [xmin],
                "ymin": [ymin],
                "xmax": [xmax],
                "ymax": [ymax],
                "name": [name],
                "confidence": [confidence],
            }

            if "color" in detection.data:
                data["color"] = [detection.data["color"]]
            if "label" in detection.data:
                data["label"] = [detection.data["label"]]
            if "classification" in detection.data:
                data["classification"] = [detection.data["classification"]]

            df_new_row = pd.DataFrame.from_records(data)
            df = pd.concat([df, df_new_row], ignore_index=True)

        return df

    @staticmethod
    def TrackedObjects_to_Detections_nor(
            tracked_objects: List[norfair.tracker.TrackedObject], cls
        ) -> List[norfair.Detection]:
            """
            Converts a list of norfair.tracker.TrackedObject to a list of norfair.Detection

            Parameters
            ----------
            tracked_objects : List[norfair.tracker.TrackedObject]
                List of norfair.tracker.TrackedObject

            Returns
            -------
            List[norfair.Detection]
                List of norfair.Detection
            """

            live_objects = [
                entity for entity in tracked_objects if entity.live_points.any()
            ]

            detections = []

            for tracked_object in live_objects:
                detection = tracked_object.last_detection
                detection.data["id"] = int(tracked_object.id)
                detection.label = cls
                detections.append(detection)

            return detections

    @staticmethod
    
    def TrackedObjects_to_Detections(
            tracked_objects: List[norfair.tracker.TrackedObject],cls: int
    ) -> List[List]:
        """
        Converts a list of norfair.TrackedObject to a list of tuples containing detection information.

        Parameters
        ----------
        tracked_objects : List[norfair.TrackedObject]
            List of norfair.TrackedObject

        Returns
        -------
        List[tuple]
            List of tuples containing (cls, x1, y1, x2, y2)
        """

        live_objects = [
            entity for entity in tracked_objects if entity.live_points.any()
        ]

        detections = []

        for tracked_object in live_objects:
            detection = tracked_object.last_detection
            points = detection.points
            if points.shape[0] == 2:  # Assuming points contain two corners of the bounding box
                x1, y1 = points[0]
                x2, y2 = points[1]
                detections.append([cls, x1, y1, x2, y2])

        return detections

