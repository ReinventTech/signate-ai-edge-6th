import numpy as np
import bev_predictor
import os
import tensorflow as tf
from visualizer import visualize_predictions
from numba import njit

# import pp_predictor
from time import time


@njit
def sort_predictions(bev_pedestrian_preds, bev_vehicle_preds, cam_ego_txy):
    pedestrian_preds = []
    vehicle_preds = []

    rem_pedestrian_preds = []
    for p in bev_pedestrian_preds:
        d = np.linalg.norm(p[:2] - cam_ego_txy, ord=2)
        if d > 40:
            continue
        if d > 39:
            pred = np.float32([-p[3] * 0.95, 1e10, p[3] * 0.95, p[0], p[1]])
            rem_pedestrian_preds.append(pred)
        else:
            pred = np.float32([-p[3], 1e10, p[3], p[0], p[1]])
            rem_pedestrian_preds.append(pred)

    rem_pedestrian_preds.sort(key=lambda x: x[0])
    rem_pedestrian_preds = rem_pedestrian_preds[:55]

    while len(rem_pedestrian_preds) > 0 and len(pedestrian_preds) < 50:
        rem_pedestrian_preds.sort(key=lambda x: x[0])
        pred = rem_pedestrian_preds.pop(0)
        pedestrian_preds.append([pred[3], pred[4], -pred[0]])
        for n in range(len(rem_pedestrian_preds)):
            dist = np.linalg.norm(rem_pedestrian_preds[n][3:5] - pred[3:5], ord=2)
            lb, ub, d = 0.0, 1.0, 0.8
            rem_pedestrian_preds[n][1] = min(dist, rem_pedestrian_preds[n][1])
            if rem_pedestrian_preds[n][1] < 0.4:
                r = 0.01
            r = lb + (ub - lb) * min(rem_pedestrian_preds[n][1], d) / d
            if rem_pedestrian_preds[n][1] == 0:
                r = 1
            rem_pedestrian_preds[n][0] = -rem_pedestrian_preds[n][2] * r

    rem_vehicle_preds = []
    for p in bev_vehicle_preds:
        d = np.linalg.norm(p[:2] - cam_ego_txy, ord=2)
        if d > 50:
            continue
        if d > 49.5:
            pred = np.float32([-p[3] * 0.9, 1e10, p[3] * 0.9, p[0], p[1]])
            rem_vehicle_preds.append(pred)
        else:
            pred = np.float32([-p[3], 1e10, p[3], p[0], p[1]])
            rem_vehicle_preds.append(pred)

    rem_vehicle_preds.sort(key=lambda x: x[0])
    rem_vehicle_preds = rem_vehicle_preds[:60]

    while len(rem_vehicle_preds) > 0 and len(vehicle_preds) < 50:
        rem_vehicle_preds.sort(key=lambda x: x[0])
        pred = rem_vehicle_preds.pop(0)
        vehicle_preds.append([pred[3], pred[4], -pred[0]])
        for n in range(len(rem_vehicle_preds)):
            dist = np.linalg.norm(rem_vehicle_preds[n][3:5] - pred[3:5], ord=2)
            lb, ub, d = 0.0, 1.0, 2.2
            rem_vehicle_preds[n][1] = min(dist, rem_vehicle_preds[n][1])
            r = lb + (ub - lb) * min(rem_vehicle_preds[n][1], d) / d
            if rem_vehicle_preds[n][1] < 0.4:
                r = 0.01
            rem_vehicle_preds[n][0] = -rem_vehicle_preds[n][2] * r

    pedestrian_preds = [[p[0], p[1], p[2]] for p in pedestrian_preds]
    pedestrian_preds = pedestrian_preds[:50]
    vehicle_preds = [[p[0], p[1], p[2]] for p in vehicle_preds]
    vehicle_preds = vehicle_preds[:50]
    return pedestrian_preds, vehicle_preds


class ScoringService(object):
    @classmethod
    def get_model(cls, model_path):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.

        Returns:
            bool: The return value. True for success.
        """
        cls.bev_model = tf.saved_model.load(
            os.path.join(model_path, "bev_model")
        ).signatures["serving_default"]

        cls.bev_model(tf.zeros((3, 1152, 1152, 24)))["detector"].numpy()

        # if use_pp:
        # cls.use_pp = use_pp
        # cls.pp_model = tf.saved_model.load(
        # os.path.join(model_path, "pp_model")
        # ).signatures["serving_default"]

        cls.count = 0
        cls.ego_pose_records = []
        cls.pred_records = []
        cls.last_scene = None

        return True

    @classmethod
    def predict(cls, input):
        """Predict method

        Args:
            input: meta data of the sample you want to make inference from (dict)

        Returns:
            dict: Inference for the given input.

        """
        # load sample
        t0 = time()
        lidar = np.fromfile(input["lidar_path"], dtype=np.float32).reshape((-1, 5))
        cam_ego_pose = input["cam_ego_pose"]

        scene = input["test_key"].split("_")[0]
        if cls.last_scene != scene:
            cls.ego_pose_records = []
            cls.pred_records = []
        cls.last_scene = scene

        cam_calib = input["cam_calibration"]
        lidar_calib = input["lidar_calibration"]

        t1 = time()
        # make prediction
        summary = False
        bev_pedestrian_preds, bev_vehicle_preds, summary_image = bev_predictor.predict(
            cls.bev_model,
            lidar.copy(),
            cam_ego_pose,
            cam_calib,
            lidar_calib,
            cls.ego_pose_records,
            cls.pred_records,
            summary,
        )
        cls.ego_pose_records = cls.ego_pose_records[-2:]
        cls.pred_records = cls.pred_records[-2:]
        t2 = time()
        cam_ego_txy = np.array(cam_ego_pose["translation"])[:2]

        pedestrian_preds, vehicle_preds = sort_predictions(
            bev_pedestrian_preds, bev_vehicle_preds, cam_ego_txy
        )

        prediction = {}
        if len(pedestrian_preds) > 0:
            prediction["pedestrian"] = pedestrian_preds
        if len(vehicle_preds) > 0:
            prediction["vehicle"] = vehicle_preds

        # make output
        output = {input["test_key"]: prediction}
        # if cls.count % 10 == 0:
        # print(cls.count, len(pedestrian_preds), len(vehicle_preds))
        cls.count += 1
        t3 = time()
        print("root", t1 - t0, t2 - t1, t3 - t2)

        if summary:
            import cv2
            import json

            with open("../tmp/ans.json") as f:
                ans = json.load(f)
                ans = ans[input["test_key"]]
            summary_image = visualize_predictions(
                pedestrian_preds, vehicle_preds, summary_image, cam_ego_pose, ans
            )
            cv2.imwrite(f"summary/summary_{cls.count}.png", summary_image)

        return output
