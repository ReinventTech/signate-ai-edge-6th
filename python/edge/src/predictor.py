import sys
import numpy as np
import bev_predictor

sys.path.append("../../src")
from visualizer import visualize_predictions
from numba import njit, prange
from time import time


@njit
def sort_predictions(
    bev_pedestrian_preds, bev_vehicle_preds, cam_ego_txy_pedestrian, cam_ego_txy_vehicle
):
    pedestrian_preds = []
    vehicle_preds = []

    rem_pedestrian_preds = []
    ds = bev_pedestrian_preds[:, :2] - cam_ego_txy_pedestrian
    ds = ds**2
    ds = np.sqrt(ds[:, 0] + ds[:, 1])
    m, f = ds <= 40, ds > 39
    bev_pedestrian_preds[f, 3] = bev_pedestrian_preds[f, 3] * 0.95
    bev_pedestrian_preds = bev_pedestrian_preds[m]
    for n in prange(bev_pedestrian_preds.shape[0]):
        p = bev_pedestrian_preds[n]
        pred = np.float32([-p[3], 1e10, p[3], p[0], p[1]])
        rem_pedestrian_preds.append(pred)

    rem_pedestrian_preds.sort(key=lambda x: x[0])
    rem_pedestrian_preds = rem_pedestrian_preds[:55]

    lb, ub, d = 0.0, 1.0, 0.8
    m = (ub - lb) / d
    while len(rem_pedestrian_preds) > 0 and len(pedestrian_preds) < 50:
        rem_pedestrian_preds.sort(key=lambda x: x[0])
        pred = rem_pedestrian_preds.pop(0)
        pedestrian_preds.append([pred[3], pred[4], -pred[0]])
        for n in prange(len(rem_pedestrian_preds)):
            dist = np.linalg.norm(rem_pedestrian_preds[n][3:5] - pred[3:5], ord=2)
            rem_pedestrian_preds[n][1] = min(dist, rem_pedestrian_preds[n][1])
            if rem_pedestrian_preds[n][1] < 0.4:
                r = 0.01
            if rem_pedestrian_preds[n][1] == 0:
                r = 1
            else:
                r = lb + m * min(rem_pedestrian_preds[n][1], d)
            rem_pedestrian_preds[n][0] = -rem_pedestrian_preds[n][2] * r

    rem_vehicle_preds = []
    ds = bev_vehicle_preds[:, :2] - cam_ego_txy_vehicle
    ds = ds**2
    ds = np.sqrt(ds[:, 0] + ds[:, 1])
    m, f = ds <= 50, ds > 49.5
    bev_vehicle_preds[f, 3] = bev_vehicle_preds[f, 3] * 0.9
    bev_vehicle_preds = bev_vehicle_preds[m]
    for n in prange(bev_vehicle_preds.shape[0]):
        p = bev_vehicle_preds[n]
        pred = np.float32([-p[3], 1e10, p[3], p[0], p[1]])
        rem_vehicle_preds.append(pred)

    rem_vehicle_preds.sort(key=lambda x: x[0])
    rem_vehicle_preds = rem_vehicle_preds[:60]

    lb, ub, d = 0.0, 1.0, 2.2
    m = (ub - lb) / d
    while len(rem_vehicle_preds) > 0 and len(vehicle_preds) < 50:
        rem_vehicle_preds.sort(key=lambda x: x[0])
        pred = rem_vehicle_preds.pop(0)
        vehicle_preds.append([pred[3], pred[4], -pred[0]])
        for n in prange(len(rem_vehicle_preds)):
            dist = np.linalg.norm(rem_vehicle_preds[n][3:5] - pred[3:5], ord=2)
            rem_vehicle_preds[n][1] = min(dist, rem_vehicle_preds[n][1])
            if rem_vehicle_preds[n][1] < 0.4:
                r = 0.01
            else:
                r = lb + m * min(rem_vehicle_preds[n][1], d)
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
        summary = False

        # make prediction
        bev_pedestrian_preds, bev_vehicle_preds, summary_image = bev_predictor.predict(
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
        cam_ego_txy = np.array(cam_ego_pose["translation"], np.float32)[:2]

        pedestrian_preds, vehicle_preds = (
            sort_predictions(
                bev_pedestrian_preds,
                bev_vehicle_preds,
                np.tile(cam_ego_txy[np.newaxis], (bev_pedestrian_preds.shape[0], 1)),
                np.tile(cam_ego_txy[np.newaxis], (bev_vehicle_preds.shape[0], 1)),
            )
            if len(bev_pedestrian_preds) > 0 and len(bev_vehicle_preds) > 0
            else ([], [])
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

        print(cls.count)
        # if cls.count > 10:
        # exit()

        return output
