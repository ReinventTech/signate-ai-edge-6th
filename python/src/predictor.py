import numpy as np
import pp_predictor
import bev_predictor
import os
import tensorflow as tf
from skimage import io
from time import time


class ScoringService(object):
    @classmethod
    def get_model(cls, model_path, use_pp=False):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.

        Returns:
            bool: The return value. True for success.
        """
        cls.bev_model = tf.saved_model.load(
            os.path.join(model_path, "bev_model")
        ).signatures["serving_default"]
        # cls.bev_model = tf.keras.models.load_model(
        # os.path.join(model_path, "quantized_model.h5")
        # )
        cls.bev_model(tf.zeros((3, 1152, 1152, 41)))

        cls.use_pp = use_pp

        if use_pp:
            cls.pp_model = tf.saved_model.load(
                os.path.join(model_path, "pp_model")
            ).signatures["serving_default"]

        cls.count = 0

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
        # t0 = time()
        # image = io.imread(input["cam_path"])
        lidar = np.fromfile(input["lidar_path"], dtype=np.float32).reshape((-1, 5))
        cam_ego_pose = input["cam_ego_pose"]
        # lidar_ego_pose = input["lidar_ego_pose"]
        cam_calib = input["cam_calibration"]
        lidar_calib = input["lidar_calibration"]

        # t1 = time()
        # make prediction
        bev_pedestrian_preds, bev_vehicle_preds = bev_predictor.predict(
            cls.bev_model, lidar.copy(), cam_ego_pose, cam_calib, lidar_calib
        )
        # t2 = time()
        cam_ego_txy = np.array(cam_ego_pose["translation"])[:2]
        pedestrian_preds = []
        vehicle_preds = []

        bev_pedestrian_preds = [
            [-p[3], 1e10, p[3], p[:2]]
            for p in bev_pedestrian_preds
            if np.linalg.norm(p[:2] - cam_ego_txy, ord=2) <= 40
        ]

        rem_pedestrian_preds = bev_pedestrian_preds

        while len(rem_pedestrian_preds) > 0 and len(pedestrian_preds) < 50:
            rem_pedestrian_preds.sort(key=lambda x: x[0])
            pred = rem_pedestrian_preds.pop(0)
            pedestrian_preds.append([*pred[3][:2], -pred[0]])
            for n in range(len(rem_pedestrian_preds)):
                dist = np.linalg.norm(
                    rem_pedestrian_preds[n][3] - np.array(pred[3]), ord=2
                )
                lb, ub, d = 0.5, 1.0, 0.9
                rem_pedestrian_preds[n][1] = min(dist, rem_pedestrian_preds[n][1])
                r = lb + (ub - lb) * min(rem_pedestrian_preds[n][1], d) / d
                rem_pedestrian_preds[n][0] = -rem_pedestrian_preds[n][2] * r

        bev_vehicle_preds = [
            [-p[3], 1e10, p[3], p[:2]]
            for p in bev_vehicle_preds
            if np.linalg.norm(p[:2] - cam_ego_txy, ord=2) <= 50
        ]

        rem_vehicle_preds = bev_vehicle_preds

        while len(rem_vehicle_preds) > 0 and len(vehicle_preds) < 50:
            rem_vehicle_preds.sort(key=lambda x: x[0])
            pred = rem_vehicle_preds.pop(0)
            vehicle_preds.append([*pred[3][:2], -pred[0]])
            for n in range(len(rem_vehicle_preds)):
                dist = np.linalg.norm(
                    rem_vehicle_preds[n][3] - np.array(pred[3]), ord=2
                )
                lb, ub, d = 0.6, 1.0, 1.6
                rem_vehicle_preds[n][1] = min(dist, rem_vehicle_preds[n][1])
                r = lb + (ub - lb) * min(rem_vehicle_preds[n][1], d) / d
                rem_vehicle_preds[n][0] = -rem_vehicle_preds[n][2] * r

        pedestrian_preds = [[p[0], p[1], p[2]] for p in pedestrian_preds]
        pedestrian_preds = pedestrian_preds[:50]
        vehicle_preds = [[p[0], p[1], p[2]] for p in vehicle_preds]
        vehicle_preds = vehicle_preds[:50]

        prediction = {}
        if len(pedestrian_preds) > 0:
            prediction["pedestrian"] = pedestrian_preds
        if len(vehicle_preds) > 0:
            prediction["vehicle"] = vehicle_preds

        # make output
        output = {input["test_key"]: prediction}
        print(cls.count)
        cls.count += 1
        # t3 = time()
        # print("root", t1 - t0, t2 - t1, t3 - t2)
        # print(prediction)
        # exit()

        return output
