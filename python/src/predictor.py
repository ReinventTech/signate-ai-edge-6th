import numpy as np
import pp_predictor
import bev_predictor
import os
import tensorflow as tf
from skimage import io


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

        cls.use_pp = use_pp

        if use_pp:
            cls.pp_model = tf.saved_model.load(
                os.path.join(model_path, "pp_model")
            ).signatures["serving_default"]

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
        image = io.imread(input["cam_path"])
        lidar = np.fromfile(input["lidar_path"], dtype=np.float32).reshape((-1, 5))
        cam_ego_pose = input["cam_ego_pose"]
        # lidar_ego_pose = input["lidar_ego_pose"]
        cam_calib = input["cam_calibration"]
        lidar_calib = input["lidar_calibration"]

        # make prediction
        bev_pedestrian_preds, bev_vehicle_preds = bev_predictor.predict(
            cls.bev_model, image, lidar.copy(), cam_ego_pose, cam_calib, lidar_calib
        )
        cam_ego_txy = np.array(cam_ego_pose["translation"])[:2]
        pedestrian_preds = []
        vehicle_preds = []
        if cls.use_pp:
            pp_pedestrian_preds, pp_vehicle_preds = pp_predictor.predict(
                cls.pp_model, image, lidar, cam_ego_pose, cam_calib, lidar_calib
            )
            pp_pedestrian_preds = [
                [-p[3] * 1.5, 1e10, p[3] * 1.5, p[:2]]
                for p in pp_pedestrian_preds
                if np.linalg.norm(p[:2] - cam_ego_txy, ord=2) <= 40
            ]
            # pp_vehicle_preds = [
            # [-p[3] * 0.5, 1e10, p[3] * 0.5, p[:2]]
            # for p in pp_vehicle_preds
            # if np.linalg.norm(p[:2] - cam_ego_txy, ord=2) <= 50
            # ]
            pp_vehicle_preds = []
        else:
            pp_pedestrian_preds = []
            pp_vehicle_preds = []

        bev_pedestrian_preds = [
            [-p[3], 1e10, p[3], p[:2]]
            for p in bev_pedestrian_preds
            if np.linalg.norm(p[:2] - cam_ego_txy, ord=2) <= 40
        ]

        rem_pedestrian_preds = pp_pedestrian_preds + bev_pedestrian_preds

        while len(rem_pedestrian_preds) > 0 and len(pedestrian_preds) < 50:
            rem_pedestrian_preds.sort(key=lambda x: x[0])
            pred = rem_pedestrian_preds.pop(0)
            pedestrian_preds.append([*pred[3][:2], -pred[0]])
            for n in range(len(rem_pedestrian_preds)):
                dist = np.linalg.norm(
                    rem_pedestrian_preds[n][3] - np.array(pred[3]), ord=2
                )
                lb, ub, d = 0.2, 1.0, 1.0
                rem_pedestrian_preds[n][1] = min(dist, rem_pedestrian_preds[n][1])
                r = lb + (ub - lb) / d * min(rem_pedestrian_preds[n][1], d)
                rem_pedestrian_preds[n][0] = -rem_pedestrian_preds[n][2] * r

        bev_vehicle_preds = [
            [-p[3], 1e10, p[3], p[:2]]
            for p in bev_vehicle_preds
            if np.linalg.norm(p[:2] - cam_ego_txy, ord=2) <= 50
        ]

        rem_vehicle_preds = pp_vehicle_preds + bev_vehicle_preds

        while len(rem_vehicle_preds) > 0 and len(vehicle_preds) < 50:
            rem_vehicle_preds.sort(key=lambda x: x[0])
            pred = rem_vehicle_preds.pop(0)
            vehicle_preds.append([*pred[3][:2], -pred[0]])
            for n in range(len(rem_vehicle_preds)):
                dist = np.linalg.norm(rem_vehicle_preds[n][3] - np.array(pred[3]), ord=2)
                lb, ub, d = 0.4, 1.0, 2.0
                rem_vehicle_preds[n][1] = min(dist, rem_vehicle_preds[n][1])
                r = lb + (ub - lb) / d * min(rem_vehicle_preds[n][1], d)
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

        return output
