import numpy as np
import bev_predictor
import os
import tensorflow as tf

# import pp_predictor
# from time import time


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
        # t0 = time()
        lidar = np.fromfile(input["lidar_path"], dtype=np.float32).reshape((-1, 5))
        cam_ego_pose = input["cam_ego_pose"]

        scene = input["test_key"].split("_")[0]
        if cls.last_scene != scene:
            cls.ego_pose_records = []
            cls.pred_records = []
        cls.last_scene = scene

        cam_calib = input["cam_calibration"]
        lidar_calib = input["lidar_calibration"]

        # t1 = time()
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
        # t2 = time()
        cam_ego_txy = np.array(cam_ego_pose["translation"])[:2]
        pedestrian_preds = []
        vehicle_preds = []

        rem_pedestrian_preds = []
        for p in bev_pedestrian_preds:
            d = np.linalg.norm(p[:2] - cam_ego_txy, ord=2)
            if d > 40:
                continue
            if d > 39:
                rem_pedestrian_preds.append([-p[3] * 0.95, 1e10, p[3] * 0.95, p[:2]])
            else:
                rem_pedestrian_preds.append([-p[3], 1e10, p[3], p[:2]])

        rem_pedestrian_preds.sort(key=lambda x: x[0])
        rem_pedestrian_preds = rem_pedestrian_preds[:55]

        suppress_close_objects = True
        if suppress_close_objects:
            while len(rem_pedestrian_preds) > 0 and len(pedestrian_preds) < 50:
                rem_pedestrian_preds.sort(key=lambda x: x[0])
                pred = rem_pedestrian_preds.pop(0)
                pedestrian_preds.append([*pred[3][:2], -pred[0]])
                for n in range(len(rem_pedestrian_preds)):
                    dist = np.linalg.norm(
                        rem_pedestrian_preds[n][3] - np.array(pred[3]), ord=2
                    )
                    lb, ub, d = 0.0, 1.0, 0.8
                    rem_pedestrian_preds[n][1] = min(dist, rem_pedestrian_preds[n][1])
                    if rem_pedestrian_preds[n][1] < 0.4:
                        r = 0.01
                    r = lb + (ub - lb) * min(rem_pedestrian_preds[n][1], d) / d
                    if rem_pedestrian_preds[n][1] == 0:
                        r = 1
                    rem_pedestrian_preds[n][0] = -rem_pedestrian_preds[n][2] * r
        else:
            while len(rem_pedestrian_preds) > 0 and len(pedestrian_preds) < 50:
                pred = rem_pedestrian_preds.pop(0)
                pedestrian_preds.append([*pred[3][:2], -pred[0]])

        rem_vehicle_preds = []
        for p in bev_vehicle_preds:
            d = np.linalg.norm(p[:2] - cam_ego_txy, ord=2)
            if d > 50:
                continue
            if d > 49.5:
                rem_vehicle_preds.append([-p[3] * 0.9, 1e10, p[3] * 0.9, p[:2]])
            else:
                rem_vehicle_preds.append([-p[3], 1e10, p[3], p[:2]])

        rem_vehicle_preds.sort(key=lambda x: x[0])
        rem_vehicle_preds = rem_vehicle_preds[:60]

        if suppress_close_objects:
            while len(rem_vehicle_preds) > 0 and len(vehicle_preds) < 50:
                rem_vehicle_preds.sort(key=lambda x: x[0])
                pred = rem_vehicle_preds.pop(0)
                vehicle_preds.append([*pred[3][:2], -pred[0]])
                for n in range(len(rem_vehicle_preds)):
                    dist = np.linalg.norm(
                        rem_vehicle_preds[n][3] - np.array(pred[3]), ord=2
                    )
                    lb, ub, d = 0.0, 1.0, 2.2
                    rem_vehicle_preds[n][1] = min(dist, rem_vehicle_preds[n][1])
                    r = lb + (ub - lb) * min(rem_vehicle_preds[n][1], d) / d
                    if rem_vehicle_preds[n][1] < 0.4:
                        r = 0.01
                    rem_vehicle_preds[n][0] = -rem_vehicle_preds[n][2] * r
        else:
            while len(rem_vehicle_preds) > 0 and len(vehicle_preds) < 50:
                pred = rem_vehicle_preds.pop(0)
                vehicle_preds.append([*pred[3][:2], -pred[0]])

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
        # if cls.count % 10 == 0:
        # print(cls.count, len(pedestrian_preds), len(vehicle_preds))
        cls.count += 1
        # t3 = time()
        # print("root", t1 - t0, t2 - t1, t3 - t2)

        if summary:
            import cv2
            import quaternion
            import json

            ego_txyz = tf.cast(cam_ego_pose["translation"], np.float32)
            ego_qt = quaternion.as_quat_array(cam_ego_pose["rotation"])

            # pred_image = np.zeros((1152, 1152, 3), np.uint8)
            pred_image = summary_image[:, :1152].copy()
            gt_image = summary_image[:, :1152].copy()
            with open("../tmp/ans.json") as f:
                ans = json.load(f)
                ans = ans[input["test_key"]]
            for n, p in enumerate(ans["pedestrian"]):
                xyz = np.array([p[0], p[1], 0])
                xyz = xyz - ego_txyz
                xyz = quaternion.rotate_vectors(ego_qt.inverse(), xyz)
                xyz[0] = xyz[0] * 10 + 512 + 64
                xyz[1] = -xyz[1] * 10 + 512 + 64
                xyz = np.clip(np.round(xyz), 0, 1152).astype(int)
                gt_image = cv2.circle(
                    gt_image, (xyz[0], xyz[1]), 1, (255, 0, 0), 3, cv2.LINE_AA
                )
                pred_image = cv2.circle(
                    pred_image, (xyz[0], xyz[1]), 7, (0, 255, 255), 2, cv2.LINE_AA
                )
            for n, p in enumerate(ans["vehicle"]):
                xyz = np.array([p[0], p[1], 0])
                xyz = xyz - ego_txyz
                xyz = quaternion.rotate_vectors(ego_qt.inverse(), xyz)
                xyz[0] = xyz[0] * 10 + 512 + 64
                xyz[1] = -xyz[1] * 10 + 512 + 64
                xyz = np.clip(np.round(xyz), 0, 1152).astype(int)
                gt_image = cv2.circle(
                    gt_image, (xyz[0], xyz[1]), 1, (0, 255, 0), 3, cv2.LINE_AA
                )
                pred_image = cv2.circle(
                    pred_image, (xyz[0], xyz[1]), 7, (255, 0, 255), 2, cv2.LINE_AA
                )
            for n, p in enumerate(pedestrian_preds):
                color = (
                    (255, 0, 0)
                    if p[2] >= 0.5
                    else (144, 0, 0)
                    if p[2] >= 0.2
                    else (96, 0, 0)
                )
                thickness = 4 if p[2] >= 0.5 else 3 if p[2] >= 0.2 else 2
                xyz = np.array([p[0], p[1], 0])
                xyz = xyz - ego_txyz
                xyz = quaternion.rotate_vectors(ego_qt.inverse(), xyz)
                xyz[0] = xyz[0] * 10 + 512 + 64
                xyz[1] = -xyz[1] * 10 + 512 + 64
                xyz = np.clip(np.round(xyz), 0, 1152).astype(int)
                pred_image = cv2.circle(
                    pred_image, (xyz[0], xyz[1]), 1, color, thickness, cv2.LINE_AA
                )
                pred_image = cv2.putText(
                    pred_image,
                    f"{n}:{int(p[2]*100)}",
                    (xyz[0] + 3, xyz[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            for n, p in enumerate(vehicle_preds):
                color = (
                    (0, 255, 0)
                    if p[2] >= 0.5
                    else (0, 144, 0)
                    if p[2] >= 0.2
                    else (0, 96, 0)
                )
                thickness = 4 if p[2] >= 0.5 else 3 if p[2] >= 0.2 else 2
                xyz = np.array([p[0], p[1], 0])
                xyz = xyz - ego_txyz
                xyz = quaternion.rotate_vectors(ego_qt.inverse(), xyz)
                xyz[0] = xyz[0] * 10 + 512 + 64
                xyz[1] = -xyz[1] * 10 + 512 + 64
                xyz = np.clip(np.round(xyz), 0, 1152).astype(int)
                pred_image = cv2.circle(
                    pred_image, (xyz[0], xyz[1]), 1, color, thickness, cv2.LINE_AA
                )
                pred_image = cv2.putText(
                    pred_image,
                    f"{n}:{int(p[2]*100)}",
                    (xyz[0] + 3, xyz[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            # summary_image = np.concatenate([summary_image, pred_image, gt_image], 1)
            summary_image = np.concatenate([summary_image[:, 1152:], pred_image], 1)

            summary_image = cv2.cvtColor(summary_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"summary/summary_{cls.count}.png", summary_image)
            # exit()


        return output
