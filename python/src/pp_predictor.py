import os
import quaternion
import numpy as np
import cv2
import logging
import json

os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
import tensorflow as tf

logging.getLogger("tensorflow").setLevel(logging.ERROR)


def get_cam_lidar_ego_camcalib_lidarcalib_ann(cam_path, lidar_path):
    with open("../train/3d_labels/v1.0-trainval/sample_data.json") as f:
        sample_data = json.load(f)
        name_to_calib_sensor_token = {}
        name_to_ego_pose_token = {}
        name_to_sample_token = {}
        for d in sample_data:
            filename = d["filename"].split("/")[-1]
            calib_sensor_token = d["calibrated_sensor_token"]
            ego_pose_token = d["ego_pose_token"]
            sample_token = d["sample_token"]
            name_to_calib_sensor_token[filename] = calib_sensor_token
            name_to_ego_pose_token[filename] = ego_pose_token
            name_to_sample_token[filename] = sample_token
    with open("../train/3d_labels/v1.0-trainval/sensor.json") as f:
        sensor = json.load(f)
        lidar_token = [s for s in sensor if s["channel"] == "LIDAR_TOP"][0]["token"]
        # camera_token = [s for s in sensor if s["channel"] == "CAM_FRONT"][0]["token"]

    with open("../train/3d_labels/v1.0-trainval/ego_pose.json") as f:
        ego_pose = json.load(f)

    with open("../train/3d_labels/v1.0-trainval/sample_annotation.json") as f:
        sample_annotations = json.load(f)

    with open("../train/3d_labels/v1.0-trainval/calibrated_sensor.json") as f:
        sensor_data = json.load(f)
        # camera_data = [d for d in sensor_data if d["sensor_token"] == camera_token][0]
        lidar_data = [d for d in sensor_data if d["sensor_token"] == lidar_token][0]

    cam_base_name = os.path.basename(cam_path)
    lidar_base_name = os.path.basename(lidar_path)

    cam_calib_sensor_token = name_to_calib_sensor_token[cam_base_name]
    lidar_calib_sensor_token = name_to_calib_sensor_token[lidar_base_name]
    ego_pose_token = name_to_ego_pose_token[cam_base_name]
    sample_token = name_to_sample_token[cam_base_name]
    ego_pose_data = [d for d in ego_pose if d["token"] == ego_pose_token][0]
    camera_data = [d for d in sensor_data if d["token"] == cam_calib_sensor_token][0]
    lidar_data = [d for d in sensor_data if d["token"] == lidar_calib_sensor_token][0]
    sample_anns = [d for d in sample_annotations if d["sample_token"] == sample_token]

    return (
        cv2.cvtColor(cv2.imread(cam_path), cv2.COLOR_BGR2RGB),
        np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 5)),
        ego_pose_data,
        camera_data,
        lidar_data,
        sample_anns,
    )


def predict(segmentation, cam_image, lidar, ego, cam_calib, lidar_calib):
    pred = segmentation(tf.convert_to_tensor(cam_image))
    semantic_probs = pred["semantic_probs"][0].numpy()
    panoptic_pred = pred["panoptic_pred"][0].numpy()
    h, w = panoptic_pred.shape[:2]
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    yxs = np.stack([ys, xs], -1)
    a = h * 5 // 6
    b = h * 7 // 8
    c = w // 2
    d = w * 7 // 8
    e = h * 15 // 16
    yxs = yxs[:, :, 0] >= np.maximum(
        np.maximum(
            yxs[:, :, 1] * (a - e) // c + e, (yxs[:, :, 1] - d) * (b - a) // (w - d) + a
        ),
        a,
    )
    # panoptic_pred[1000:] = 0
    panoptic_pred[yxs] = 0
    # from vis_cityscapes import vis_segmentation, cityscapes_dataset_information
    # vis_segmentation(cam_image, panoptic_pred, cityscapes_dataset_information())

    ego_txyz = np.array(ego["translation"])
    ego_qt = quaternion.as_quat_array(ego["rotation"])

    cam_tx, cam_ty, cam_tz = cam_calib["translation"]
    cam_txyz = np.array([cam_tx, cam_ty, cam_tz])
    cam_qt = quaternion.as_quat_array(cam_calib["rotation"])
    [fx, _, cx], [_, fy, cy], [_, _, _] = cam_calib["camera_intrinsic"]

    lidar_txyz = np.array(lidar_calib["translation"])
    lidar_qt = quaternion.as_quat_array(lidar_calib["rotation"])

    lidar_xyz = lidar[:, :3] + lidar_txyz[np.newaxis]
    lidar_xyz = quaternion.rotate_vectors(lidar_qt, lidar_xyz)
    lidar_xyz = lidar[:, :3] - cam_txyz[np.newaxis]
    lidar_xyz = quaternion.rotate_vectors(cam_qt.inverse(), lidar_xyz)
    lidar_intensity = lidar[:, 3]

    u = np.round(cx + lidar_xyz[:, 0] * fx / lidar_xyz[:, 2]).astype(int)
    v = np.round(cy + lidar_xyz[:, 1] * fy / lidar_xyz[:, 2]).astype(int)
    indices = np.logical_and.reduce(
        (
            u >= 0,
            u < cam_image.shape[1],
            v >= 0,
            v < cam_image.shape[0],
        )
    )
    i = lidar_intensity[indices]
    d = np.linalg.norm(lidar_xyz, ord=2, axis=-1)[indices] / 100
    xyz = lidar_xyz[indices]
    u = u[indices]
    v = v[indices]
    lidar_image = np.zeros((cam_image.shape[0], cam_image.shape[1], 5))
    lidar_image[v, u] = np.concatenate([xyz, np.stack([d, i], -1)], -1)
    if False:
        demo = np.round(
            np.clip(
                np.concatenate(
                    [
                        1 - lidar_image[..., 3:4],
                        lidar_image[..., 4:5],
                        np.zeros((lidar_image.shape[0], lidar_image.shape[1], 1)),
                    ],
                    -1,
                ).astype(np.float32),
                0,
                1,
            )
            * 255
        ).astype(np.uint8)
        demo = (cam_image.astype(np.float32) * demo).astype(np.uint8)
        demo = np.maximum(cam_image, demo)
        # cv2.imshow("window", demo)
        # cv2.waitKey(0)

    pedestrian_centers = []
    panoptic_ids = np.unique(panoptic_pred)
    pedestrian_ids = panoptic_ids[panoptic_ids // 1000 == 11]
    for i in pedestrian_ids:
        pedestrian_mask = panoptic_pred == i
        pedestrian_pixels = pedestrian_mask.astype(np.int32).sum()
        if pedestrian_pixels == 0:
            continue
        lidar_points = lidar_image[pedestrian_mask]
        confidence = semantic_probs[pedestrian_mask][:, 11].mean()
        lidar_mask = (lidar_points > 0).any(-1)
        lidar_points = lidar_points[lidar_mask]
        lidar_points = lidar_points[lidar_points[:, 3] < 0.6]
        if lidar_points.shape[0] == 0:
            continue
        weight = np.exp(-1 / lidar_points.shape[0])
        if lidar_points.shape[0] > 4:
            lidar_points = lidar_points[lidar_points[:, 3].argsort()][
                : -lidar_points.shape[0] // 4
            ]
        lidar_points = lidar_points[lidar_points[:, 4].argsort()][::-1][
            : lidar_points.shape[0] // 2 + 1
        ]
        lidar_x = lidar_points[lidar_points[:, 0].argsort()][
            lidar_points.shape[0] // 2
        ][0]
        lidar_y = lidar_points[lidar_points[:, 1].argsort()][
            lidar_points.shape[0] // 2
        ][1]
        lidar_z = lidar_points[lidar_points[:, 2].argsort()][
            lidar_points.shape[0] // 2
        ][2]
        lidar_i = lidar_points[lidar_points[:, 4].argsort()][
            lidar_points.shape[0] // 2
        ][4]
        confidence = confidence * np.sqrt(lidar_i) * weight
        lidar_cxyz = np.array([lidar_x, lidar_y, lidar_z])
        d = np.linalg.norm(lidar_cxyz, ord=2)
        r = (d + 0.2) / d
        lidar_cxyz = lidar_cxyz * r
        lidar_cxyz = quaternion.rotate_vectors(cam_qt, lidar_cxyz)
        lidar_cxyz = lidar_cxyz + cam_txyz
        lidar_cxyz = quaternion.rotate_vectors(ego_qt, lidar_cxyz)
        lidar_cxyz = lidar_cxyz + ego_txyz
        lidar_cxyz = np.append(lidar_cxyz, confidence)
        pedestrian_centers.append(lidar_cxyz)

    vehicle_centers = []
    vehicle_ids = panoptic_ids[panoptic_ids // 1000 == 13]
    for i in vehicle_ids:
        vehicle_mask = panoptic_pred == i
        vehicle_pixels = vehicle_mask.astype(np.int32).sum()
        if vehicle_pixels == 0:
            continue
        lidar_points = lidar_image[vehicle_mask]
        confidence = semantic_probs[vehicle_mask][:, 13].mean()
        lidar_mask = (lidar_points > 0).any(-1)
        lidar_points = lidar_points[lidar_mask]
        lidar_points = lidar_points[lidar_points[:, 3] < 0.6]
        if lidar_points.shape[0] == 0:
            continue
        weight = np.exp(-1 / lidar_points.shape[0])
        if lidar_points.shape[0] > 4:
            lidar_points = lidar_points[lidar_points[:, 3].argsort()][
                : -lidar_points.shape[0] // 5
            ]
        lidar_points = lidar_points[lidar_points[:, 4].argsort()][::-1][
            : lidar_points.shape[0] // 20 + 1
        ]
        lidar_x = lidar_points[lidar_points[:, 0].argsort()][
            lidar_points.shape[0] // 2
        ][0]
        lidar_y = lidar_points[lidar_points[:, 1].argsort()][
            lidar_points.shape[0] // 2
        ][1]
        lidar_z = lidar_points[lidar_points[:, 2].argsort()][
            lidar_points.shape[0] // 2
        ][2]
        lidar_i = lidar_points[lidar_points[:, 4].argsort()][
            lidar_points.shape[0] // 2
        ][4]
        confidence = confidence * np.sqrt(lidar_i) * weight
        lidar_cxyz = np.array([lidar_x, lidar_y, lidar_z])
        d = np.linalg.norm(lidar_cxyz, ord=2)
        r = (d + 1.6) / d
        lidar_cxyz = lidar_cxyz * r
        lidar_cxyz = quaternion.rotate_vectors(cam_qt, lidar_cxyz)
        lidar_cxyz = lidar_cxyz + cam_txyz
        lidar_cxyz = quaternion.rotate_vectors(ego_qt, lidar_cxyz)
        lidar_cxyz = lidar_cxyz + ego_txyz
        lidar_cxyz = np.append(lidar_cxyz, confidence)
        vehicle_centers.append(lidar_cxyz)

    return pedestrian_centers, vehicle_centers
