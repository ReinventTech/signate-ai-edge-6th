import os
import quaternion
import numpy as np
import cv2
import logging
import json
from time import time

os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
import tensorflow as tf
import tensorflow_addons as tfa


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


# @tf.function
def preprocess(lidar_points, z_offset=3.7):
    lidar_xs = tf.cast(tf.math.round((lidar_points[:, 0] + 57.6) * 10), tf.int32)
    lidar_ys = tf.cast(tf.math.round((-lidar_points[:, 1] + 57.6) * 10), tf.int32)
    lidar_zs = tf.cast(tf.math.round((lidar_points[:, 2] + z_offset) * 10), tf.int32)
    indices = tf.math.logical_and(
        tf.math.logical_and(
            tf.math.logical_and(lidar_xs >= 0, lidar_xs < 1152),
            tf.math.logical_and(lidar_ys >= 0, lidar_ys < 1152),
        ),
        tf.math.logical_and(lidar_zs >= 0, lidar_zs < 40),
    )
    lidar_xs = tf.boolean_mask(lidar_xs, indices)
    lidar_ys = tf.boolean_mask(lidar_ys, indices)
    lidar_zs = tf.boolean_mask(lidar_zs, indices)
    intensities = tf.boolean_mask(lidar_points[:, 3], indices)

    lidar_image = tf.zeros((1152, 1152, 40), tf.float32)
    lidar_image = tf.tensor_scatter_nd_max(
        lidar_image, tf.stack([lidar_ys, lidar_xs, lidar_zs], -1), intensities
    )
    xs = tf.expand_dims(tf.range(1152, dtype=np.float32), 0)
    ys = tf.expand_dims(tf.range(1152, dtype=np.float32), -1)
    xs = tf.tile(xs, (1152, 1))
    ys = tf.tile(ys, (1, 1152))
    xys = tf.stack([xs, ys], -1)
    vs = xys - tf.convert_to_tensor([[[576, 576]]], tf.float32)
    ds = tf.linalg.norm(vs, ord=2, axis=-1) / 1000

    input_image = tf.expand_dims(
        tf.concat([lidar_image, tf.expand_dims(ds, -1)], -1), 0
    )
    return input_image


xs = np.arange(1024)
ys = np.arange(1024)
pedestrian_mask = (xs[np.newaxis, :] - 512) ** 2 + (
    ys[:, np.newaxis] - 512
) ** 2 > 410**2
vehicle_mask = (xs[np.newaxis, :] - 512) ** 2 + (
    ys[:, np.newaxis] - 512
) ** 2 > 510**2


# @tf.function
def postprocess(pred):
    # pred = pred.numpy()
    # t0 = time()
    y = pred[:, 64:-64, 64:-64, :]
    y[:, pedestrian_mask, 0] = 0
    y[:, vehicle_mask, 1] = 0
    pedestrian_fy = y[..., 0].copy()
    vehicle_fy = y[..., 1].copy()
    # pr = np.clip(pedestrian_fy.sum() / 2300, 0.15, 0.35)
    # vr = np.clip(vehicle_fy.sum() / 24000, 0.5, 0.75)
    # print(pedestrian_fy.sum(), vehicle_fy.sum())
    # t1 = time()
    # m = np.stack([y[..., 0] > 0.25, y[..., 1] > 0.64], -1)
    # pedestrian_m = pedestrian_fy > pr
    # vehicle_m = vehicle_fy > vr
    pedestrian_m = pedestrian_fy > 0.32
    vehicle_m = vehicle_fy > 0.64
    # t2 = time()
    # fy = y * m  # .astype(np.float32)
    pedestrian_fy[np.logical_not(pedestrian_m)] = 0
    vehicle_fy[np.logical_not(vehicle_m)] = 0
    # t3 = time()
    (
        n_pedestrian,
        pedestrian_ccs,
        _,
        pedestrian_centroid,
    ) = cv2.connectedComponentsWithStats(
        pedestrian_m[0].astype(np.uint8), connectivity=4
    )
    # t4 = time()
    pedestrian_centroid1 = pedestrian_centroid[:, [1, 0]]
    (
        n_vehicle,
        vehicle_ccs,
        _,
        vehicle_centroid,
    ) = cv2.connectedComponentsWithStats(vehicle_m[0].astype(np.uint8), connectivity=4)
    # t5 = time()
    vehicle_centroid1 = vehicle_centroid[:, [1, 0]]
    pedestrian_confidence1 = np.zeros([n_pedestrian])
    # t6 = time()
    # np.maximum.at(
    # pedestrian_confidence1,
    # np.reshape(pedestrian_ccs, [-1]),
    # np.reshape(fy[..., 0], [-1]),
    # )
    # print(pedestrian_confidence1.shape, pedestrian_ccs.shape, fy[..., 0].shape)
    pedestrian_confidence1 = tf.tensor_scatter_nd_max(
        pedestrian_confidence1,
        np.reshape(pedestrian_ccs, [-1, 1]),
        np.reshape(pedestrian_fy, [-1]),
    ).numpy()
    # t7 = time()
    pedestrian_centroid1 = pedestrian_centroid1[1:]
    pedestrian_confidence1 = pedestrian_confidence1[1:]

    vehicle_confidence1 = np.zeros([n_vehicle])
    # t8 = time()
    # np.maximum.at(
    # vehicle_confidence1,
    # np.reshape(vehicle_ccs, [-1]),
    # np.reshape(fy[..., 1], [-1]),
    # )
    vehicle_confidence1 = tf.tensor_scatter_nd_max(
        vehicle_confidence1,
        np.reshape(vehicle_ccs, [-1, 1]),
        np.reshape(vehicle_fy, [-1]),
    ).numpy()
    vehicle_centroid1 = vehicle_centroid1[1:]
    vehicle_confidence1 = vehicle_confidence1[1:]
    # t9 = time()
    # print(pedestrian_centroid1.shape[0], vehicle_centroid1.shape[0])
    # print(
    # "post",
    # t1 - t0,
    # t2 - t1,
    # t3 - t2,
    # t4 - t3,
    # t5 - t4,
    # t6 - t5,
    # t7 - t6,
    # t8 - t7,
    # t9 - t8,
    # )
    # print("first", pedestrian_centroid1.shape, vehicle_centroid1.shape)

    # return (
    # pedestrian_centroid1,
    # pedestrian_confidence1,
    # vehicle_centroid1,
    # vehicle_confidence1,
    # )

    pedestrian_fy = y[..., 0]
    vehicle_fy = y[..., 1]
    pedestrian_m = np.logical_and(pedestrian_fy <= 0.29, pedestrian_fy > 0.01)
    vehicle_m = np.logical_and(vehicle_fy <= 0.54, vehicle_fy > 0.2)
    pedestrian_fy[np.logical_not(pedestrian_m)] = 0
    vehicle_fy[np.logical_not(vehicle_m)] = 0

    (
        n_pedestrian,
        pedestrian_ccs,
        _,
        pedestrian_centroid,
    ) = cv2.connectedComponentsWithStats(
        pedestrian_m[0].astype(np.uint8), connectivity=4
    )
    # t4 = time()
    pedestrian_centroid2 = pedestrian_centroid[:, [1, 0]]
    (
        n_vehicle,
        vehicle_ccs,
        _,
        vehicle_centroid,
    ) = cv2.connectedComponentsWithStats(vehicle_m[0].astype(np.uint8), connectivity=4)

    # total_pedestrian += n_pedestrian
    # total_vehicle += n_vehicle

    # if total_pedestrian > 100:
    # if total_vehicle > 100:
    # m = np.stack([y[..., 0] > 0.05, y[..., 1] > 0.4], -1)
    # else:
    # m = np.stack([y[..., 0] > 0.05, y[..., 1] > 0.2], -1)
    # else:
    # if total_vehicle > 100:
    # m = np.stack([y[..., 0] > 0.01, y[..., 1] > 0.4], -1)
    # else:
    # m = np.stack([y[..., 0] > 0.01, y[..., 1] > 0.2], -1)

    vehicle_centroid2 = vehicle_centroid[:, [1, 0]]
    pedestrian_confidence2 = np.zeros([n_pedestrian])

    pedestrian_confidence2 = tf.tensor_scatter_nd_max(
        pedestrian_confidence2,
        np.reshape(pedestrian_ccs, [-1, 1]),
        np.reshape(pedestrian_fy, [-1]),
    ).numpy()
    # t7 = time()
    pedestrian_centroid2 = pedestrian_centroid2[1:]
    pedestrian_confidence2 = pedestrian_confidence2[1:]  # * 0.5

    vehicle_confidence2 = np.zeros([n_vehicle])
    vehicle_confidence2 = tf.tensor_scatter_nd_max(
        vehicle_confidence2,
        np.reshape(vehicle_ccs, [-1, 1]),
        np.reshape(vehicle_fy, [-1]),
    ).numpy()
    vehicle_centroid2 = vehicle_centroid2[1:]
    vehicle_confidence2 = vehicle_confidence2[1:]  # * 0.5

    pedestrian_centroid = np.concatenate(
        [pedestrian_centroid1, pedestrian_centroid2], 0
    )
    vehicle_centroid = np.concatenate([vehicle_centroid1, vehicle_centroid2], 0)
    pedestrian_confidence = np.concatenate(
        [pedestrian_confidence1, pedestrian_confidence2], 0
    )
    vehicle_confidence = np.concatenate([vehicle_confidence1, vehicle_confidence2], 0)
    # print("second", pedestrian_centroid.shape, vehicle_centroid.shape)
    return (
        pedestrian_centroid,
        pedestrian_confidence,
        vehicle_centroid,
        vehicle_confidence,
    )


# @tf.function
def predict_tf(bev_model, lidar_points):
    # t0 = time()
    input_image = preprocess(lidar_points, 3.6)
    input_image2 = preprocess(lidar_points, 3.1)
    # t1 = time()
    summary = False
    if summary:
        input_summary_image = np.max(input_image.numpy()[..., :-1], -1)[0][
            :, :, np.newaxis
        ]
        input_summary_image = np.tile(input_summary_image, (1, 1, 3))
    # print("lidar maen", tf.reduce_mean(input_image))
    preds = bev_model(
        tf.concat(
            [
                input_image,
                input_image2[:, :, ::-1, :],
                # # input_image[:, :, ::-1, :],
                input_image[:, ::-1, :, :],
                # input_image2[:, ::-1, ::-1, :],  #
                # tf.transpose(input_image, (0, 2, 1, 3)),
            ],
            0,
        ),
    )["detector"]
    pred = (
        preds[:1]
        + preds[1:2, :, ::-1, :]
        # # + preds[1:2, :, ::-1, :]
        + preds[2:3, ::-1, :, :]
        # + preds[3:4, ::-1, ::-1, :]  #
        # + tf.transpose(preds[3:4], (0, 2, 1, 3))
    ) / preds.shape[0]
    pred = pred.numpy()
    # t2 = time()
    # print(pred)
    (
        pedestrian_centroid,
        pedestrian_confidence,
        vehicle_centroid,
        vehicle_confidence,
    ) = postprocess(pred)
    # t3 = time()
    # print(t1 - t0, t2 - t1, t3 - t2)
    if summary:
        pred_summary_image = np.concatenate(
            [pred[0].numpy(), np.zeros((1152, 1152, 1), np.float32)], -1
        )
        summary_image = (
            np.concatenate([input_summary_image, pred_summary_image], 1) * 255
        )
        summary_image = summary_image.astype(np.uint8)
        cv2.imwrite("./summary.png", summary_image)
        exit()

    return (
        pedestrian_centroid,
        pedestrian_confidence,
        vehicle_centroid,
        vehicle_confidence,
    )


def predict(bev_model, lidar, ego, cam_calib, lidar_calib):

    lidar_points = tf.cast(lidar, tf.float32)
    (
        pedestrian_centroid,
        pedestrian_confidence,
        vehicle_centroid,
        vehicle_confidence,
    ) = predict_tf(bev_model, lidar_points)

    pedestrian_centers = []
    vehicle_centers = []
    for n in range(pedestrian_centroid.shape[0]):
        pedestrian_centers.append(
            np.append(
                # pedestrian_centroid[n].numpy(), [0, pedestrian_confidence[n].numpy()]
                pedestrian_centroid[n],
                [0, pedestrian_confidence[n]],
            )
        )

    for n in range(vehicle_centroid.shape[0]):
        vehicle_centers.append(
            # np.append(vehicle_centroid[n].numpy(), [0, vehicle_confidence[n].numpy()])
            np.append(vehicle_centroid[n], [0, vehicle_confidence[n]])
        )

    ego_txyz = tf.cast(ego["translation"], np.float32)
    ego_qt = quaternion.as_quat_array(ego["rotation"])

    if len(pedestrian_centers) > 0:
        pedestrian_xyz = np.array(pedestrian_centers)
        pedestrian_xyz[:, :2] = pedestrian_xyz[:, :2] / 10 - 51.2
        pedestrian_xyz[:, 2] = pedestrian_xyz[:, 2] / 10 - 3.5 + 2
        pedestrian_xyz[:, [0, 1]] = pedestrian_xyz[:, [1, 0]]
        pedestrian_xyz[:, 1] = -pedestrian_xyz[:, 1]
        pedestrian_xyz[:, :3] = quaternion.rotate_vectors(ego_qt, pedestrian_xyz[:, :3])
        pedestrian_xyz[:, :3] = pedestrian_xyz[:, :3] + ego_txyz[np.newaxis]
        pedestrian_xyz = list(pedestrian_xyz)
    else:
        pedestrian_xyz = []

    if len(vehicle_centers) > 0:
        vehicle_xyz = np.array(vehicle_centers)
        vehicle_xyz[:, :2] = vehicle_xyz[:, :2] / 10 - 51.2
        vehicle_xyz[:, 2] = vehicle_xyz[:, 2] / 10 - 3.5 + 2
        vehicle_xyz[:, [0, 1]] = vehicle_xyz[:, [1, 0]]
        vehicle_xyz[:, 1] = -vehicle_xyz[:, 1]
        vehicle_xyz[:, :3] = quaternion.rotate_vectors(ego_qt, vehicle_xyz[:, :3])
        vehicle_xyz[:, :3] = vehicle_xyz[:, :3] + ego_txyz[np.newaxis]
        vehicle_xyz = list(vehicle_xyz)
    else:
        vehicle_xyz = []

    return pedestrian_xyz, vehicle_xyz
