import os
import quaternion
import numpy as np
import cv2
import logging
import json

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
def preprocess(lidar_points):
    lidar_xs = tf.cast(tf.math.round((lidar_points[:, 0] + 57.6) * 10), tf.int32)
    lidar_ys = tf.cast(tf.math.round((-lidar_points[:, 1] + 57.6) * 10), tf.int32)
    lidar_zs = tf.cast(tf.math.round((lidar_points[:, 2] + 3.5) * 10), tf.int32)
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


@tf.function
def postprocess(pred):
    y = pred[:, 64:-64, 64:-64, :]
    y = tf.concat([y, tf.zeros((*y.shape[:3], 1))], -1)
    m = tf.stack([y[..., 0] > 0.2, y[..., 1] > 0.68, y[..., 2] > 0], -1)
    fy = y * tf.cast(m, tf.float32)
    pedestrian_ccs = tfa.image.connected_components(m[..., 0])
    vehicle_ccs = tfa.image.connected_components(m[..., 1])

    n_pedestrian = tf.math.minimum(tf.reduce_max(pedestrian_ccs), 100)

    pedestrian_cc = tf.tile(pedestrian_ccs, (n_pedestrian, 1, 1)) == tf.reshape(
        tf.range(1, n_pedestrian + 1, dtype=tf.int32), (-1, 1, 1)
    )
    pedestrian_filter = (
        tf.reduce_sum(tf.cast(pedestrian_cc, tf.int32), (1, 2), keepdims=True) > 0
    )
    area = tf.reduce_sum(tf.cast(pedestrian_filter, tf.int32), (1, 2)) > 0

    pedestrian_cc = tf.boolean_mask(pedestrian_cc, area)
    pedestrian_filter = tf.boolean_mask(pedestrian_filter, area)

    pedestrian_cc = tf.math.logical_and(pedestrian_cc, pedestrian_filter)
    xs, ys = tf.meshgrid(tf.range(m.shape[2]), tf.range(m.shape[1]))
    yxs = tf.stack([ys, xs], -1)
    yxs = tf.tile(tf.expand_dims(yxs, 0), (tf.shape(pedestrian_cc)[0], 1, 1, 1))
    pedestrian_centroid = yxs * tf.expand_dims(tf.cast(pedestrian_cc, tf.int32), -1)
    total = tf.reduce_sum(tf.cast(pedestrian_cc, tf.float32), (1, 2))
    pedestrian_centroid1 = tf.reduce_sum(
        tf.cast(pedestrian_centroid, tf.float32), (1, 2)
    ) / tf.expand_dims(total, -1)
    pedestrian_confidence1 = tf.reduce_max(
        fy[..., 0] * tf.cast(pedestrian_cc, tf.float32), (1, 2)
    )  # / total

    n_vehicle = tf.math.minimum(tf.reduce_max(vehicle_ccs), 100)

    vehicle_cc = tf.tile(vehicle_ccs, (n_vehicle, 1, 1)) == tf.reshape(
        tf.range(1, n_vehicle + 1, dtype=tf.int32), (-1, 1, 1)
    )
    vehicle_filter = (
        tf.reduce_sum(tf.cast(vehicle_cc, tf.int32), (1, 2), keepdims=True) > 0
    )
    area = tf.reduce_sum(tf.cast(vehicle_filter, tf.int32), (1, 2)) > 0

    vehicle_cc = tf.boolean_mask(vehicle_cc, area)
    vehicle_filter = tf.boolean_mask(vehicle_filter, area)

    vehicle_cc = tf.math.logical_and(vehicle_cc, vehicle_filter)
    xs, ys = tf.meshgrid(tf.range(m.shape[2]), tf.range(m.shape[1]))
    yxs = tf.stack([ys, xs], -1)
    yxs = tf.tile(tf.expand_dims(yxs, 0), (tf.shape(vehicle_cc)[0], 1, 1, 1))
    vehicle_centroid = yxs * tf.expand_dims(tf.cast(vehicle_cc, tf.int32), -1)
    total = tf.reduce_sum(tf.cast(vehicle_cc, tf.float32), (1, 2))
    vehicle_centroid1 = tf.reduce_sum(
        tf.cast(vehicle_centroid, tf.float32), (1, 2)
    ) / tf.expand_dims(total, -1)
    vehicle_confidence1 = tf.reduce_max(
        fy[..., 1] * tf.cast(vehicle_cc, tf.float32), (1, 2)
    )  # / total

    m = tf.stack([y[..., 0] <= 0.15, y[..., 1] <= 0.58, y[..., 2] > 0], -1)
    y = y * tf.cast(m, tf.float32)
    y = tfa.image.gaussian_filter2d(y)
    y = tfa.image.gaussian_filter2d(y)
    y = tfa.image.gaussian_filter2d(y)
    m = tf.stack([y[..., 0] > 0.01, y[..., 1] > 0.2, y[..., 2] > 0], -1)
    fy = y * tf.cast(m, tf.float32)
    pedestrian_ccs = tfa.image.connected_components(m[..., 0])
    vehicle_ccs = tfa.image.connected_components(m[..., 1])

    n_pedestrian = tf.math.minimum(tf.reduce_max(pedestrian_ccs), 100)

    pedestrian_cc = tf.tile(pedestrian_ccs, (n_pedestrian, 1, 1)) == tf.reshape(
        tf.range(1, n_pedestrian + 1, dtype=tf.int32), (-1, 1, 1)
    )
    pedestrian_filter = (
        tf.reduce_sum(tf.cast(pedestrian_cc, tf.int32), (1, 2), keepdims=True) > 0
    )
    area = tf.reduce_sum(tf.cast(pedestrian_filter, tf.int32), (1, 2)) > 0

    pedestrian_cc = tf.boolean_mask(pedestrian_cc, area)
    pedestrian_filter = tf.boolean_mask(pedestrian_filter, area)

    pedestrian_cc = tf.math.logical_and(pedestrian_cc, pedestrian_filter)
    xs, ys = tf.meshgrid(tf.range(m.shape[2]), tf.range(m.shape[1]))
    yxs = tf.stack([ys, xs], -1)
    yxs = tf.tile(tf.expand_dims(yxs, 0), (tf.shape(pedestrian_cc)[0], 1, 1, 1))
    pedestrian_centroid = yxs * tf.expand_dims(tf.cast(pedestrian_cc, tf.int32), -1)
    total = tf.reduce_sum(tf.cast(pedestrian_cc, tf.float32), (1, 2))
    pedestrian_centroid2 = tf.reduce_sum(
        tf.cast(pedestrian_centroid, tf.float32), (1, 2)
    ) / tf.expand_dims(total, -1)
    pedestrian_confidence2 = (
        tf.reduce_max(fy[..., 0] * tf.cast(pedestrian_cc, tf.float32), (1, 2)) * 0.5
    )  # / total

    n_vehicle = tf.math.minimum(tf.reduce_max(vehicle_ccs), 100)

    vehicle_cc = tf.tile(vehicle_ccs, (n_vehicle, 1, 1)) == tf.reshape(
        tf.range(1, n_vehicle + 1, dtype=tf.int32), (-1, 1, 1)
    )
    vehicle_filter = (
        tf.reduce_sum(tf.cast(vehicle_cc, tf.int32), (1, 2), keepdims=True) > 0
    )
    area = tf.reduce_sum(tf.cast(vehicle_filter, tf.int32), (1, 2)) > 0

    vehicle_cc = tf.boolean_mask(vehicle_cc, area)
    vehicle_filter = tf.boolean_mask(vehicle_filter, area)

    vehicle_cc = tf.math.logical_and(vehicle_cc, vehicle_filter)
    xs, ys = tf.meshgrid(tf.range(m.shape[2]), tf.range(m.shape[1]))
    yxs = tf.stack([ys, xs], -1)
    yxs = tf.tile(tf.expand_dims(yxs, 0), (tf.shape(vehicle_cc)[0], 1, 1, 1))
    vehicle_centroid = yxs * tf.expand_dims(tf.cast(vehicle_cc, tf.int32), -1)
    total = tf.reduce_sum(tf.cast(vehicle_cc, tf.float32), (1, 2))
    vehicle_centroid2 = tf.reduce_sum(
        tf.cast(vehicle_centroid, tf.float32), (1, 2)
    ) / tf.expand_dims(total, -1)
    vehicle_confidence2 = (
        tf.reduce_max(fy[..., 1] * tf.cast(vehicle_cc, tf.float32), (1, 2)) * 0.5
    )  # / total
    pedestrian_centroid = tf.concat([pedestrian_centroid1, pedestrian_centroid2], 0)
    vehicle_centroid = tf.concat([vehicle_centroid1, vehicle_centroid2], 0)
    pedestrian_confidence = tf.concat(
        [pedestrian_confidence1, pedestrian_confidence2], 0
    )
    vehicle_confidence = tf.concat([vehicle_confidence1, vehicle_confidence2], 0)
    return (
        pedestrian_centroid,
        pedestrian_confidence,
        vehicle_centroid,
        vehicle_confidence,
    )


# @tf.function
def predict_tf(bev_model, lidar_points):
    input_image = preprocess(lidar_points)
    preds = bev_model(
        tf.concat(
            [
                input_image,
                input_image[:, :, ::-1, :],
                input_image[:, ::-1, :, :],
                input_image[:, ::-1, ::-1, :],  #
                tf.transpose(input_image, (0, 2, 1, 3)),
            ],
            0,
        )
    )["output_0"]
    pred = (
        preds[:1]
        + preds[1:2, :, ::-1, :]
        + preds[2:3, ::-1, :, :]
        + preds[3:4, ::-1, ::-1, :]  #
        + tf.transpose(preds[4:5], (0, 2, 1, 3))
    ) / preds.shape[0]
    (
        pedestrian_centroid,
        pedestrian_confidence,
        vehicle_centroid,
        vehicle_confidence,
    ) = postprocess(pred)

    return (
        pedestrian_centroid,
        pedestrian_confidence,
        vehicle_centroid,
        vehicle_confidence,
    )


def predict(bev_model, cam_image, lidar, ego, cam_calib, lidar_calib):

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
                pedestrian_centroid[n].numpy(), [0, pedestrian_confidence[n].numpy()]
            )
        )

    for n in range(vehicle_centroid.shape[0]):
        vehicle_centers.append(
            np.append(vehicle_centroid[n].numpy(), [0, vehicle_confidence[n].numpy()])
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
