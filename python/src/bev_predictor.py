import os
import quaternion
import numpy as np
import cv2
import logging

from time import time

os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
import tensorflow as tf
import tensorflow_addons as tfa


logging.getLogger("tensorflow").setLevel(logging.ERROR)


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, 5), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    ]
)
def preprocess(lidar_points, z_offset=3.7):
    lidar_xs = tf.cast(tf.math.round((lidar_points[:, 0]) * 10), tf.int32) + 576
    lidar_ys = tf.cast(tf.math.round((-lidar_points[:, 1]) * 10), tf.int32) + 576
    lidar_zs = tf.cast(tf.math.round((lidar_points[:, 2] + z_offset) * 5), tf.int32)
    indices = tf.math.logical_and(
        tf.math.logical_and(
            tf.math.logical_and(lidar_xs >= 0, lidar_xs < 1152),
            tf.math.logical_and(lidar_ys >= 0, lidar_ys < 1152),
        ),
        tf.math.logical_and(lidar_zs >= 0, lidar_zs < 24),
    )

    lidar_xs = tf.boolean_mask(lidar_xs, indices)
    lidar_ys = tf.boolean_mask(lidar_ys, indices)
    lidar_zs = tf.boolean_mask(lidar_zs, indices)
    intensities = tf.boolean_mask(lidar_points[:, 3], indices)

    lidar_image = tf.zeros((1152, 1152, 24), tf.float32)
    lidar_image = tf.tensor_scatter_nd_max(
        lidar_image, tf.stack([lidar_ys, lidar_xs, lidar_zs], -1), intensities
    )

    input_image = tf.expand_dims(lidar_image, 0)
    return input_image


def cc_tf(image):
    (n, ccs, _stats, centroid,) = cv2.connectedComponentsWithStatsWithAlgorithm(
        image,
        connectivity=4,
        ltype=cv2.CV_32S,
        ccltype=cv2.CCL_GRANA,
    )
    return n, ccs, centroid[:, [1, 0]]


@tf.function(
    input_signature=[tf.TensorSpec(shape=(1, 1152, 1152, 2), dtype=tf.float32)],
)
def postprocess_tf(pred):
    y = pred[:, 64:-64, 64:-64, :]
    pedestrian_fy = y[..., 0]
    vehicle_fy = y[..., 1]
    pedestrian_m = pedestrian_fy > 0.33
    vehicle_m = vehicle_fy > 0.52

    p_m = tf.pad(pedestrian_m, ((0, 0), (1, 1), (1, 1)))
    p_m = tf.math.logical_or(
        tf.math.logical_or(tf.roll(p_m, 1, 1), tf.roll(p_m, -1, 1)),
        tf.math.logical_or(tf.roll(p_m, 1, 2), tf.roll(p_m, -1, 2)),
    )[:, 1:-1, 1:-1]
    v_m = tf.pad(vehicle_m, ((0, 0), (1, 1), (1, 1)))
    v_m = tf.math.logical_or(
        tf.math.logical_or(tf.roll(v_m, 1, 1), tf.roll(v_m, -1, 1)),
        tf.math.logical_or(tf.roll(v_m, 1, 2), tf.roll(v_m, -1, 2)),
    )[:, 1:-1, 1:-1]
    pedestrian_m = tf.math.logical_or(
        pedestrian_m,
        tf.math.logical_and(
            tf.math.logical_not(p_m),
            tf.math.logical_and(pedestrian_fy <= 0.30, pedestrian_fy > 0.005),
        ),
    )
    vehicle_m = tf.math.logical_or(
        vehicle_m,
        tf.logical_and(
            tf.logical_not(v_m),
            tf.math.logical_and(vehicle_fy <= 0.40, vehicle_fy > 0.2),
        ),
    )
    (n_pedestrian, pedestrian_ccs, pedestrian_centroid) = tf.numpy_function(
        cc_tf,
        [tf.cast(pedestrian_m[0], tf.uint8)],
        Tout=[tf.int64, tf.int32, tf.float64],
    )
    pedestrian_centroid = tf.cast(pedestrian_centroid, tf.float32)

    (n_vehicle, vehicle_ccs, vehicle_centroid) = tf.numpy_function(
        cc_tf,
        [tf.cast(vehicle_m[0], tf.uint8)],
        Tout=[tf.int64, tf.int32, tf.float64],
    )
    vehicle_centroid = tf.cast(vehicle_centroid, tf.float32)
    pedestrian_confidence = tf.zeros([n_pedestrian])
    pedestrian_confidence = tf.tensor_scatter_nd_max(
        pedestrian_confidence,
        tf.reshape(pedestrian_ccs, [-1, 1]),
        tf.reshape(pedestrian_fy, [-1]),
    )
    pedestrian_centroid = pedestrian_centroid[1:]
    pedestrian_confidence = pedestrian_confidence[1:]

    vehicle_confidence = tf.zeros([n_vehicle])
    vehicle_confidence = tf.tensor_scatter_nd_max(
        vehicle_confidence,
        tf.reshape(vehicle_ccs, [-1, 1]),
        tf.reshape(vehicle_fy, [-1]),
    )
    vehicle_centroid = vehicle_centroid[1:]
    vehicle_confidence = vehicle_confidence[1:]

    return (
        pedestrian_centroid,
        pedestrian_confidence,
        vehicle_centroid,
        vehicle_confidence,
    )


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(1024, 1024, 2), dtype=tf.float32),
        tf.TensorSpec(shape=(1024, 1024, 2), dtype=tf.float32),
        tf.TensorSpec(shape=(1024, 1024, 2), dtype=tf.float32),
    ],
)
def merge3(x1, x2, x3):
    m = tf.stack([x1, x2, x3], -1)
    return tf.math.maximum(
        x1, tf.reduce_sum(m, -1) - tf.reduce_max(m, -1) - tf.reduce_min(m, -1)
    )


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(1024, 1024, 2), dtype=tf.float32),
        tf.TensorSpec(shape=(1024, 1024, 2), dtype=tf.float32),
        tf.TensorSpec(shape=(1024, 1024, 2), dtype=tf.float32),
        tf.TensorSpec(shape=[8], dtype=tf.float32),
        tf.TensorSpec(shape=[8], dtype=tf.float32),
    ],
)
def merge3ex(x0, x1, x2, r1, r2):
    y1 = tfa.image.transform(
        x1,
        r1,
        interpolation="bilinear",
    )

    y2 = tfa.image.transform(
        x2,
        r2,
        interpolation="bilinear",
    )
    y = merge3(x0, y1, y2)
    return y


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(1024, 1024), dtype=tf.float32),
        tf.TensorSpec(shape=(1024, 1024), dtype=tf.float32),
    ],
    jit_compile=True,
)
def get_mask(pedestrian_fy, vehicle_fy):
    pedestrian_m = pedestrian_fy > 0.30
    vehicle_m = vehicle_fy > 0.21

    p_m = tf.pad(pedestrian_m, ((1, 1), (1, 1)))
    p_m = tf.math.logical_or(
        tf.math.logical_or(tf.roll(p_m, 1, 0), tf.roll(p_m, -1, 0)),
        tf.math.logical_or(tf.roll(p_m, 1, 1), tf.roll(p_m, -1, 1)),
    )[1:-1, 1:-1]
    v_m = tf.pad(vehicle_m, ((4, 4), (4, 4)))
    v_m = tf.math.logical_or(
        tf.math.logical_or(tf.roll(v_m, 4, 0), tf.roll(v_m, -4, 0)),
        tf.math.logical_or(tf.roll(v_m, 4, 1), tf.roll(v_m, -4, 1)),
    )[4:-4, 4:-4]
    pedestrian_m = tf.math.logical_or(
        pedestrian_m,
        tf.math.logical_and(
            tf.math.logical_not(p_m),
            tf.math.logical_and(pedestrian_fy <= 0.30, pedestrian_fy > 0.009),
        ),
    )
    vehicle_m = tf.math.logical_or(
        vehicle_m,
        tf.math.logical_and(
            tf.math.logical_not(v_m),
            tf.math.logical_and(vehicle_fy <= 0.21, vehicle_fy > 0.07),
        ),
    )

    return tf.cast(pedestrian_m, tf.uint8), tf.cast(vehicle_m, tf.uint8)


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.float32),
        tf.TensorSpec(shape=[None], dtype=tf.float32),
        tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.float32),
        tf.TensorSpec(shape=[None], dtype=tf.float32),
        tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
    ],
)
def refine(
    pedestrian_ccs,
    pedestrian_fy,
    pedestrian_area,
    pedestrian_centroid,
    vehicle_ccs,
    vehicle_fy,
    vehicle_area,
    vehicle_centroid,
):
    pedestrian_confidence = tf.zeros([tf.shape(pedestrian_centroid)[0]])
    pedestrian_confidence = tf.tensor_scatter_nd_max(
        pedestrian_confidence,
        pedestrian_ccs,
        pedestrian_fy,
    )
    additional_indices = tf.math.logical_and(
        pedestrian_area[1:] > 77, pedestrian_confidence[1:] > 0.30
    )
    additional_centroid = tf.boolean_mask(pedestrian_centroid[1:], additional_indices)
    additional_confidence = tf.boolean_mask(
        pedestrian_confidence[1:], additional_indices
    )  # * 0.99
    pedestrian_centroid = pedestrian_centroid[1:]
    pedestrian_confidence = pedestrian_confidence[1:]
    pedestrian_centroid = tf.cond(
        tf.size(additional_centroid) > 0,
        lambda: tf.concat([pedestrian_centroid, additional_centroid], 0),
        lambda: pedestrian_centroid,
    )
    pedestrian_confidence = tf.cond(
        tf.size(additional_confidence) > 0,
        lambda: tf.concat([pedestrian_confidence, additional_confidence], 0),
        lambda: pedestrian_confidence,
    )

    vehicle_confidence = tf.zeros([tf.shape(vehicle_centroid)[0]])
    vehicle_confidence = tf.tensor_scatter_nd_max(
        vehicle_confidence,
        vehicle_ccs,
        vehicle_fy,
    )
    vehicle_centroid = vehicle_centroid[1:]
    vehicle_confidence = vehicle_confidence[1:]
    return (
        pedestrian_centroid,
        pedestrian_confidence,
        vehicle_centroid,
        vehicle_confidence,
    )


def postprocess(pred, ego_pose, ego_pose_records, pred_records):
    # t0 = time()
    y = pred[0, 64:-64, 64:-64, :]  # .copy()
    # t1 = time()

    if len(ego_pose_records) >= 2:
        txyz0 = -np.array(ego_pose_records[-1]["translation"]) + np.array(
            ego_pose["translation"]
        )
        txyz1 = -np.array(ego_pose_records[-2]["translation"]) + np.array(
            ego_pose["translation"]
        )
        qt0 = (
            quaternion.as_quat_array(ego_pose_records[-1]["rotation"])
            * quaternion.as_quat_array(ego_pose["rotation"]).inverse()
        ).inverse()
        qt1 = (
            quaternion.as_quat_array(ego_pose_records[-2]["rotation"])
            * quaternion.as_quat_array(ego_pose["rotation"]).inverse()
        ).inverse()
        rot0 = quaternion.as_rotation_matrix(qt0)
        rot1 = quaternion.as_rotation_matrix(qt1)
        cos0 = rot0[0, 0]
        sin0 = rot0[1, 0]
        cos1 = rot1[0, 0]
        sin1 = rot1[1, 0]
        a0 = np.arctan2(sin0, cos0) * 180 / np.pi
        a1 = np.arctan2(sin1, cos1) * 180 / np.pi
        use_tf = True
        if use_tf:
            r0 = cv2.getRotationMatrix2D((512, 512), a0, 1)
            r0[0, 2] = r0[0, 2] + txyz0[0] * 10
            r0[1, 2] = r0[1, 2] - txyz0[1] * 10

            r1 = cv2.getRotationMatrix2D((512, 512), a1, 1)
            r1[0, 2] = r1[0, 2] + txyz1[0] * 10
            r1[1, 2] = r1[1, 2] - txyz1[1] * 10

            y = merge3ex(
                y,
                pred_records[-1][0, 64:-64, 64:-64, :],
                pred_records[-2][0, 64:-64, 64:-64, :],
                [r0[0, 0], r0[0, 1], r0[0, 2], r0[1, 0], r0[1, 1], r0[1, 2], 0, 0],
                [r1[0, 0], r1[0, 1], r1[0, 2], r1[1, 0], r1[1, 1], r1[1, 2], 0, 0],
            )
        else:
            r0 = cv2.getRotationMatrix2D((512, 512), -a0, 1)
            r0[0, 2] = r0[0, 2] - txyz0[0] * 10
            r0[1, 2] = r0[1, 2] + txyz0[1] * 10
            y0 = cv2.warpAffine(
                pred_records[-1][0, 64:-64, 64:-64, :],
                r0,
                (1024, 1024),
            )

            r1 = cv2.getRotationMatrix2D((512, 512), -a1, 1)
            r1[0, 2] = r1[0, 2] - txyz1[0] * 10
            r1[1, 2] = r1[1, 2] + txyz1[1] * 10
            y1 = cv2.warpAffine(
                pred_records[-2][0, 64:-64, 64:-64, :],
                r1,
                (1024, 1024),
            )
            y = merge3(y, y0, y1)  # .numpy()
    # t2 = time()

    pedestrian_fy = y[..., 0]
    vehicle_fy = y[..., 1]
    pedestrian_m, vehicle_m = get_mask(pedestrian_fy, vehicle_fy)
    (
        n_pedestrian,
        pedestrian_ccs,
        _pedestrian_stats,
        pedestrian_centroid,
    ) = cv2.connectedComponentsWithStatsWithAlgorithm(
        pedestrian_m.numpy(),
        connectivity=4,
        ltype=cv2.CV_32S,
        ccltype=cv2.CCL_GRANA,
    )
    # t4 = time()
    (
        n_vehicle,
        vehicle_ccs,
        _vehicle_stats,
        vehicle_centroid,
    ) = cv2.connectedComponentsWithStatsWithAlgorithm(
        vehicle_m.numpy(),
        connectivity=4,
        ltype=cv2.CV_32S,
        ccltype=cv2.CCL_GRANA,
    )
    # t5 = time()

    pedestrian_centroid = pedestrian_centroid[:, [1, 0]]
    vehicle_centroid = vehicle_centroid[:, [1, 0]]

    use_tf = False
    if use_tf:
        (
            pedestrian_centroid,
            pedestrian_confidence,
            vehicle_centroid,
            vehicle_confidence,
        ) = refine(
            np.reshape(pedestrian_ccs, [-1, 1]),
            np.reshape(pedestrian_fy, [-1]),
            _pedestrian_stats[:, -1],
            pedestrian_centroid,
            np.reshape(vehicle_ccs, [-1, 1]),
            np.reshape(vehicle_fy, [-1]),
            _vehicle_stats[:, -1],
            vehicle_centroid,
        )
        pedestrian_centroid = pedestrian_centroid.numpy()
        pedestrian_confidence = pedestrian_confidence.numpy()
        vehicle_centroid = vehicle_centroid.numpy()
        vehicle_confidence = vehicle_confidence.numpy()
    else:
        pedestrian_confidence = tf.zeros([n_pedestrian])
        pedestrian_confidence = tf.tensor_scatter_nd_max(
            pedestrian_confidence,
            np.reshape(pedestrian_ccs, [-1, 1]),
            np.reshape(pedestrian_fy, [-1]),
        ).numpy()
        additional_indices = np.logical_and(
            _pedestrian_stats[1:, -1] > 78, pedestrian_confidence[1:] > 0.29
        )
        additional_centroid = pedestrian_centroid[1:][additional_indices]
        additional_confidence = pedestrian_confidence[1:][additional_indices]  # * 0.99
        pedestrian_centroid = pedestrian_centroid[1:]
        pedestrian_confidence = pedestrian_confidence[1:]
        if len(additional_centroid) > 0:
            pedestrian_centroid = np.concatenate(
                [pedestrian_centroid, additional_centroid], 0
            )
            pedestrian_confidence = np.concatenate(
                [pedestrian_confidence, additional_confidence], 0
            )

        vehicle_confidence = tf.zeros([n_vehicle])
        vehicle_confidence = tf.tensor_scatter_nd_max(
            vehicle_confidence,
            np.reshape(vehicle_ccs, [-1, 1]),
            np.reshape(vehicle_fy, [-1]),
        ).numpy()
        vehicle_centroid = vehicle_centroid[1:]
        vehicle_confidence = vehicle_confidence[1:]
    # t6 = time()
    # print("pp", t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5)
    # print(pedestrian_centroid1.shape[0], vehicle_centroid1.shape[0])

    return (
        pedestrian_centroid,
        pedestrian_confidence,
        vehicle_centroid,
        vehicle_confidence,
    )


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, 5), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    ]
)
def preprocess3(lidar_points, z_offset1, z_offset2, z_offset3):
    input_image1 = preprocess(lidar_points, z_offset1)
    input_image2 = preprocess(lidar_points, z_offset2)
    input_image3 = preprocess(lidar_points, z_offset3)
    return input_image1, input_image2, input_image3


# @tf.function
def predict_tf(
    bev_model, lidar_points, ego_pose, ego_pose_records, pred_records, summary=False
):
    t0 = time()
    # input_image = preprocess(lidar_points, 3.6)
    # input_image1 = preprocess(lidar_points, 4.4)
    # input_image2 = preprocess(lidar_points, 3.7)
    # input_image3 = preprocess(lidar_points, 3.0)
    input_image1, input_image2, input_image3 = preprocess3(lidar_points, 4.4, 3.7, 3.0)
    t1 = time()
    if summary:
        input_summary_image1 = np.max(input_image1.numpy()[..., :-1], -1)[0][
            :, :, np.newaxis
        ]
        input_summary_image2 = np.max(input_image2.numpy()[..., :-1], -1)[0][
            :, :, np.newaxis
        ]
        input_summary_image = np.maximum(input_summary_image1, input_summary_image2)
        input_summary_image = np.tile(input_summary_image, (1, 1, 3))
    preds = bev_model(
        tf.concat(
            [
                # input_image,
                input_image1[:, :, ::-1, :],
                # # input_image[:, :, ::-1, :],
                input_image2[:, ::-1, :, :],
                input_image3[:, :, :, :],  #
                # tf.transpose(input_image, (0, 2, 1, 3)),
            ],
            0,
        ),
    )["detector"]
    t2 = time()
    pred = (
        # preds[:1]
        preds[0:1, :, ::-1, :]
        # # + preds[1:2, :, ::-1, :]
        + preds[1:2, ::-1, :, :]
        + preds[2:3, :, :, :]  #
        # + tf.transpose(preds[3:4], (0, 2, 1, 3))
    ) / preds.shape[0]
    pred1 = tf.reduce_max(
        tf.stack(
            [
                # preds[:1],
                preds[0:1, :, ::-1, :],
                preds[1:2, ::-1, :, :],
                preds[2:3, :, :, :],
            ],
            -1,
        ),
        -1,
    )
    t3 = time()
    pred = pred.numpy()
    t4 = time()
    # pred[..., 0] = (pred[..., 0] * 5 + pred1[..., 0]) / 6
    pred[..., 1] = (pred[..., 1] + pred1[..., 1]) / 2
    t5 = time()
    # print(pred)
    (
        pedestrian_centroid,
        pedestrian_confidence,
        vehicle_centroid,
        vehicle_confidence,
    ) = postprocess(pred, ego_pose, ego_pose_records, pred_records)

    suppress_predictions_without_lidar_points = False
    if suppress_predictions_without_lidar_points:
        input_intensities = tf.math.maximum(
            input_image1, tf.math.maximum(input_image2, input_image3)
        )
        for n in range(pedestrian_centroid.shape[0]):
            y, x = pedestrian_centroid[n]
            y, x = round(y) + 64, round(x) + 64
            points = tf.reduce_sum(
                input_intensities[:, y - 7 : y + 7, x - 7 : x + 7, :]
            )
            if points == 0:
                pedestrian_confidence[n] = pedestrian_confidence[n] * 0.1

    pred_records.append(pred)

    t6 = time()
    # print(t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5)
    if summary:
        pred_summary_image = np.concatenate(
            [pred[0], np.zeros((1152, 1152, 1), np.float32)], -1
        )
        pred_summary_image = np.maximum(pred_summary_image, input_summary_image)
        summary_image = (
            np.concatenate([input_summary_image, pred_summary_image], 1) * 255
        )
        summary_image = summary_image.astype(np.uint8)

    return (
        pedestrian_centroid,
        pedestrian_confidence,
        vehicle_centroid,
        vehicle_confidence,
        summary_image if summary else None,
    )


def predict(
    bev_model,
    lidar,
    ego,
    cam_calib,
    lidar_calib,
    ego_pose_records,
    pred_records,
    summary=False,
):
    lidar_points = tf.cast(lidar, tf.float32)
    (
        pedestrian_centroid,
        pedestrian_confidence,
        vehicle_centroid,
        vehicle_confidence,
        summary_image,
    ) = predict_tf(
        bev_model, lidar_points, ego, ego_pose_records, pred_records, summary
    )

    ego_pose_records.append(ego)

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
        pedestrian_xyz[:, 2] = pedestrian_xyz[:, 2] / 5 + 1.5
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
        vehicle_xyz[:, 2] = vehicle_xyz[:, 2] / 5 + 1.5
        vehicle_xyz[:, [0, 1]] = vehicle_xyz[:, [1, 0]]
        vehicle_xyz[:, 1] = -vehicle_xyz[:, 1]
        vehicle_xyz[:, :3] = quaternion.rotate_vectors(ego_qt, vehicle_xyz[:, :3])
        vehicle_xyz[:, :3] = vehicle_xyz[:, :3] + ego_txyz[np.newaxis]
        vehicle_xyz = list(vehicle_xyz)
    else:
        vehicle_xyz = []

    return pedestrian_xyz, vehicle_xyz, summary_image
