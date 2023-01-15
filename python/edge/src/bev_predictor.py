import os
import quaternion
import numpy as np
import cv2
import xir
import vart
from time import time
from numba import njit, prange


@njit(parallel=True)
def max_at(y, a, b, n):
    for i in prange(n):
        y[a[i]] = max(y[a[i]], b[i])
    return y


@njit(parallel=True)
def max_at3(y, a, b, c, d, n):
    for i in prange(n):
        y[a[i], b[i], c[i]] = max(y[a[i], b[i], c[i]], d[i])
    return y


@njit(parallel=True)
def merge3(y0, y1, y2):
    y = np.maximum(y0, np.minimum(y1, np.maximum(y0, y2)))
    return y


@njit(parallel=True)
def preprocess(lidar_points, iscale, z_offset=3.7):
    # t0 = time()
    lidar_xs = ((lidar_points[:, 0] * 10 + 0.5)).astype(np.int32) + 576
    lidar_ys = ((-lidar_points[:, 1] * 10 + 0.5)).astype(np.int32) + 576
    lidar_zs = (((lidar_points[:, 2] + z_offset) * 5 + 0.5)).astype(np.int32)
    indices = np.logical_and(
        np.logical_and(
            np.logical_and(lidar_xs >= 0, lidar_xs < 1152),
            np.logical_and(lidar_ys >= 0, lidar_ys < 1152),
        ),
        np.logical_and(lidar_zs >= 0, lidar_zs < 24),
    )
    lidar_xs = lidar_xs[indices]
    lidar_ys = lidar_ys[indices]
    lidar_zs = lidar_zs[indices]
    intensities = np.clip(
        (lidar_points[:, 3][indices] * (2**iscale) + 0.5), -128, 127
    ).astype(np.uint8)

    # t1 = time()
    lidar_image = max_at3(
        np.zeros((1152, 1152, 24), np.int8),
        lidar_ys,
        lidar_xs,
        lidar_zs,
        intensities,
        lidar_xs.shape[0],
    )

    input_image = np.expand_dims(lidar_image, 0)
    # t2 = time()
    # print("pre", t1 - t0, t2 - t1)
    return input_image


@njit(parallel=True)
def get_mask(pedestrian_fy, vehicle_fy):
    pedestrian_m = pedestrian_fy > 0.28
    vehicle_m = vehicle_fy > 0.20

    # p_m = np.pad(pedestrian_m, ((1, 1), (1, 1)))
    # p_m = np.logical_or(
    # np.logical_or(np.roll(p_m, 1, 0), np.roll(p_m, -1, 0)),
    # np.logical_or(np.roll(p_m, 1, 1), np.roll(p_m, -1, 1)),
    # )[1:-1, 1:-1]
    p_m = pedestrian_m.copy()
    for y in prange(1023):
        for x in prange(1024):
            p_m[y, x] = p_m[y, x] or pedestrian_m[y + 1, x]
            p_m[y + 1, x] = p_m[y + 1, x] or pedestrian_m[y, x]
    for y in prange(1024):
        for x in prange(1023):
            p_m[y, x] = p_m[y, x] or pedestrian_m[y, x + 1]
            p_m[y, x + 1] = p_m[y, x + 1] or pedestrian_m[y, x]
    # v_m = np.pad(vehicle_m, ((4, 4), (4, 4)))
    # v_m = np.logical_or(
    # np.logical_or(np.roll(v_m, 4, 0), np.roll(v_m, -4, 0)),
    # np.logical_or(np.roll(v_m, 4, 1), np.roll(v_m, -4, 1)),
    # )[4:-4, 4:-4]
    v_m = vehicle_m.copy()
    for y in prange(1023):
        for x in prange(1024):
            v_m[y, x] = v_m[y, x] or vehicle_m[y + 1, x]
            v_m[y + 1, x] = v_m[y + 1, x] or vehicle_m[y, x]
    for y in prange(1024):
        for x in prange(1023):
            v_m[y, x] = v_m[y, x] or vehicle_m[y, x + 1]
            v_m[y, x + 1] = v_m[y, x + 1] or vehicle_m[y, x]
    pedestrian_m = np.logical_or(
        pedestrian_m,
        np.logical_and(
            np.logical_not(p_m),
            pedestrian_fy > 0.009,
        ),
    )
    vehicle_m = np.logical_or(
        vehicle_m,
        np.logical_and(np.logical_not(v_m), vehicle_fy > 0.05),
    )

    return pedestrian_m.astype(np.uint8), vehicle_m.astype(np.uint8)


def postprocess(pred, ego_pose, ego_pose_records, pred_records):
    t0 = time()
    y = pred[0, 64:-64, 64:-64, :]

    if len(ego_pose_records) >= 2:
        s0 = time()
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
        s1 = time()

        r0 = cv2.getRotationMatrix2D((512, 512), -a0, 1)
        r0[0, 2] = r0[0, 2] - txyz0[0] * 10
        r0[1, 2] = r0[1, 2] + txyz0[1] * 10
        y0 = cv2.warpAffine(
            pred_records[-1][0, 64:-64, 64:-64, :],
            r0,
            (1024, 1024),
            flags=cv2.INTER_NEAREST,
        )

        r1 = cv2.getRotationMatrix2D((512, 512), -a1, 1)
        r1[0, 2] = r1[0, 2] - txyz1[0] * 10
        r1[1, 2] = r1[1, 2] + txyz1[1] * 10
        y1 = cv2.warpAffine(
            pred_records[-2][0, 64:-64, 64:-64, :],
            r1,
            (1024, 1024),
            flags=cv2.INTER_NEAREST,
        )
        s2 = time()
        y = merge3(y, y0, y1)
        s3 = time()
        print("s", s1 - s0, s2 - s1, s3 - s2)

    pedestrian_fy = y[..., 0]
    vehicle_fy = y[..., 1]
    t1 = time()
    pedestrian_m, vehicle_m = get_mask(pedestrian_fy, vehicle_fy)
    t2 = time()
    (
        n_pedestrian,
        pedestrian_ccs,
        _pedestrian_stats,
        pedestrian_centroid,
    ) = cv2.connectedComponentsWithStatsWithAlgorithm(
        pedestrian_m,
        connectivity=4,
        ltype=cv2.CV_32S,
        ccltype=cv2.CCL_GRANA,
    )
    pedestrian_centroid = pedestrian_centroid.astype(np.float32)
    (
        n_vehicle,
        vehicle_ccs,
        _vehicle_stats,
        vehicle_centroid,
    ) = cv2.connectedComponentsWithStatsWithAlgorithm(
        vehicle_m,
        connectivity=4,
        ltype=cv2.CV_32S,
        ccltype=cv2.CCL_GRANA,
    )
    vehicle_centroid = vehicle_centroid.astype(np.float32)
    t3 = time()

    pedestrian_centroid = pedestrian_centroid[:, [1, 0]]
    vehicle_centroid = vehicle_centroid[:, [1, 0]]

    t4 = time()
    pedestrian_confidence = max_at(
        np.zeros([n_pedestrian], np.float32),
        np.reshape(pedestrian_ccs, [-1]),
        np.reshape(pedestrian_fy, [-1]),
        pedestrian_ccs.size,
    )
    t5 = time()
    additional_indices = np.logical_and(
        _pedestrian_stats[1:, -1] > 78, pedestrian_confidence[1:] > 0.28
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
    t6 = time()

    vehicle_confidence = max_at(
        np.zeros([n_vehicle], np.float32),
        np.reshape(vehicle_ccs, [-1]),
        np.reshape(vehicle_fy, [-1]),
        vehicle_ccs.size,
    )
    t7 = time()
    vehicle_centroid = vehicle_centroid[1:]
    vehicle_confidence = vehicle_confidence[1:]
    t8 = time()
    print(
        "pp",
        t1 - t0,
        t2 - t1,
        t3 - t2,
        t4 - t3,
        t5 - t4,
        t6 - t5,
        t7 - t6,
        t8 - t7,
    )

    return (
        pedestrian_centroid,
        pedestrian_confidence,
        vehicle_centroid,
        vehicle_confidence,
    )


xir_graph = xir.Graph.deserialize(os.path.join("../models", "bev.xmodel"))
xir_root_subgraph = xir_graph.get_root_subgraph()
xir_child_subgraphs = xir_root_subgraph.toposort_child_subgraph()
runner = vart.Runner.create_runner(xir_child_subgraphs[1], "run")
iscale = runner.get_input_tensors()[0].get_attr("fix_point")
oscale = runner.get_output_tensors()[0].get_attr("fix_point")
sigmoid_table = []
for x in range(-128, 128):
    sigmoid_table.append(1 / (1 + np.exp(-np.float32(x) / (2**oscale))))


@njit(parallel=True)
def sigmoid_(x, sigmoid_table):
    return sigmoid_table[int(x) + 128]


@njit(parallel=True)
def sigmoid(inp, sigmoid_table):
    outp = np.empty_like(inp, dtype=np.float32)
    for y in prange(inp.shape[1]):
        for x in prange(inp.shape[2]):
            outp[0, y, x, 0] = sigmoid_(inp[0, y, x, 0], sigmoid_table)
            outp[0, y, x, 1] = sigmoid_(inp[0, y, x, 1], sigmoid_table)
    return outp


@njit(parallel=True)
def suppress_predictions_without_lidar_points(
    input_image, pedestrian_centroid, pedestrian_confidence
):
    input_intensities = input_image
    for n in prange(pedestrian_centroid.shape[0]):
        y, x = pedestrian_centroid[n]
        y, x = round(y) + 64, round(x) + 64
        if not input_intensities[:, y - 7 : y + 7, x - 7 : x + 7, :].any():
            pedestrian_confidence[n] = pedestrian_confidence[n] * 0.1
    return pedestrian_confidence


def predict_tf(lidar_points, ego_pose, ego_pose_records, pred_records, summary=False):
    t0 = time()
    input_image = preprocess(lidar_points, iscale, 3.7)
    t1 = time()
    if summary:
        input_summary_image = np.max(input_image[..., :-1], -1)[0][:, :, np.newaxis]
        input_summary_image = np.tile(input_summary_image, (1, 1, 3))
    preds = np.ones((1, 1152, 1152, 2), np.int8)
    t2 = time()
    job_id = runner.execute_async(input_image, preds)
    runner.wait(job_id)
    t3 = time()
    # pred = preds.astype(np.float32) / (2**oscale)
    # pred = 1 / (1 + np.exp(-pred))
    pred = sigmoid(preds, sigmoid_table)
    t4 = time()
    (
        pedestrian_centroid,
        pedestrian_confidence,
        vehicle_centroid,
        vehicle_confidence,
    ) = postprocess(pred, ego_pose, ego_pose_records, pred_records)

    t5 = time()
    pedestrian_confidence = suppress_predictions_without_lidar_points(
        input_image, pedestrian_centroid, pedestrian_confidence
    )

    pred_records.append(pred)
    t6 = time()
    print(t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5)

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
    lidar,
    ego,
    cam_calib,
    lidar_calib,
    ego_pose_records,
    pred_records,
    summary=False,
):

    lidar_points = lidar.astype(np.float32)
    (
        pedestrian_centroid,
        pedestrian_confidence,
        vehicle_centroid,
        vehicle_confidence,
        summary_image,
    ) = predict_tf(lidar_points, ego, ego_pose_records, pred_records, summary)

    ego_pose_records.append(ego)

    if pedestrian_centroid.shape[0] > 0:
        pedestrian_centers = np.concatenate(
            [
                pedestrian_centroid,
                np.full((pedestrian_centroid.shape[0], 1), 1.5, np.float32),
                pedestrian_confidence[:, np.newaxis],
            ],
            -1,
        )
    else:
        pedestrian_centers = []

    if vehicle_centroid.shape[0] > 0:
        vehicle_centers = np.concatenate(
            [
                vehicle_centroid,
                np.full((vehicle_centroid.shape[0], 1), 1.5, np.float32),
                vehicle_confidence[:, np.newaxis],
            ],
            -1,
        )
    else:
        vehicle_centers = []

    ego_txyz = np.array(ego["translation"], np.float32)
    ego_qt = quaternion.as_quat_array(ego["rotation"])

    if len(pedestrian_centers) > 0:
        pedestrian_xyz = pedestrian_centers
        pedestrian_xyz[:, :2] = pedestrian_xyz[:, :2] / 10 - 51.2
        # pedestrian_xyz[:, 2] = pedestrian_xyz[:, 2] / 5 + 1.5
        pedestrian_xyz[:, [0, 1]] = pedestrian_xyz[:, [1, 0]]
        pedestrian_xyz[:, 1] = -pedestrian_xyz[:, 1]
        pedestrian_xyz[:, :3] = quaternion.rotate_vectors(ego_qt, pedestrian_xyz[:, :3])
        pedestrian_xyz[:, :3] = pedestrian_xyz[:, :3] + ego_txyz[np.newaxis]
        # pedestrian_xyz = list(pedestrian_xyz)
    else:
        pedestrian_xyz = []

    if len(vehicle_centers) > 0:
        vehicle_xyz = vehicle_centers
        vehicle_xyz[:, :2] = vehicle_xyz[:, :2] / 10 - 51.2
        # vehicle_xyz[:, 2] = vehicle_xyz[:, 2] / 5 + 1.5
        vehicle_xyz[:, [0, 1]] = vehicle_xyz[:, [1, 0]]
        vehicle_xyz[:, 1] = -vehicle_xyz[:, 1]
        vehicle_xyz[:, :3] = quaternion.rotate_vectors(ego_qt, vehicle_xyz[:, :3])
        vehicle_xyz[:, :3] = vehicle_xyz[:, :3] + ego_txyz[np.newaxis]
        # vehicle_xyz = list(vehicle_xyz)
    else:
        vehicle_xyz = []

    return pedestrian_xyz, vehicle_xyz, summary_image
