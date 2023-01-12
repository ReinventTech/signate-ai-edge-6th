import os
import quaternion
import numpy as np
import cv2
import xir
import vart
from time import time


def preprocess(lidar_points, z_offset=3.7):
    lidar_xs = (np.round(lidar_points[:, 0] * 10)).astype(np.int32) + 576
    lidar_ys = (np.round(-lidar_points[:, 1] * 10)).astype(np.int32) + 576
    lidar_zs = (np.round((lidar_points[:, 2] + z_offset) * 5)).astype(np.int32)
    indices = np.logical_and.reduce(
        (
            lidar_xs >= 0,
            lidar_xs < 1152,
            lidar_ys >= 0,
            lidar_ys < 1152,
            lidar_zs >= 0,
            lidar_zs < 24,
        )
    )
    lidar_xs = lidar_xs[indices]
    lidar_ys = lidar_ys[indices]
    lidar_zs = lidar_zs[indices]
    intensities = lidar_points[:, 3][indices]

    lidar_image = np.zeros((1152, 1152, 24), np.float32)
    np.maximum.at(lidar_image, (lidar_ys, lidar_xs, lidar_zs), intensities)

    input_image = lidar_image[np.newaxis]
    return input_image


def get_mask(pedestrian_fy, vehicle_fy):
    pedestrian_m = pedestrian_fy > 0.29
    vehicle_m = vehicle_fy > 0.21

    p_m = np.pad(pedestrian_m, ((1, 1), (1, 1)))
    p_m = np.logical_or(
        np.logical_or(np.roll(p_m, 1, 0), np.roll(p_m, -1, 0)),
        np.logical_or(np.roll(p_m, 1, 1), np.roll(p_m, -1, 1)),
    )[1:-1, 1:-1]
    v_m = np.pad(vehicle_m, ((4, 4), (4, 4)))
    v_m = np.logical_or(
        np.logical_or(np.roll(v_m, 4, 0), np.roll(v_m, -4, 0)),
        np.logical_or(np.roll(v_m, 4, 1), np.roll(v_m, -4, 1)),
    )[4:-4, 4:-4]
    pedestrian_m = np.logical_or(
        pedestrian_m,
        np.logical_and(
            np.logical_not(p_m),
            np.logical_and(pedestrian_fy <= 0.29, pedestrian_fy > 0.009),
        ),
    )
    vehicle_m = np.logical_or(
        vehicle_m,
        np.logical_and(
            np.logical_not(v_m),
            np.logical_and(vehicle_fy <= 0.21, vehicle_fy > 0.07),
        ),
    )

    return pedestrian_m.astype(np.uint8), vehicle_m.astype(np.uint8)


def postprocess(pred, ego_pose, ego_pose_records, pred_records):
    y = pred[0, 64:-64, 64:-64, :]

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
        m = np.stack([y, y0, y1], -1)
        y = np.maximum(y, m.sum(-1) - m.max(-1) - m.min(-1))

    pedestrian_fy = y[..., 0]
    vehicle_fy = y[..., 1]
    pedestrian_m, vehicle_m = get_mask(pedestrian_fy, vehicle_fy)
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

    pedestrian_centroid = pedestrian_centroid[:, [1, 0]]
    vehicle_centroid = vehicle_centroid[:, [1, 0]]

    pedestrian_confidence = np.zeros([n_pedestrian])
    np.maximum.at(
        pedestrian_confidence,
        np.reshape(pedestrian_ccs, [-1]),
        np.reshape(pedestrian_fy, [-1]),
    )
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

    vehicle_confidence = np.zeros([n_vehicle])
    np.maximum.at(
        vehicle_confidence,
        np.reshape(vehicle_ccs, [-1]),
        np.reshape(vehicle_fy, [-1]),
    )
    vehicle_centroid = vehicle_centroid[1:]
    vehicle_confidence = vehicle_confidence[1:]

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


def predict_tf(lidar_points, ego_pose, ego_pose_records, pred_records, summary=False):
    # for s in xir_child_subgraphs:
    # print(s.get_children())
    iscale = runner.get_input_tensors()[0].get_attr("fix_point")
    oscale = runner.get_output_tensors()[0].get_attr("fix_point")
    t0 = time()
    input_image = preprocess(lidar_points, 3.7)
    if summary:
        input_summary_image = np.max(input_image[..., :-1], -1)[0][:, :, np.newaxis]
        input_summary_image = np.tile(input_summary_image, (1, 1, 3))
    input_image = np.clip(np.round(input_image * (2**iscale)), -128, 127).astype(
        np.int8
    )
    preds = np.ones((1, 1152, 1152, 2), np.int8)
    t1 = time()
    job_id = runner.execute_async(input_image, preds)
    runner.wait(job_id)
    t2 = time()
    # print(t3 - t2)
    pred = preds.astype(np.float32) / (2**oscale)
    pred = 1 / (1 + np.exp(-pred))
    (
        pedestrian_centroid,
        pedestrian_confidence,
        vehicle_centroid,
        vehicle_confidence,
    ) = postprocess(pred, ego_pose, ego_pose_records, pred_records)

    pred_records.append(pred)
    t3 = time()
    print(t1 - t0, t2 - t1, t3 - t2)

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

    pedestrian_centers = []
    vehicle_centers = []
    for n in range(pedestrian_centroid.shape[0]):
        pedestrian_centers.append(
            np.append(pedestrian_centroid[n], [0, pedestrian_confidence[n]])
        )

    for n in range(vehicle_centroid.shape[0]):
        vehicle_centers.append(
            np.append(vehicle_centroid[n], [0, vehicle_confidence[n]])
        )

    ego_txyz = np.array(ego["translation"], np.float32)
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
