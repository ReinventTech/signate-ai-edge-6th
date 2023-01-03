import os
import quaternion
import numpy as np
import cv2
from time import time


def preprocess(lidar_points):
    lidar_xs = (np.round((lidar_points[:, 0] + 57.6) * 10)).astype(np.int32)
    lidar_ys = (np.round((-lidar_points[:, 1] + 57.6) * 10)).astype(np.int32)
    lidar_zs = (np.round((lidar_points[:, 2] + 3.5) * 10)).astype(np.int32)
    indices = np.logical_and.reduce(
        (
            lidar_xs >= 0,
            lidar_xs < 1152,
            lidar_ys >= 0,
            lidar_ys < 1152,
            lidar_zs >= 0,
            lidar_zs < 40,
        )
    )
    lidar_xs = lidar_xs[indices]
    lidar_ys = lidar_ys[indices]
    lidar_zs = lidar_zs[indices]
    intensities = lidar_points[:, 3][indices]

    lidar_image = np.zeros((1152, 1152, 40), np.float32)
    np.maximum.at(lidar_image, (lidar_ys, lidar_xs, lidar_zs), intensities)
    xs = np.arange(1152, dtype=np.float32)[np.newaxis]
    ys = np.arange(1152, dtype=np.float32)[:, np.newaxis]
    xs = np.tile(xs, (1152, 1))
    ys = np.tile(ys, (1, 1152))
    xys = np.stack([xs, ys], -1)
    vs = xys - 576
    ds = np.linalg.norm(vs, ord=2, axis=-1) / 1000

    input_image = np.concatenate([lidar_image, ds[:, :, np.newaxis]], -1)[np.newaxis]
    return input_image


def postprocess(pred):
    y = pred[:, 64:-64, 64:-64, :]
    y = np.concatenate([y, np.zeros((*y.shape[:3], 1))], -1)
    m = np.stack([y[..., 0] > 0.25, y[..., 1] > 0.64, y[..., 2] > 0], -1)
    fy = y * m.astype(np.float32)
    (
        n_pedestrian,
        pedestrian_ccs,
        _,
        pedestrian_centroid,
    ) = cv2.connectedComponentsWithStats(m[0, :, :, 0].astype(np.uint8), connectivity=4)
    pedestrian_centroid1 = pedestrian_centroid[:, [1, 0]]
    (
        n_vehicle,
        vehicle_ccs,
        _,
        vehicle_centroid,
    ) = cv2.connectedComponentsWithStats(m[0, :, :, 1].astype(np.uint8), connectivity=4)
    vehicle_centroid1 = vehicle_centroid[:, [1, 0]]
    pedestrian_confidence1 = np.zeros([n_pedestrian])
    np.maximum.at(
        pedestrian_confidence1,
        np.reshape(pedestrian_ccs, [-1]),
        np.reshape(fy[..., 0], [-1]),
    )
    pedestrian_centroid1 = pedestrian_centroid1[1:]
    pedestrian_confidence1 = pedestrian_confidence1[1:]

    vehicle_confidence1 = np.zeros([n_vehicle])
    np.maximum.at(
        vehicle_confidence1,
        np.reshape(vehicle_ccs, [-1]),
        np.reshape(fy[..., 1], [-1]),
    )
    vehicle_centroid1 = vehicle_centroid1[1:]
    vehicle_confidence1 = vehicle_confidence1[1:]
    print(pedestrian_centroid1.shape, vehicle_centroid1.shape)

    return (
        pedestrian_centroid1,
        pedestrian_confidence1,
        vehicle_centroid1,
        vehicle_confidence1,
    )

    # m = tf.stack([y[..., 0] <= 0.15, y[..., 1] <= 0.59, y[..., 2] > 0], -1)
    # y = y * tf.cast(m, tf.float32)
    # y = tfa.image.gaussian_filter2d(y)
    # y = tfa.image.gaussian_filter2d(y)
    # y = tfa.image.gaussian_filter2d(y)
    # m = tf.stack([y[..., 0] > 0.02, y[..., 1] > 0.22, y[..., 2] > 0], -1)
    # fy = y * tf.cast(m, tf.float32)
    # pedestrian_ccs = tfa.image.connected_components(m[..., 0])
    # vehicle_ccs = tfa.image.connected_components(m[..., 1])

    # pedestrian_centroid = tf.concat([pedestrian_centroid1, pedestrian_centroid2], 0)
    # vehicle_centroid = tf.concat([vehicle_centroid1, vehicle_centroid2], 0)
    # pedestrian_confidence = tf.concat(
    # [pedestrian_confidence1, pedestrian_confidence2], 0
    # )
    # vehicle_confidence = tf.concat([vehicle_confidence1, vehicle_confidence2], 0)
    # return (
    # pedestrian_centroid,
    # pedestrian_confidence,
    # vehicle_centroid,
    # vehicle_confidence,
    # )


# @tf.function
def predict_tf(runner, lidar_points):
    import xir
    import vart

    xir_graph = xir.Graph.deserialize(os.path.join("../models", "bev.xmodel"))
    xir_root_subgraph = xir_graph.get_root_subgraph()
    xir_child_subgraphs = xir_root_subgraph.toposort_child_subgraph()
    for s in xir_child_subgraphs:
        print(s.get_children())
    print()
    runner = vart.Runner.create_runner(xir_child_subgraphs[1], "run")
    iscale = runner.get_input_tensors()[0].get_attr("fix_point")
    oscale = runner.get_output_tensors()[0].get_attr("fix_point")
    print("input attrs", (runner.get_input_tensors()[0].get_attrs()))
    print("output attrs", (runner.get_output_tensors()[0].get_attrs()))
    t0 = time()
    input_image = preprocess(lidar_points)
    summary = True
    if summary:
        input_summary_image = np.max(input_image[..., :-1], -1)[0][:, :, np.newaxis]
        input_summary_image = np.tile(input_summary_image, (1, 1, 3))
    t1 = time()
    input_image = np.clip(np.round(input_image * (2**iscale)), -128, 127).astype(
        np.int8
    )
    # print(input_image[:, :, :, :40][input_image[:, :, :, :40] > 0])
    preds = np.ones((1, 1152, 1152, 2), np.int8)
    # input_image = np.ascontiguousarray(input_image)
    # preds = np.ascontiguousarray(preds)
    job_id = runner.execute_async(input_image, preds)
    runner.wait(job_id)
    print(preds)
    # print("all zero", np.all(preds == 0))
    t2 = time()
    # preds = bev_model(
    # tf.concat(
    # [
    # input_image,
    # input_image[:, :, ::-1, :],
    # input_image[:, ::-1, :, :],
    # input_image[:, ::-1, ::-1, :],  #
    # tf.transpose(input_image, (0, 2, 1, 3)),
    # ],
    # 0,
    # ),
    # )["detector"]
    # pred = (
    # preds[:1]
    # + preds[1:2, :, ::-1, :]
    # + preds[2:3, ::-1, :, :]
    # + preds[3:4, ::-1, ::-1, :]  #
    # + tf.transpose(preds[4:5], (0, 2, 1, 3))
    # ) / preds.shape[0]
    # print("pred completed")
    # print(preds)
    pred = preds.astype(np.float32) / (2**oscale)
    # pred = np.clip(pred + 3, 0, 6) / 6
    pred = 1 / (1 + np.exp(-pred))
    if summary:
        pred_summary_image = np.concatenate(
            [pred[0], np.zeros((1152, 1152, 1), np.float32)], -1
        )
        summary_image = (
            np.concatenate([input_summary_image, pred_summary_image], 1) * 255
        )
        summary_image = np.clip(summary_image, 0, 255).astype(np.uint8)
        cv2.imwrite("./summary.png", summary_image)
    (
        pedestrian_centroid,
        pedestrian_confidence,
        vehicle_centroid,
        vehicle_confidence,
    ) = postprocess(pred)
    t3 = time()
    # print(t1 - t0, t2 - t1, t3 - t2)

    return (
        pedestrian_centroid,
        pedestrian_confidence,
        vehicle_centroid,
        vehicle_confidence,
    )


def predict(bev_model, lidar, ego, cam_calib, lidar_calib):

    lidar_points = lidar.astype(np.float32)
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
