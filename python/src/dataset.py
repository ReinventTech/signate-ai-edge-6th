import json
import quaternion
import numpy as np
import os
import cv2
import tensorflow as tf
import gzip
import argparse


sample_token_to_lidar_sample = {}
sample_token_to_camera_sample = {}
sample_token_to_annotation = {}
token_to_lidar_sample = {}
token_to_camera_sample = {}
token_to_scene = {}
token_to_ego_pose = {}
token_to_calib_sensor = {}


def get_lidar_data(root):
    with open(os.path.join(root, "v1.0-trainval/calibrated_sensor.json")) as f:
        sensor_data = json.load(f)
    with open(os.path.join(root, "v1.0-trainval/sensor.json")) as f:
        sensor = json.load(f)
        lidar_token = [s for s in sensor if s["channel"] == "LIDAR_TOP"][0]["token"]
        lidar_data = [d for d in sensor_data if d["sensor_token"] == lidar_token][0]
        print(lidar_data)


def init_sample_data(root):
    with open(os.path.join(root, "v1.0-trainval/sample_data.json")) as f:
        sample_data = json.load(f)
        for sample in sample_data:
            if sample["fileformat"] == ".bin":
                token_to_lidar_sample[sample["token"]] = sample
                sample_token_to_lidar_sample[sample["sample_token"]] = sample
            else:
                token_to_camera_sample[sample["token"]] = sample
                sample_token_to_camera_sample[sample["sample_token"]] = sample


def get_lidar_samples_by_scene_token(token):
    scene = token_to_scene[token]
    first_sample_token = scene["first_sample_token"]
    samples = [sample_token_to_lidar_sample[first_sample_token]]
    while True:
        last_sample = samples[-1]
        next_token = last_sample["next"]
        if len(next_token) == 0:
            break
        next_sample = token_to_lidar_sample[next_token]
        samples.append(next_sample)
    return samples


def get_camera_samples_by_scene_token(token):
    scene = token_to_scene[token]
    first_sample_token = scene["first_sample_token"]
    samples = [sample_token_to_camera_sample[first_sample_token]]
    while True:
        last_sample = samples[-1]
        next_token = last_sample["next"]
        if len(next_token) == 0:
            break
        next_sample = token_to_camera_sample[next_token]
        samples.append(next_sample)
    return samples


def init_scenes(root):
    with open(os.path.join(root, "v1.0-trainval/scene.json")) as f:
        scenes = json.load(f)
        for scene in scenes:
            token_to_scene[scene["token"]] = scene


def init_ego_pose(root):
    with open(os.path.join(root, "v1.0-trainval/ego_pose.json")) as f:
        ego_poses = json.load(f)
        for ego_pose in ego_poses:
            token_to_ego_pose[ego_pose["token"]] = ego_pose


def init_calibrated_sensor(root):
    with open(os.path.join(root, "v1.0-trainval/calibrated_sensor.json")) as f:
        calib_sensors = json.load(f)
        for calib_sensor in calib_sensors:
            token_to_calib_sensor[calib_sensor["token"]] = calib_sensor


def init_sample_annotation(root):
    with open(os.path.join(root, "v1.0-trainval/sample_annotation.json")) as f:
        sample_annotations = json.load(f)
        for sample_annotation in sample_annotations:
            if sample_annotation["sample_token"] not in sample_token_to_annotation:
                sample_token_to_annotation[sample_annotation["sample_token"]] = []
            sample_token_to_annotation[sample_annotation["sample_token"]].append(
                sample_annotation
            )


def create_dataset(root=".", train=True, split=True):
    init_sample_data(root)
    init_scenes(root)
    init_ego_pose(root)
    init_calibrated_sensor(root)
    init_sample_annotation(root)
    dataset = []
    for token, scene in token_to_scene.items():
        if train:
            if scene["name"] in [
                "scene-0109",
            ]:
                continue
        else:
            if scene["name"] not in [
                "scene-0109",
            ]:
                continue
        lidar_samples = get_lidar_samples_by_scene_token(token)
        camera_samples = get_camera_samples_by_scene_token(token)
        scene_dataset = []
        for lidar_sample, camera_sample in zip(lidar_samples, camera_samples):
            ego_pose = token_to_ego_pose[lidar_sample["ego_pose_token"]]
            calib_lidar = token_to_calib_sensor[lidar_sample["calibrated_sensor_token"]]
            calib_camera = token_to_calib_sensor[
                camera_sample["calibrated_sensor_token"]
            ]

            [fx, _, cx], [_, fy, cy], [_, _, _] = calib_camera["camera_intrinsic"]

            image_path = camera_sample["filename"]
            lidar_path = lidar_sample["filename"]
            if lidar_sample["sample_token"] in sample_token_to_annotation.keys():
                annotation = sample_token_to_annotation[lidar_sample["sample_token"]]
            else:
                annotation = []

            lidar = np.fromfile(
                os.path.join(root, lidar_path), dtype=np.float32
            ).reshape((-1, 5))
            lidar_xyz = lidar[:, :3]
            lidar_txyz = np.array(calib_lidar["translation"])
            lidar_qt = quaternion.as_quat_array(calib_lidar["rotation"])
            lidar_xyz = lidar_xyz + lidar_txyz[np.newaxis]
            lidar_xyz = quaternion.rotate_vectors(lidar_qt, lidar_xyz)
            lidar = np.concatenate(
                [lidar_xyz, lidar[:, -2:]], -1
            )  # [gx, gy, gz, intensity]

            data = {
                "image_path": image_path,
                "lidar_path": lidar_path,
                "ego_pose": ego_pose,
                "calib_lidar": calib_lidar,
                "calib_camera": calib_camera,
                "annotation": annotation,
                "lidar_points": lidar,
            }
            scene_dataset.append(data)
        dataset.append(scene_dataset)

    relative_dir = (
        ("signate/train" if split else "signate/calib2")
        if train
        else "signate/test"
        if split
        else "signate/calib"
    )
    dataset_dir = os.path.join(root, relative_dir)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    with tf.io.TFRecordWriter(
        f"{dataset_dir}/signate.tfrecords",
        options=tf.io.TFRecordOptions(compression_type="GZIP"),
    ) as writer:
        for scene_idx, scene_data in enumerate(dataset):
            for frame_idx, data in enumerate(scene_data):
                tag = f"{scene_idx}_{frame_idx}"
                lidar_points = np.array(
                    [np.array(p) for p in data["lidar_points"]]
                )  # [(x,y,z,intensity)]
                ego_xyz = np.array(data["ego_pose"]["translation"])
                ego_qt = quaternion.as_quat_array(data["ego_pose"]["rotation"])
                annotations = data["annotation"]
                bbs = []
                for annotation in annotations:
                    bb_xyz = np.array(annotation["translation"]) - ego_xyz
                    bb_xyz = quaternion.rotate_vectors(ego_qt.inverse(), bb_xyz)
                    bb_sxyz = np.array(annotation["size"])
                    bb_rxyz = np.array(annotation["rotation"])
                    bb_qt = quaternion.as_quat_array(annotation["rotation"])
                    bb_qt = bb_qt * ego_qt.inverse()
                    bb_rxyz = quaternion.as_float_array(bb_qt)
                    category = annotation["category_name"]
                    if category not in [
                        "human.pedestrian.adult",
                        "human.pedestrian.construction_worker",
                        "vehicle.car",
                    ]:
                        continue
                    category = 1 if category == "vehicle.car" else 0
                    d_xyz = np.linalg.norm(bb_xyz, ord=2)
                    if category == 0 and d_xyz >= 45:
                        continue
                    if category == 1 and d_xyz >= 60:
                        continue
                    bb = np.concatenate(
                        (bb_xyz, bb_sxyz, bb_rxyz, np.array([category]))
                    )
                    bbs.append(bb)
                lidar_image = np.zeros((1152, 1152, 30))

                lidar_points[:, 0] = np.round((lidar_points[:, 0] + 57.6) * 10)
                lidar_points[:, 1] = np.round((-lidar_points[:, 1] + 57.6) * 10)
                lidar_points[:, 2] = np.round((lidar_points[:, 2] + 4.0) * 5)
                indices = np.logical_and.reduce(
                    (
                        lidar_points[:, 0] >= 0,
                        lidar_points[:, 0] < 1152,
                        lidar_points[:, 1] >= 0,
                        lidar_points[:, 1] < 1152,
                        lidar_points[:, 2] >= 0,
                        lidar_points[:, 2] < 30,
                    )
                )
                lidar_points = lidar_points[indices]
                lidar_xs, lidar_ys, lidar_zs, intensities = (
                    lidar_points[:, 0].astype(int),
                    lidar_points[:, 1].astype(int),
                    lidar_points[:, 2].astype(int),
                    lidar_points[:, 3],
                )
                np.maximum.at(lidar_image, (lidar_ys, lidar_xs, lidar_zs), intensities)

                xs = np.arange(1152)[np.newaxis, :]
                ys = np.arange(1152)[:, np.newaxis]
                xs = np.tile(xs, (1152, 1))
                ys = np.tile(ys, (1, 1152))

                lidar_demo = np.tile(
                    np.clip(lidar_image.max(-1, keepdims=True) * 255, 0, 255).astype(
                        np.uint8
                    ),
                    (1, 1, 3),
                )
                for bb in bbs:
                    x, y = bb[:2]
                    x, y = round(x * 10 + 576), round(-y * 10 + 576)
                    lidar_demo = cv2.circle(
                        lidar_demo,
                        center=(x, y),
                        radius=3,
                        color=(0, 255, 0) if bb[-1] == 0 else (0, 0, 255),
                        thickness=1,
                    )
                lidar_image = (lidar_image * 255).astype(np.uint8)
                bbs = np.array(bbs)

                calib_camera = data["calib_camera"]
                [fx, _, cx], [_, fy, cy], [_, _, _] = calib_camera["camera_intrinsic"]

                lidar = np.fromfile(
                    os.path.join(root, data["lidar_path"]), dtype=np.float32
                ).reshape((-1, 5))

                if split:
                    for dy in [0, 320, 640]:
                        for dx in [0, 320, 640]:
                            split_tag = f"{tag}_{dy}_{dx}"
                            split_lidar_image = lidar_image[
                                dy : dy + 512, dx : dx + 512
                            ]
                            if bbs.shape[0] > 0:
                                split_bbs = bbs[
                                    np.logical_and.reduce(
                                        (
                                            (bbs[:, 0] + 57.6) * 10 >= dx,
                                            (bbs[:, 0] + 57.6) * 10 < dx + 512,
                                            (-bbs[:, 1] + 57.6) * 10 >= dy,
                                            (-bbs[:, 1] + 57.6) * 10 < dy + 512,
                                        )
                                    )
                                ]
                                split_bbs[:, 0] = split_bbs[:, 0] - dx / 10
                                split_bbs[:, 1] = split_bbs[:, 1] + dy / 10
                            else:
                                split_bbs = bbs
                            if split_bbs.shape[0] == 0 or np.all(
                                split_lidar_image == 0
                            ):
                                continue

                            lidar_bytes = split_lidar_image.tobytes()
                            lidar_gzip = gzip.compress(lidar_bytes)
                            lidar_gzip_path = (
                                f"{relative_dir}/{split_tag}_lidar_image.gzip"
                            )
                            with open(os.path.join(root, lidar_gzip_path), "wb") as f:
                                f.write(lidar_gzip)
                            bbs_bytes = split_bbs.tobytes()
                            lidar_gzip_path = bytes(lidar_gzip_path, "utf-8")
                            record = tf.train.Example(
                                features=tf.train.Features(
                                    feature={
                                        "lidar_gzip_path": tf.train.Feature(
                                            bytes_list=tf.train.BytesList(
                                                value=[lidar_gzip_path]
                                            )
                                        ),
                                        "bbs": tf.train.Feature(
                                            bytes_list=tf.train.BytesList(
                                                value=[bbs_bytes]
                                            )
                                        ),
                                    }
                                )
                            ).SerializeToString()
                            writer.write(record)
                else:
                    split_tag = tag
                    split_lidar_image = lidar_image
                    split_bbs = bbs
                    if split_bbs.shape[0] == 0 or np.all(split_lidar_image == 0):
                        continue

                    lidar_bytes = split_lidar_image.tobytes()
                    lidar_gzip = gzip.compress(lidar_bytes)
                    lidar_gzip_path = f"{relative_dir}/{split_tag}_lidar_image.gzip"
                    with open(os.path.join(root, lidar_gzip_path), "wb") as f:
                        f.write(lidar_gzip)
                    bbs_bytes = split_bbs.tobytes()
                    lidar_gzip_path = bytes(lidar_gzip_path, "utf-8")
                    record = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                "lidar_gzip_path": tf.train.Feature(
                                    bytes_list=tf.train.BytesList(
                                        value=[lidar_gzip_path]
                                    )
                                ),
                                "bbs": tf.train.Feature(
                                    bytes_list=tf.train.BytesList(value=[bbs_bytes])
                                ),
                            }
                        )
                    ).SerializeToString()
                    writer.write(record)
            print(f"scene #{scene_idx} completed")


def get_dataset(root=".", train=True, split=True):
    dataset_path = os.path.join(
        root,
        (
            "signate/train/signate.tfrecords"
            if split
            else "signate/calib2/signate.tfrecords"
        )
        if train
        else "signate/test/signate.tfrecords"
        if split
        else "signate/calib/signate.tfrecords",
    )
    return tf.data.TFRecordDataset(dataset_path, compression_type="GZIP")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A command line tool for dataset")
    subparsers = parser.add_subparsers(help="sub-command help")
    parser_create = subparsers.add_parser(
        "create", help="create a dataset (.tfrecords)"
    )
    parser_create.add_argument("--root", type=str)
    parser_create.set_defaults(
        func=lambda args: [
            create_dataset(args.root, True),
            create_dataset(args.root, False),
            create_dataset(args.root, False, False),
            create_dataset(args.root, True, False),
        ]
    )
    args = parser.parse_args()
    args.func(args)
