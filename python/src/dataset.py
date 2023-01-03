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

            camera_txyz = np.array(calib_camera["translation"])
            camera_qt = quaternion.as_quat_array(calib_camera["rotation"])
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
            lidar_proj_xyz = lidar[:, :3]
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
    for scene_idx, scene_data in enumerate(dataset):
        for frame_idx, data in enumerate(scene_data):
            print(scene_idx, data["image_path"])
            break

    exit()

    relative_dir = (
        ("signate3/train" if split else "signate3/calib2")
        if train
        else "signate3/test"
        if split
        else "signate3/calib"
    )
    dataset_dir = os.path.join(root, relative_dir)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    with tf.io.TFRecordWriter(
        f"{dataset_dir}/signate.tfrecords",
        options=tf.io.TFRecordOptions(compression_type="GZIP"),
    ) as writer:
        for scene_idx, scene_data in enumerate(dataset):
            # n_frames = len(scene_data)
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
                lidar_image = np.zeros((1152, 1152, 60))

                lidar_points[:, 0] = np.round((lidar_points[:, 0] + 57.6) * 10)
                lidar_points[:, 1] = np.round((-lidar_points[:, 1] + 57.6) * 10)
                lidar_points[:, 2] = np.round((lidar_points[:, 2] + 4.0) * 10)
                indices = np.logical_and.reduce(
                    (
                        lidar_points[:, 0] >= 0,
                        lidar_points[:, 0] < 1152,
                        lidar_points[:, 1] >= 0,
                        lidar_points[:, 1] < 1152,
                        lidar_points[:, 2] >= 0,
                        lidar_points[:, 2] < 60,
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

                orientation_image = np.zeros((1152, 1152, 1), np.float32)
                xs = np.arange(1152)[np.newaxis, :]
                ys = np.arange(1152)[:, np.newaxis]
                xs = np.tile(xs, (1152, 1))
                ys = np.tile(ys, (1, 1152))
                xys = np.stack([xs, ys], -1)
                ds = np.linalg.norm(
                    xys - np.array([576, 576])[np.newaxis, np.newaxis, :],
                    ord=2,
                    axis=-1,
                ) / (
                    1000
                )  #  ~ 1.0
                # dxs = xys[:, :, 0] - 576
                # dys = xys[:, :, 1] - 576
                # angles = np.arctan2(dys, dxs)
                # cosines = np.reshape(np.cos(angles), (1120, 1120))
                # sines = np.reshape(np.sin(angles), (1120, 1120))
                orientation_image[:, :, 0] = ds
                # orientation_image[:, :, 1] = cosines
                # orientation_image[:, :, 2] = sines

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
                lidar_image = (lidar_image * 65535).astype(np.uint16)
                bbs = np.array(bbs)

                camera_image = cv2.imread(os.path.join(root, data["image_path"]))
                calib_camera = data["calib_camera"]
                camera_txyz = np.array(calib_camera["translation"])
                camera_qt = quaternion.as_quat_array(calib_camera["rotation"])
                [fx, _, cx], [_, fy, cy], [_, _, _] = calib_camera["camera_intrinsic"]

                lidar = np.fromfile(
                    os.path.join(root, data["lidar_path"]), dtype=np.float32
                ).reshape((-1, 5))
                lidar_proj_image = np.zeros(
                    (4, camera_image.shape[0], camera_image.shape[1], 4)
                )
                # depth, intensity
                intensity = lidar[:, 3]
                lidar_proj_xyz = lidar[:, :3]
                lidar_proj_xyz = lidar_proj_xyz - camera_txyz
                lidar_proj_xyz = quaternion.rotate_vectors(
                    camera_qt.inverse(), lidar_proj_xyz
                )
                qt = quaternion.as_quat_array([1, 0, 1, 0])
                for a in range(4):
                    u = np.round(
                        cx + lidar_proj_xyz[:, 0] * fx / lidar_proj_xyz[:, 2]
                    ).astype(int)
                    v = np.round(
                        cy + lidar_proj_xyz[:, 1] * fy / lidar_proj_xyz[:, 2]
                    ).astype(int)
                    indices = np.logical_and.reduce(
                        (
                            u >= 0,
                            u < camera_image.shape[1],
                            v >= 0,
                            v < camera_image.shape[0],
                        )
                    )
                    i = np.round(intensity[indices] * 65535).astype(np.uint16)
                    i_msb = (i // 256).astype(np.uint8)
                    i_lsb = (i % 256).astype(np.uint8)
                    d = np.minimum(
                        np.round(
                            np.linalg.norm(lidar_proj_xyz, ord=2, axis=-1)[indices]
                            * 256
                        ),
                        65535,
                    ).astype(np.uint16)
                    d_msb = (d // 256).astype(np.uint8)
                    d_lsb = (d % 256).astype(np.uint8)
                    u = u[indices]
                    v = v[indices]
                    lidar_proj_image[a][v, u] = np.stack(
                        [d_msb, d_lsb, i_msb, i_lsb], -1
                    )
                    lidar_proj_xyz = quaternion.rotate_vectors(qt, lidar_proj_xyz)
                lidar_proj_image_paths = [None, None, None, None]
                for a in range(4):
                    lidar_proj_image_paths[a] = f"{relative_dir}/{tag}_proj_{a}.png"
                    cv2.imwrite(
                        os.path.join(root, lidar_proj_image_paths[a]),
                        lidar_proj_image[a],
                    )
                # lidar_proj_image_paths_bytes = [
                # bytes(p, "utf-8") for p in lidar_proj_image_paths
                # ]
                # camera_image_path_bytes = bytes(data["image_path"], "utf-8")

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

                            split_orientation_image = orientation_image[
                                dy : dy + 512, dx : dx + 512
                            ].copy()
                            split_orientation_image[:, :, 0] = np.minimum(
                                np.round(split_orientation_image[:, :, 0] * 65535),
                                65535,
                            )
                            split_orientation_image = split_orientation_image.astype(
                                np.uint16
                            )
                            split_orientation_image_msb = (
                                split_orientation_image // 256
                            ).astype(np.uint8)
                            split_orientation_image_lsb = (
                                split_orientation_image % 256
                            ).astype(np.uint8)
                            split_orientation_image_msb_path = (
                                f"{relative_dir}/{split_tag}_orientation_msb.png"
                            )
                            split_orientation_image_lsb_path = (
                                f"{relative_dir}/{split_tag}_orientation_lsb.png"
                            )
                            cv2.imwrite(
                                os.path.join(root, split_orientation_image_msb_path),
                                split_orientation_image_msb,
                            )
                            cv2.imwrite(
                                os.path.join(root, split_orientation_image_lsb_path),
                                split_orientation_image_lsb,
                            )
                            orientation_msb_bytes = bytes(
                                split_orientation_image_msb_path, "utf-8"
                            )
                            orientation_lsb_bytes = bytes(
                                split_orientation_image_lsb_path, "utf-8"
                            )
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
                                        "scene_idx": tf.train.Feature(
                                            int64_list=tf.train.Int64List(
                                                value=[scene_idx]
                                            )
                                        ),
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
                                        "orientation_msb_path": tf.train.Feature(
                                            bytes_list=tf.train.BytesList(
                                                value=[orientation_msb_bytes]
                                            )
                                        ),
                                        "orientation_lsb_path": tf.train.Feature(
                                            bytes_list=tf.train.BytesList(
                                                value=[orientation_lsb_bytes]
                                            )
                                        ),
                                        # "lidar_proj_image_0_path": tf.train.Feature(
                                        # bytes_list=tf.train.BytesList(
                                        # value=[lidar_proj_image_paths_bytes[0]]
                                        # )
                                        # ),
                                        # "lidar_proj_image_90_path": tf.train.Feature(
                                        # bytes_list=tf.train.BytesList(
                                        # value=[lidar_proj_image_paths_bytes[1]]
                                        # )
                                        # ),
                                        # "lidar_proj_image_180_path": tf.train.Feature(
                                        # bytes_list=tf.train.BytesList(
                                        # value=[lidar_proj_image_paths_bytes[2]]
                                        # )
                                        # ),
                                        # "lidar_proj_image_270_path": tf.train.Feature(
                                        # bytes_list=tf.train.BytesList(
                                        # value=[lidar_proj_image_paths_bytes[3]]
                                        # )
                                        # ),
                                        # "camera_image_path": tf.train.Feature(
                                        # bytes_list=tf.train.BytesList(
                                        # value=[camera_image_path_bytes]
                                        # )
                                        # ),
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

                    split_orientation_image = orientation_image.copy()
                    split_orientation_image[:, :, 0] = np.minimum(
                        np.round(split_orientation_image[:, :, 0] * 65535),
                        65535,
                    )
                    split_orientation_image = split_orientation_image.astype(np.uint16)
                    split_orientation_image_msb = (
                        split_orientation_image // 256
                    ).astype(np.uint8)
                    split_orientation_image_lsb = (
                        split_orientation_image % 256
                    ).astype(np.uint8)
                    split_orientation_image_msb_path = (
                        f"{relative_dir}/{split_tag}_orientation_msb.png"
                    )
                    split_orientation_image_lsb_path = (
                        f"{relative_dir}/{split_tag}_orientation_lsb.png"
                    )
                    cv2.imwrite(
                        os.path.join(root, split_orientation_image_msb_path),
                        split_orientation_image_msb,
                    )
                    cv2.imwrite(
                        os.path.join(root, split_orientation_image_lsb_path),
                        split_orientation_image_lsb,
                    )
                    orientation_msb_bytes = bytes(
                        split_orientation_image_msb_path, "utf-8"
                    )
                    orientation_lsb_bytes = bytes(
                        split_orientation_image_lsb_path, "utf-8"
                    )
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
                                "scene_idx": tf.train.Feature(
                                    int64_list=tf.train.Int64List(value=[scene_idx])
                                ),
                                "lidar_gzip_path": tf.train.Feature(
                                    bytes_list=tf.train.BytesList(
                                        value=[lidar_gzip_path]
                                    )
                                ),
                                "bbs": tf.train.Feature(
                                    bytes_list=tf.train.BytesList(value=[bbs_bytes])
                                ),
                                "orientation_msb_path": tf.train.Feature(
                                    bytes_list=tf.train.BytesList(
                                        value=[orientation_msb_bytes]
                                    )
                                ),
                                "orientation_lsb_path": tf.train.Feature(
                                    bytes_list=tf.train.BytesList(
                                        value=[orientation_lsb_bytes]
                                    )
                                ),
                                # "lidar_proj_image_0_path": tf.train.Feature(
                                # bytes_list=tf.train.BytesList(
                                # value=[lidar_proj_image_paths_bytes[0]]
                                # )
                                # ),
                                # "lidar_proj_image_90_path": tf.train.Feature(
                                # bytes_list=tf.train.BytesList(
                                # value=[lidar_proj_image_paths_bytes[1]]
                                # )
                                # ),
                                # "lidar_proj_image_180_path": tf.train.Feature(
                                # bytes_list=tf.train.BytesList(
                                # value=[lidar_proj_image_paths_bytes[2]]
                                # )
                                # ),
                                # "lidar_proj_image_270_path": tf.train.Feature(
                                # bytes_list=tf.train.BytesList(
                                # value=[lidar_proj_image_paths_bytes[3]]
                                # )
                                # ),
                                # "camera_image_path": tf.train.Feature(
                                # bytes_list=tf.train.BytesList(
                                # value=[camera_image_path_bytes]
                                # )
                                # ),
                            }
                        )
                    ).SerializeToString()
                    writer.write(record)
            print(f"scene #{scene_idx} completed")


def get_dataset(root=".", train=True, split=True):
    dataset_path = os.path.join(
        root,
        (
            "signate2/train/signate.tfrecords"
            # "signate_train_with_front6/signate.tfrecords"
            if split
            else "signate2/calib2/signate.tfrecords"
        )
        if train
        else "signate2/test/signate.tfrecords"
        # else "signate_test_with_front6/signate.tfrecords"
        if split else "signate2/calib/signate.tfrecords",
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
