import os
import dataset

os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
import tensorflow as tf
import numpy as np
import argparse
import tensorflow_addons as tfa
from bev_model import Detector

TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 8
TRAIN_H, TRAIN_W = 256, 256
TEST_H, TEST_W = 512, 512

parser = argparse.ArgumentParser(description="Train BEV model")
parser.add_argument("--path", type=str, help="The training/test dataset path")
parser.add_argument("--ckpt", type=str, help="The checkpoint path")
parser.add_argument("--epoch", type=int, default=100, help="The max number of epochs")
args = parser.parse_args()


@tf.function
def train_path_to_lidar_bbs(frame, dataset_dir):
    fmt = {
        "lidar_gzip_path": tf.io.FixedLenFeature([], tf.string),
        "bbs": tf.io.FixedLenFeature([], tf.string),
        "orientation_msb_path": tf.io.FixedLenFeature([], tf.string),
        "orientation_lsb_path": tf.io.FixedLenFeature([], tf.string),
        "lidar_proj_image_0_path": tf.io.FixedLenFeature([], tf.string),
        "lidar_proj_image_90_path": tf.io.FixedLenFeature([], tf.string),
        "lidar_proj_image_180_path": tf.io.FixedLenFeature([], tf.string),
        "lidar_proj_image_270_path": tf.io.FixedLenFeature([], tf.string),
        "camera_image_path": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(frame, fmt)
    lidar = tf.io.read_file(dataset_dir + "/" + example["lidar_gzip_path"])
    lidar = tf.io.decode_compressed(lidar, compression_type="GZIP")
    lidar = tf.io.decode_raw(lidar, tf.uint16)
    lidar = tf.reshape(lidar, (512, 512, 40))
    bbs = tf.io.decode_raw(example["bbs"], tf.float64)
    bbs = tf.reshape(bbs, (-1, 11))
    orientation_msb_path = dataset_dir + "/" + example["orientation_msb_path"]
    orientation_lsb_path = dataset_dir + "/" + example["orientation_lsb_path"]
    orientation_msb = tf.image.decode_image(
        tf.io.read_file(orientation_msb_path), channels=3, name="decode_orientation_msb"
    )
    orientation_lsb = tf.image.decode_image(
        tf.io.read_file(orientation_lsb_path), channels=3, name="decode_orientation_lsb"
    )
    orientation_image = tf.cast(orientation_msb, tf.float32) * 256 + tf.cast(
        orientation_lsb, tf.float32
    )
    orientation_image = tf.stack(
        [
            orientation_image[:, :, 0] / 65535,
            # orientation_image[:, :, 1] / 32768 - 1,  # cos
            # orientation_image[:, :, 2] / 32768 - 1,  # sin
        ],
        -1,
    )  # (512, 512, 3)
    lidar_proj_image_paths = tf.stack(
        [
            dataset_dir + "/" + example["lidar_proj_image_0_path"],
            dataset_dir + "/" + example["lidar_proj_image_90_path"],
            dataset_dir + "/" + example["lidar_proj_image_180_path"],
            dataset_dir + "/" + example["lidar_proj_image_270_path"],
        ]
    )
    # lidar_proj_image_paths = tf.stack([lidar_proj_image_paths] * 4, 0)
    lidar_proj_images = [tf.io.read_file(lidar_proj_image_paths[i]) for i in range(4)]
    lidar_proj_images = [
        tf.image.decode_image(lidar_proj_images[i], channels=4, name=f"decode_proj_{i}")
        for i in range(4)
    ]
    lidar_proj_images = [tf.cast(lidar_proj_images[i], tf.float32) for i in range(4)]
    lidar_proj_images = [
        tf.stack(
            [
                lidar_proj_images[i][:, :, 0] * 256 + lidar_proj_images[i][:, :, 1],
                lidar_proj_images[i][:, :, 2] * 256 + lidar_proj_images[i][:, :, 3],
            ],
            -1,
        )
        for i in range(4)
    ]
    lidar_proj_images = tf.stack([lidar_proj_images[i] / 65535 for i in range(4)], 0)
    camera_image_path = dataset_dir + "/" + example["camera_image_path"]
    camera_image = tf.io.read_file(camera_image_path)
    camera_image = tf.image.decode_image(camera_image, channels=3, name="decode_camera")

    center_image = tf.zeros((512, 512, 2), tf.bool)
    if len(bbs.shape) > 1:
        bbx = (bbs[:, 0] + 57.6) * 10
        bby = (-bbs[:, 1] + 57.6) * 10
        category = bbs[:, -1]
        d = 2
        for dy in range(-d, d + 1):
            for dx in range(-d, d + 1):
                if dx**2 + dy**2 > d**2:
                    continue
                bbs = tf.cast(tf.stack([bby + dy, bbx + dx, category], -1), tf.int32)
                bbs = tf.clip_by_value(bbs, 0, 511)
                updates = tf.ones((tf.shape(bbs)[0]), tf.bool)
                center_image = tf.tensor_scatter_nd_update(center_image, bbs, updates)
    lidar = tf.cast(lidar, tf.float32) / 65535
    center_image = tf.cast(center_image, tf.float32)
    lidar_bbs = tf.concat([lidar, orientation_image[..., -1:], center_image], -1)
    return lidar_bbs


@tf.function
def test_path_to_lidar_bbs(frame, dataset_dir):
    fmt = {
        "lidar_gzip_path": tf.io.FixedLenFeature([], tf.string),
        "bbs": tf.io.FixedLenFeature([], tf.string),
        "orientation_msb_path": tf.io.FixedLenFeature([], tf.string),
        "orientation_lsb_path": tf.io.FixedLenFeature([], tf.string),
        "lidar_proj_image_0_path": tf.io.FixedLenFeature([], tf.string),
        "lidar_proj_image_90_path": tf.io.FixedLenFeature([], tf.string),
        "lidar_proj_image_180_path": tf.io.FixedLenFeature([], tf.string),
        "lidar_proj_image_270_path": tf.io.FixedLenFeature([], tf.string),
        "camera_image_path": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(frame, fmt)
    lidar = tf.io.read_file(dataset_dir + "/" + example["lidar_gzip_path"])
    lidar = tf.io.decode_compressed(lidar, compression_type="GZIP")
    lidar = tf.io.decode_raw(lidar, tf.uint16)
    lidar = tf.reshape(lidar, (512, 512, 40))
    bbs = tf.io.decode_raw(example["bbs"], tf.float64)
    bbs = tf.reshape(bbs, (-1, 11))
    orientation_msb_path = dataset_dir + "/" + example["orientation_msb_path"]
    orientation_lsb_path = dataset_dir + "/" + example["orientation_lsb_path"]
    orientation_msb = tf.image.decode_image(
        tf.io.read_file(orientation_msb_path), channels=3, name="decode_orientation_msb"
    )
    orientation_lsb = tf.image.decode_image(
        tf.io.read_file(orientation_lsb_path), channels=3, name="decode_orientation_lsb"
    )
    orientation_image = tf.cast(orientation_msb, tf.float32) * 256 + tf.cast(
        orientation_lsb, tf.float32
    )
    orientation_image = tf.stack(
        [
            orientation_image[:, :, 0] / 65535,  # distance(10m) / 65535
            # orientation_image[:, :, 1] / 32768 - 1,  # cos
            # orientation_image[:, :, 2] / 32768 - 1,  # sin
        ],
        -1,
    )  # (512, 512, 3)
    lidar_proj_image_paths = tf.stack(
        [
            dataset_dir + "/" + example["lidar_proj_image_0_path"],
            dataset_dir + "/" + example["lidar_proj_image_90_path"],
            dataset_dir + "/" + example["lidar_proj_image_180_path"],
            dataset_dir + "/" + example["lidar_proj_image_270_path"],
        ]
    )
    # lidar_proj_image_paths = tf.stack([lidar_proj_image_paths] * 4, 0)
    lidar_proj_images = [tf.io.read_file(lidar_proj_image_paths[i]) for i in range(4)]
    lidar_proj_images = [
        tf.image.decode_image(lidar_proj_images[i], channels=4, name=f"decode_proj_{i}")
        for i in range(4)
    ]
    lidar_proj_images = [tf.cast(lidar_proj_images[i], tf.float32) for i in range(4)]
    lidar_proj_images = [
        tf.stack(
            [
                lidar_proj_images[i][:, :, 0] * 256 + lidar_proj_images[i][:, :, 1],
                lidar_proj_images[i][:, :, 2] * 256 + lidar_proj_images[i][:, :, 3],
            ],
            -1,
        )
        for i in range(4)
    ]
    lidar_proj_images = [lidar_proj_images[i] / 65535 for i in range(4)]
    camera_image_path = dataset_dir + "/" + example["camera_image_path"]
    camera_image = tf.io.read_file(camera_image_path)
    camera_image = tf.image.decode_image(camera_image, channels=3, name="decode_camera")

    center_image = tf.zeros((512, 512, 2), tf.bool)
    if len(bbs.shape) > 1:
        bbx = (bbs[:, 0] + 57.6) * 10
        bby = (-bbs[:, 1] + 57.6) * 10
        category = bbs[:, -1]
        d = 2
        for dy in range(-d, d + 1):
            for dx in range(-d, d + 1):
                if dx**2 + dy**2 > d**2 + 1:
                    continue
                bbs = tf.cast(tf.stack([bby + dy, bbx + dx, category], -1), tf.int32)
                bbs = tf.clip_by_value(bbs, 0, 511)
                updates = tf.ones((tf.shape(bbs)[0]), tf.bool)
                center_image = tf.tensor_scatter_nd_update(center_image, bbs, updates)
    lidar = tf.cast(lidar, tf.float32) / 65535
    center_image = tf.cast(center_image, tf.float32)
    lidar_bbs = tf.concat([lidar, orientation_image[..., -1:], center_image], -1)
    return lidar_bbs


@tf.function
def preprocess(frames, dataset_dir, training=True):
    with tf.device("/gpu:0"):
        if training:
            lidar_bbs = tf.map_fn(
                lambda x: train_path_to_lidar_bbs(x, dataset_dir),
                frames,
                dtype=tf.float32,
                parallel_iterations=12,
            )
            # lidar_bbs = tf.vectorized_map(
            # train_path_to_lidar_bbs,
            # frames,
            # )
        else:
            lidar_bbs = tf.map_fn(
                lambda x: test_path_to_lidar_bbs(x, dataset_dir),
                frames,
                dtype=tf.float32,
                parallel_iterations=1,
            )
        if training:
            lidar_bbs = tf.pad(
                lidar_bbs,
                [
                    [0, TRAIN_BATCH_SIZE - tf.shape(lidar_bbs)[0]],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
            )
            lidar_bbs = [
                tf.image.random_crop(
                    lidar_bbs, (TRAIN_BATCH_SIZE, TRAIN_H, TRAIN_W, 43)
                )
                for _ in range(2)
            ]
            lidar_bbs = tf.concat(lidar_bbs, 0)
            lidar_bbs = tf.image.random_flip_left_right(lidar_bbs)
            angles = tf.random.uniform([TRAIN_BATCH_SIZE * 2], 0, 2 * np.pi)
            # rotate orientation
            lidar, orientation, bbs = (
                lidar_bbs[..., :40],
                lidar_bbs[..., 40:41],
                lidar_bbs[..., 41:],
            )
            delta = tf.random.uniform(lidar.shape, 0.7, 1.4)
            lidar = tf.math.exp(tf.math.log(lidar + 1e-10) * delta)
            lidar_shape = lidar.shape
            lidar = tf.reshape(
                lidar,
                (
                    lidar.shape[0],
                    lidar.shape[1] // 2,
                    2,
                    lidar.shape[2] // 2,
                    2,
                    lidar.shape[3] // 2,
                    2,
                ),
            )
            lidar = tf.transpose(lidar, (2, 4, 6, 0, 1, 3, 5))
            lidar = tf.reshape(lidar, (8, *lidar.shape[3:]))
            lidar = tf.random.shuffle(lidar)
            lidar = tf.reshape(lidar, (2, 2, 2, *lidar.shape[1:]))
            lidar = tf.transpose(lidar, (3, 4, 0, 5, 1, 6, 2))
            lidar = tf.reshape(lidar, lidar_shape)
            shift = tf.random.uniform(
                shape=[lidar.shape[0]], minval=-1, maxval=1, dtype=tf.int32
            )
            lidar = tf.pad(lidar, [[0, 0], [0, 0], [0, 0], [1, 1]], mode="SYMMETRIC")
            lidar = tf.roll(lidar, shift=shift, axis=tf.fill([lidar.shape[0]], 3))
            lidar = lidar[..., 1:-1]
            # t_angles = tf.reshape(angles, [-1, 1, 1])
            # orientation = tf.stack(
            # [
            # orientation[..., 0] * tf.math.cos(t_angles)
            # - orientation[..., 1] * tf.math.sin(t_angles),
            # orientation[..., 0] * tf.math.sin(t_angles)
            # - orientation[..., 1] * tf.math.cos(t_angles),
            # orientation[..., 2],
            # ],
            # -1,
            # )
            lidar = tfa.image.rotate(lidar, angles, fill_mode="constant")
            orientation = tfa.image.rotate(orientation, angles, fill_mode="nearest")
            bbs = tfa.image.rotate(bbs, angles, fill_mode="constant")
            lidar_bbs = tf.concat([lidar, orientation, bbs], -1)
            lidar_bbs = tf.image.random_crop(
                lidar_bbs, (TRAIN_BATCH_SIZE, TRAIN_H, TRAIN_W, 43)
            )
            s = tf.reduce_sum(lidar_bbs[..., -2] * 5 + lidar_bbs[..., -1], [1, 2])
            _, indices = tf.nn.top_k(s, lidar_bbs.shape[0] // 2)
            lidar_bbs = tf.gather(lidar_bbs, indices)
        else:
            lidar_bbs = tf.pad(
                lidar_bbs,
                [
                    [0, TEST_BATCH_SIZE - tf.shape(lidar_bbs)[0]],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
            )
            lidar_bbs = tf.reshape(lidar_bbs, (TEST_BATCH_SIZE, TEST_H, TEST_W, 43))
        lidar, bbs = lidar_bbs[..., :41], lidar_bbs[..., -2:]
        pedestrian_bbs, vehicle_bbs = bbs[..., 0:1], bbs[..., 1:]
        for _ in range(3):
            pedestrian_bbs = tf.math.maximum(
                pedestrian_bbs, tfa.image.gaussian_filter2d(pedestrian_bbs, (3, 3))
            )
        for _ in range(12):
            vehicle_bbs = tf.math.maximum(
                vehicle_bbs, tfa.image.gaussian_filter2d(vehicle_bbs, (5, 5))
            )
        bbs = tf.concat([pedestrian_bbs, vehicle_bbs], -1)
        lidar_bbs = tf.concat([lidar, bbs], -1)
    return lidar_bbs, bbs


@tf.function
def train_preprocess(frame, dataset_dir):
    return preprocess(frame, dataset_dir, True)


@tf.function
def test_preprocess(frame, dataset_dir):
    return preprocess(frame, dataset_dir, False)


def create_tf_dataset(dataset_path="../train/3d_labels"):
    train_ds = dataset.get_dataset(dataset_path)
    test_ds = dataset.get_dataset(dataset_path, False)
    train_ds = (
        train_ds.shuffle(buffer_size=8192)
        .batch(TRAIN_BATCH_SIZE)
        .map(lambda x: train_preprocess(x, dataset_path), num_parallel_calls=2)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    test_ds = (
        test_ds.batch(TEST_BATCH_SIZE).map(
            lambda x: test_preprocess(x, dataset_path), num_parallel_calls=1
        )
        # .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    return train_ds, test_ds  # , train_len, test_len


@tf.function
def mae(y_val, y_pred):
    d = tf.reduce_mean(tf.math.abs(y_val - y_pred)) * (
        999 * tf.cast(y_val - y_pred > 0, tf.float32) + 1
    )
    return d


@tf.function
def ce(y_val, y_pred):
    d = (
        -y_val * tf.math.log(tf.clip_by_value(y_pred, 0, 1) + 1e-10)
        - (1 - y_val) * tf.math.log(tf.clip_by_value(1 - y_pred, 0, 1) + 1e-10)
    ) * (49 * tf.cast(y_val - y_pred > 0, tf.float32) + 1)
    return tf.reduce_mean(d) * 10


def main():
    train_ds, test_ds = create_tf_dataset(args.path)
    detector = Detector()
    adabelief = tfa.optimizers.AdaBelief(learning_rate=2e-4)
    detector.compile(
        loss=ce,
        optimizer=adabelief,
        metrics=ce,
    )

    checkpoint_dir = "checkpoints"
    checkpoint_path = f"{checkpoint_dir}/bev.ckpt"
    if os.path.exists(checkpoint_dir):
        ckpt = tf.train.Checkpoint(detector)
        ckpt.restore(checkpoint_path)
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, save_best_only=True, save_weights_only=True, verbose=1
    )

    tb_callback = tf.keras.callbacks.TensorBoard(
        "./tb_logs", update_freq=7  # , profile_batch=(20, 30)
    )
    detector.fit(
        train_ds,
        epochs=200,
        callbacks=[tb_callback, ckpt_callback],
        validation_data=test_ds,
        validation_freq=1,
    )


if __name__ == "__main__":
    main()
