import os
import dataset

os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
import tensorflow as tf
import numpy as np
import argparse
import tensorflow_addons as tfa
from bev_model import DetectorTrainer

TRAIN_BATCH_SIZE = 28
TEST_BATCH_SIZE = 2
TRAIN_H, TRAIN_W = 256, 256
TEST_H, TEST_W = 512, 512


@tf.function
def path_to_raw(frame, dataset_dir):
    fmt = {
        "lidar_gzip_path": tf.io.FixedLenFeature([], tf.string),
        "bbs": tf.io.FixedLenFeature([], tf.string),
        "orientation_msb_path": tf.io.FixedLenFeature([], tf.string),
        "orientation_lsb_path": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(frame, fmt)
    lidar = tf.io.read_file(dataset_dir + "/" + example["lidar_gzip_path"])
    bbs = example["bbs"]
    orientation_msb_path = dataset_dir + "/" + example["orientation_msb_path"]
    orientation_lsb_path = dataset_dir + "/" + example["orientation_lsb_path"]
    orientation_msb = tf.io.read_file(orientation_msb_path)
    orientation_lsb = tf.io.read_file(orientation_lsb_path)
    return {"lidar": lidar, "bbs": bbs, "msb": orientation_msb, "lsb": orientation_lsb}


@tf.function
def train_path_to_lidar_bbs(frame):
    # lidar, bbs, orientation_msb, orientation_lsb = frame
    lidar = frame["lidar"]
    bbs = frame["bbs"]
    orientation_msb = frame["msb"]
    orientation_lsb = frame["lsb"]
    lidar = tf.io.decode_compressed(lidar, compression_type="GZIP")
    lidar = tf.io.decode_raw(lidar, tf.uint16)
    lidar = tf.reshape(lidar, (512, 512, 40))
    bbs = tf.io.decode_raw(bbs, tf.float64)
    bbs = tf.reshape(bbs, (-1, 11))
    orientation_msb = tf.image.decode_image(
        orientation_msb, channels=3, name="decode_orientation_msb"
    )
    orientation_lsb = tf.image.decode_image(
        orientation_lsb, channels=3, name="decode_orientation_lsb"
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
def test_path_to_lidar_bbs(frame):
    # lidar, bbs, orientation_msb, orientation_lsb = frame
    lidar = frame["lidar"]
    bbs = frame["bbs"]
    orientation_msb = frame["msb"]
    orientation_lsb = frame["lsb"]
    # lidar, bbs, orientation_msb, orientation_lsb = path_to_raw(frame, dataset_dir)
    lidar = tf.io.decode_compressed(lidar, compression_type="GZIP")
    lidar = tf.io.decode_raw(lidar, tf.uint16)
    lidar = tf.reshape(lidar, (512, 512, 40))
    bbs = tf.io.decode_raw(bbs, tf.float64)
    bbs = tf.reshape(bbs, (-1, 11))
    orientation_msb = tf.image.decode_image(
        orientation_msb, channels=3, name="decode_orientation_msb"
    )
    orientation_lsb = tf.image.decode_image(
        orientation_lsb, channels=3, name="decode_orientation_lsb"
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
                lambda x: train_path_to_lidar_bbs(x),
                frames,
                dtype=tf.float32,
                parallel_iterations=TRAIN_BATCH_SIZE,
            )
            # lidar_bbs = tf.vectorized_map(
            # train_path_to_lidar_bbs,
            # frames,
            # )
        else:
            lidar_bbs = tf.map_fn(
                lambda x: test_path_to_lidar_bbs(x),
                frames,
                dtype=tf.float32,
                parallel_iterations=TEST_BATCH_SIZE,
            )
            # lidar_bbs = tf.vectorized_map(
            # test_path_to_lidar_bbs,
            # frames,
            # )
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
            # lidar_bbs = tf.image.random_crop(
            # lidar_bbs, (TRAIN_BATCH_SIZE, TRAIN_H, TRAIN_W, 43)
            # )
            lidar_bbs = tf.image.random_flip_left_right(lidar_bbs)
            angles = tf.random.uniform([TRAIN_BATCH_SIZE * 2], 0, 2 * np.pi)
            # angles = tf.random.uniform([TRAIN_BATCH_SIZE], 0, 2 * np.pi)
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
            dz = 2
            shift = tf.random.uniform(
                shape=[lidar.shape[0]], minval=-dz, maxval=dz, dtype=tf.int32
            )
            lidar = tf.pad(lidar, [[0, 0], [0, 0], [0, 0], [dz, dz]], mode="SYMMETRIC")
            lidar = tf.roll(lidar, shift=shift, axis=tf.fill([lidar.shape[0]], 3))
            lidar = lidar[..., 2:-2]
            # lidar = lidar[..., 15:-5]
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
            # lidar_bbs = tf.image.random_crop(
            # lidar_bbs, (TRAIN_BATCH_SIZE, TRAIN_H, TRAIN_W, 43)
            # )
            s = tf.reduce_sum(lidar_bbs[..., -2] * 5 + lidar_bbs[..., -1], [1, 2])
            _, indices = tf.nn.top_k(s, lidar_bbs.shape[0] // 2)
            lidar_bbs = tf.gather(lidar_bbs, indices)
            lidar, bbs = lidar_bbs[..., :41], lidar_bbs[..., -2:]
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
            # lidar, bbs = (
            # tf.concat([lidar_bbs[..., 15:55], lidar_bbs[..., 60:61]], -1),
            # lidar_bbs[..., -2:],
            # )
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


# TODO 40m/50m filter by euclid dist


@tf.function
def train_preprocess(frame, dataset_dir):
    return preprocess(frame, dataset_dir, True)


@tf.function
def test_preprocess(frame, dataset_dir):
    return preprocess(frame, dataset_dir, False)


def create_tf_dataset(dataset_path="../train/3d_labels", oversample=True):
    train_ds = dataset.get_dataset(dataset_path)
    test_ds = dataset.get_dataset(dataset_path, False)
    train_scenes = {}
    for d in train_ds:
        fmt = {
            "lidar_gzip_path": tf.io.FixedLenFeature([], tf.string),
            "bbs": tf.io.FixedLenFeature([], tf.string),
            "orientation_msb_path": tf.io.FixedLenFeature([], tf.string),
            "orientation_lsb_path": tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(d, fmt)
        scene_idx = (
            example["lidar_gzip_path"]
            .numpy()
            .decode("utf-8")
            .split("/")[-1]
            .split("_")[0]
        )
        if scene_idx in train_scenes.keys():
            train_scenes[scene_idx].append(d)
        else:
            train_scenes[scene_idx] = [d]
    if oversample:
        max_frames = 0
        for scene_idx, scenes in train_scenes.items():
            max_frames = max(max_frames, len(scenes))
        train_ds = []
        for scene_idx, scenes in train_scenes.items():
            if scene_idx not in ["29", "30", "31", "32", "33", "34", "35"]:
                continue
            frames = len(scenes)
            r = int(np.round((max_frames / frames) ** 0.5))
            for _ in range(r):
                for scene in scenes:
                    train_ds.append(scene)
        train_ds = tf.data.Dataset.from_tensor_slices(train_ds)

    train_ds = (
        train_ds.shuffle(buffer_size=8192)
        .map(lambda x: path_to_raw(x, dataset_path))
        .cache()
        .batch(TRAIN_BATCH_SIZE)
        .map(lambda x: train_preprocess(x, dataset_path), num_parallel_calls=2)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    test_ds = (
        test_ds.map(lambda x: path_to_raw(x, dataset_path))
        .cache()
        .batch(TEST_BATCH_SIZE)
        .map(lambda x: test_preprocess(x, dataset_path), num_parallel_calls=2)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    return train_ds, test_ds  # , train_len, test_len


@tf.function
def mae(y_val, y_pred):
    d = tf.reduce_mean(tf.math.abs(y_val - y_pred)) * (
        999 * tf.cast(y_val - y_pred > 0, tf.float32) + 1
    )
    return d


@tf.custom_gradient
def clip(x):
    y = tf.clip_by_value(x, 0, 6)

    @tf.function
    def backward(w):
        return w * tf.math.abs(y - 3) / 3

    return y, backward


@tf.function
def ce(y_val, y_pred):
    # y_pred = clip(y_pred) / 6
    d = (
        -y_val * tf.math.log(tf.clip_by_value(y_pred, 0, 1) + 1e-10)
        - (1 - y_val) * tf.math.log(tf.clip_by_value(1 - y_pred, 0, 1) + 1e-10)
    ) * (49 * tf.cast(y_val - y_pred > 0, tf.float32) + 1)
    return tf.reduce_mean(d) * 10


@tf.function
def pedestrian_ce(y_val, y_pred):
    y_pred = y_pred[..., 0]
    y_val = y_val[..., 0]
    d = (
        -y_val * tf.math.log(tf.clip_by_value(y_pred, 0, 1) + 1e-10)
        - (1 - y_val) * tf.math.log(tf.clip_by_value(1 - y_pred, 0, 1) + 1e-10)
    ) * (49 * tf.cast(y_val - y_pred > 0, tf.float32) + 1)
    return tf.reduce_mean(d) * 10


@tf.function
def vehicle_ce(y_val, y_pred):
    y_pred = y_pred[..., 1]
    y_val = y_val[..., 1]
    d = (
        -y_val * tf.math.log(tf.clip_by_value(y_pred, 0, 1) + 1e-10)
        - (1 - y_val) * tf.math.log(tf.clip_by_value(1 - y_pred, 0, 1) + 1e-10)
    ) * (49 * tf.cast(y_val - y_pred > 0, tf.float32) + 1)
    return tf.reduce_mean(d) * 10


def main():
    parser = argparse.ArgumentParser(description="Train BEV model")
    parser.add_argument("--dataset", type=str, help="The training/test dataset path")
    parser.add_argument("--ckpt", type=str, help="The checkpoint path")
    parser.add_argument(
        "--epoch", type=int, default=100, help="The max number of epochs"
    )
    args = parser.parse_args()

    train_ds, test_ds = create_tf_dataset(args.path)
    detector = DetectorTrainer()
    adabelief = tfa.optimizers.AdaBelief(learning_rate=1e-3, weight_decay=1e-3)
    detector.compile(
        loss=ce,
        optimizer=adabelief,
        metrics=[ce, pedestrian_ce, vehicle_ce],
    )
    if False:
        from keras_flops import get_flops

        detector_trainer = DetectorTrainer((1152, 1152, 41))
        flops = get_flops(detector_trainer.detector, batch_size=1)
        print(detector_trainer.detector.summary())
        print(flops)
        exit()

    checkpoint_dir = "checkpoints"
    train_checkpoint_dir = "train_checkpoints"
    best_checkpoint_dir = "best_checkpoint"
    best_checkpoint_path = f"{best_checkpoint_dir}/bev.ckpt"
    checkpoint_path = f"{checkpoint_dir}/{{epoch:03d}}/bev.ckpt"
    train_checkpoint_path = f"{train_checkpoint_dir}/{{epoch:03d}}_{{loss:.4f}}_{{pedestrian_ce:.4f}}_{{vehicle_ce:.4f}}/bev.ckpt"
    if os.path.exists(best_checkpoint_dir):
        ckpt = tf.train.Checkpoint(detector)
        ckpt.restore(best_checkpoint_path)
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_ce",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    train_ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        train_checkpoint_path,
        monitor="loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    best_ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        best_checkpoint_path,
        monitor="val_ce",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    tb_callback = tf.keras.callbacks.TensorBoard(
        "./tb_logs", update_freq=1  # , profile_batch=(20, 30)
    )
    detector.fit(
        train_ds,
        epochs=300,
        callbacks=[tb_callback, ckpt_callback, best_ckpt_callback, train_ckpt_callback],
        validation_data=test_ds,
        validation_freq=5,
    )


if __name__ == "__main__":
    main()
