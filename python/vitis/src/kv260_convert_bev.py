import os
import dataset
import numpy as np
import argparse

os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
import tensorflow as tf
import tensorflow_addons as tfa
from bev_model import DetectorTrainer, get_pedestrian_detector_layers

TRAIN_BATCH_SIZE = 1
TRAIN_H, TRAIN_W = 256, 256
TEST_BATCH_SIZE = 1
TEST_H, TEST_W = 1152, 1152


@tf.function
def path_to_raw(frame, dataset_dir):
    fmt = {
        "lidar_gzip_path": tf.io.FixedLenFeature([], tf.string),
        "bbs": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(frame, fmt)
    lidar = tf.io.read_file(dataset_dir + "/" + example["lidar_gzip_path"])
    bbs = example["bbs"]
    return {"lidar": lidar, "bbs": bbs}


@tf.function
def train_path_to_lidar_bbs(frame):
    # lidar, bbs, orientation_msb, orientation_lsb = frame
    lidar = frame["lidar"]
    bbs = frame["bbs"]
    lidar = tf.io.decode_compressed(lidar, compression_type="GZIP")
    lidar = tf.io.decode_raw(lidar, tf.uint8)
    lidar = tf.reshape(lidar, (512, 512, 30))
    bbs = tf.io.decode_raw(bbs, tf.float64)
    bbs = tf.reshape(bbs, (-1, 11))

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
    lidar = tf.cast(lidar, tf.float32) / 255
    center_image = tf.cast(center_image, tf.float32)
    lidar_bbs = tf.concat([lidar, center_image], -1)
    return lidar_bbs


@tf.function
def test_path_to_lidar_bbs(frame):
    lidar = frame["lidar"]
    bbs = frame["bbs"]
    lidar = tf.io.decode_compressed(lidar, compression_type="GZIP")
    lidar = tf.io.decode_raw(lidar, tf.uint8)
    lidar = tf.reshape(lidar, (1152, 1152, 30))
    bbs = tf.io.decode_raw(bbs, tf.float64)
    bbs = tf.reshape(bbs, (-1, 11))

    center_image = tf.zeros((1152, 1152, 2), tf.bool)
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
                bbs = tf.clip_by_value(bbs, 0, 1151)
                updates = tf.ones((tf.shape(bbs)[0]), tf.bool)
                center_image = tf.tensor_scatter_nd_update(center_image, bbs, updates)
    lidar = tf.cast(lidar, tf.float32) / 255
    center_image = tf.cast(center_image, tf.float32)
    lidar_bbs = tf.concat([lidar, center_image], -1)
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
        else:
            lidar_bbs = tf.map_fn(
                lambda x: test_path_to_lidar_bbs(x),
                frames,
                dtype=tf.float32,
                parallel_iterations=TEST_BATCH_SIZE,
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
                    lidar_bbs, (TRAIN_BATCH_SIZE, TRAIN_H, TRAIN_W, 32)
                )
                for _ in range(2)
            ]
            lidar_bbs = tf.concat(lidar_bbs, 0)
            lidar_bbs = tf.image.random_flip_left_right(lidar_bbs)
            angles = tf.random.uniform([TRAIN_BATCH_SIZE * 2], 0, 2 * np.pi)
            lidar, bbs = (
                lidar_bbs[..., :30],
                lidar_bbs[..., 30:],
            )
            delta = tf.random.uniform(lidar.shape, 0.8, 1.2)
            lidar = tf.math.exp(tf.math.log(lidar + 1e-10) * delta)
            if False:
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
            dz = 3
            shift = tf.random.uniform(
                shape=[lidar.shape[0]], minval=-dz, maxval=dz, dtype=tf.int32
            )
            lidar = tf.roll(lidar, shift=shift, axis=tf.fill([lidar.shape[0]], 3))
            lidar = lidar[..., dz:-dz]
            lidar = tfa.image.rotate(lidar, angles, fill_mode="constant")
            bbs = tfa.image.rotate(bbs, angles, fill_mode="constant")
            lidar_bbs = tf.concat([lidar, bbs], -1)
            s = tf.reduce_sum(lidar_bbs[..., -2] * 5 + lidar_bbs[..., -1], [1, 2])
            _, indices = tf.nn.top_k(s, lidar_bbs.shape[0] // 2)
            lidar_bbs = tf.gather(lidar_bbs, indices)
            lidar, bbs = lidar_bbs[..., :24], lidar_bbs[..., -2:]
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
            lidar_bbs = tf.reshape(lidar_bbs, (TEST_BATCH_SIZE, TEST_H, TEST_W, 32))
            lidar, bbs = (
                lidar_bbs[..., 1:25],
                lidar_bbs[..., -2:],
            )
        pedestrian_bbs, vehicle_bbs = bbs[..., 0:1], bbs[..., 1:]
        for _ in range(3):
            pedestrian_bbs = tf.math.maximum(
                pedestrian_bbs, tfa.image.gaussian_filter2d(pedestrian_bbs, (3, 3))
            )
        for _ in range(8):
            vehicle_bbs = tf.math.maximum(
                vehicle_bbs, tfa.image.gaussian_filter2d(vehicle_bbs, (5, 5))
            )
        bbs = tf.concat([pedestrian_bbs, vehicle_bbs], -1)
        lidar_bbs = tf.concat([lidar, bbs], -1)
    return lidar, bbs


@tf.function
def train_preprocess(frame, dataset_dir):
    return preprocess(frame, dataset_dir, True)


@tf.function
def test_preprocess(frame, dataset_dir):
    return preprocess(frame, dataset_dir, False)


def create_tf_dataset(dataset_path="../train/3d_labels", oversample=True):
    train_ds = dataset.get_dataset(dataset_path)
    test_ds = dataset.get_dataset(dataset_path, False, False)
    calib_ds = dataset.get_dataset(dataset_path, True, False)

    if oversample:
        train_scenes = {}
        calib_scenes = {}
        for d in calib_ds:
            fmt = {
                "lidar_gzip_path": tf.io.FixedLenFeature([], tf.string),
                "bbs": tf.io.FixedLenFeature([], tf.string),
            }
            example = tf.io.parse_single_example(d, fmt)
            scene_idx = (
                example["lidar_gzip_path"]
                .numpy()
                .decode("utf-8")
                .split("/")[-1]
                .split("_")[0]
            )
            if scene_idx in calib_scenes.keys():
                calib_scenes[scene_idx].append(d)
            else:
                calib_scenes[scene_idx] = [d]
        max_frames = 0
        for scene_idx, scenes in calib_scenes.items():
            max_frames = max(max_frames, len(scenes))
        calib_ds = []
        for scene_idx, scenes in calib_scenes.items():
            if scene_idx not in ["29", "30", "31", "32", "33", "34", "35"]:
                continue
            frames = len(scenes)
            r = int(np.round((max_frames / frames) ** 0.5))
            for _ in range(r):
                for scene in scenes:
                    calib_ds.append(scene)
        calib_ds = tf.data.Dataset.from_tensor_slices(calib_ds)
        for d in train_ds:
            fmt = {
                "lidar_gzip_path": tf.io.FixedLenFeature([], tf.string),
                "bbs": tf.io.FixedLenFeature([], tf.string),
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
        train_ds.map(lambda x: path_to_raw(x, dataset_path))
        .cache()
        .shuffle(buffer_size=8192)
        .batch(TRAIN_BATCH_SIZE)
        .map(lambda x: train_preprocess(x, dataset_path), num_parallel_calls=1)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    test_ds = (
        test_ds.map(lambda x: path_to_raw(x, dataset_path))
        .cache()
        .batch(TEST_BATCH_SIZE)
        .map(lambda x: test_preprocess(x, dataset_path), num_parallel_calls=1)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    calib_ds = (
        calib_ds.map(lambda x: path_to_raw(x, dataset_path))
        .cache()
        .shuffle(buffer_size=8192)
        .batch(TEST_BATCH_SIZE)
        .map(lambda x: test_preprocess(x, dataset_path), num_parallel_calls=1)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    return train_ds, test_ds, calib_ds


@tf.function
def pedestrian_ce(y_val, y_pred):
    y_pred = y_pred[..., 0]
    y_val = y_val[..., 0]
    d = (
        -y_val * tf.math.log(tf.clip_by_value(y_pred, 0, 1) + 1e-8)
        - (1 - y_val) * tf.math.log(tf.clip_by_value(1 - y_pred, 0, 1) + 1e-8)
    ) * (9 * tf.cast(y_val - y_pred > 0, tf.float32) + 1)
    return tf.reduce_mean(d) * 30


@tf.function
def vehicle_ce(y_val, y_pred):
    y_pred = y_pred[..., 1]
    y_val = y_val[..., 1]
    d = (
        -y_val * tf.math.log(tf.clip_by_value(y_pred, 0, 1) + 1e-8)
        - (1 - y_val) * tf.math.log(tf.clip_by_value(1 - y_pred, 0, 1) + 1e-8)
    ) * (4 * tf.cast(y_val - y_pred > 0, tf.float32) + 1)
    return tf.reduce_mean(d) * 10


@tf.function
def ce(y_val, y_pred):
    return pedestrian_ce(y_val, y_pred) + vehicle_ce(y_val, y_pred)


def main():
    parser = argparse.ArgumentParser(description="Train BEV model")
    parser.add_argument("--dataset", type=str, help="The training/test dataset path")
    parser.add_argument("--pckpt", type=str, help="The checkpoint path")
    parser.add_argument("--vckpt", type=str, help="The checkpoint path")
    parser.add_argument(
        "--model-size", type=str, default="small", help="Model size (small or large)"
    )
    parser.add_argument("--inspect", action="store_true", help="Inspect model or not")

    args = parser.parse_args()

    train_ds, test_ds, calib_ds = create_tf_dataset(args.dataset)
    detector_trainer = DetectorTrainer(input_shape=(None, None, 24), for_vitis=False)

    pckpt = tf.train.Checkpoint(detector_trainer)
    pckpt.restore(args.pckpt)

    pedestrian_layer_names = get_pedestrian_detector_layers(args.model_size)

    pedestrian_weights = {}
    for name in pedestrian_layer_names:
        pedestrian_weights[name] = detector_trainer.detector.get_layer(
            name
        ).get_weights()

    vckpt = tf.train.Checkpoint(detector_trainer)
    vckpt.restore(args.vckpt)

    for name in pedestrian_layer_names:
        detector_trainer.detector.get_layer(name).set_weights(pedestrian_weights[name])

    detector = detector_trainer.detector

    from tensorflow_model_optimization.quantization.keras import (
        vitis_quantize,
        vitis_inspect,
    )

    if args.inspect:
        # inspector = vitis_inspect.VitisInspector(target="DPUCZDX8G_ISA1_B3136")
        inspector = vitis_inspect.VitisInspector(target="DPUCADF8H_ISA0")
        inspector.inspect_model(
            detector,
            # convert_bn_to_dwconv=False,
            plot=True,
            plot_file="model.svg",
            dump_results=True,
            dump_results_file="inspect_results.txt",
            verbose=1,
        )

    quantizer = vitis_quantize.VitisQuantizer(detector)
    quantized_model = quantizer.quantize_model(
        calib_dataset=train_ds,
        calib_steps=50,
        calib_batch_size=1,
        ignore_layers=["activation"],
        input_symmetry=False,
        weight_symmetry=False,
        bias_symmetry=False,
        include_fast_ft=True,
        fast_ft_epochs=20,
        # include_cle=False,
        convert_sigmoid_to_hard_sigmoid=False,
    )

    quantized_model.save("quantized_model.h5")
    detector.compile(
        loss=ce,
        metrics=[ce, pedestrian_ce, vehicle_ce],
    )
    detector.evaluate(test_ds)
    quantized_model.compile(
        loss=ce,
        metrics=[ce, pedestrian_ce, vehicle_ce],
    )
    quantized_model.evaluate(test_ds)


if __name__ == "__main__":
    main()
