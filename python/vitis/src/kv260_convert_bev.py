import os
import dataset

os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
import tensorflow as tf
import numpy as np
import argparse
import tensorflow_addons as tfa
from bev_model import DetectorTrainer

TEST_BATCH_SIZE = 1
TEST_H, TEST_W = 1152, 1152

parser = argparse.ArgumentParser(description="Train BEV model")
parser.add_argument("--path", type=str, help="The training/test dataset path")
parser.add_argument("--ckpt", type=str, help="The checkpoint path")
parser.add_argument("--epoch", type=int, default=100, help="The max number of epochs")
args = parser.parse_args()


@tf.function
def test_path_to_lidar_bbs(frame, dataset_dir):
    fmt = {
        "lidar_gzip_path": tf.io.FixedLenFeature([], tf.string),
        "bbs": tf.io.FixedLenFeature([], tf.string),
        "orientation_msb_path": tf.io.FixedLenFeature([], tf.string),
        "orientation_lsb_path": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(frame, fmt)
    lidar = tf.io.read_file(dataset_dir + "/" + example["lidar_gzip_path"])
    lidar = tf.io.decode_compressed(lidar, compression_type="GZIP")
    lidar = tf.io.decode_raw(lidar, tf.uint16)
    lidar = tf.reshape(lidar, (1152, 1152, 40))
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
    lidar = tf.cast(lidar, tf.float32) / 65535
    center_image = tf.cast(center_image, tf.float32)
    lidar_bbs = tf.concat([lidar, orientation_image[..., -1:], center_image], -1)
    return lidar_bbs


@tf.function
def preprocess(frames, dataset_dir, training=True):
    with tf.device("/gpu:0"):
        lidar_bbs = tf.map_fn(
            lambda x: test_path_to_lidar_bbs(x, dataset_dir),
            frames,
            dtype=tf.float32,
            parallel_iterations=1,
        )
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
        # lidar_bbs = tf.concat([lidar, bbs], -1)
    return lidar, bbs  # _bbs, bbs


@tf.function
def test_preprocess(frame, dataset_dir):
    return preprocess(frame, dataset_dir, False)


def create_tf_dataset(dataset_path="../train/3d_labels", oversample=True):
    calib_ds = dataset.get_dataset(dataset_path, True, False)

    calib_scenes = {}
    for d in calib_ds:
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
        if scene_idx in calib_scenes.keys():
            calib_scenes[scene_idx].append(d)
        else:
            calib_scenes[scene_idx] = [d]
    if oversample:
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

    calib_ds = (
        calib_ds.shuffle(buffer_size=8192)
        .batch(TEST_BATCH_SIZE)
        .map(lambda x: test_preprocess(x, dataset_path), num_parallel_calls=1)
        # .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    test_ds = dataset.get_dataset(dataset_path, False, False)
    test_ds = (
        test_ds.shuffle(buffer_size=8192)
        .batch(TEST_BATCH_SIZE)
        .map(lambda x: test_preprocess(x, dataset_path), num_parallel_calls=1)
        # .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    return calib_ds, test_ds


@tf.function
def mae(y_val, y_pred):
    d = tf.reduce_mean(tf.math.abs(y_val - y_pred)) * (
        999 * tf.cast(y_val - y_pred > 0, tf.float32) + 1
    )
    return d


@tf.function
def ce(y_val, y_pred):
    # y_pred = tf.keras.layers.Activation("sigmoid")(y_pred)
    d = (
        -y_val * tf.math.log(tf.clip_by_value(y_pred, 0, 1) + 1e-10)
        - (1 - y_val) * tf.math.log(tf.clip_by_value(1 - y_pred, 0, 1) + 1e-10)
    ) * (49 * tf.cast(y_val - y_pred > 0, tf.float32) + 1)
    return tf.reduce_mean(d) * 10


def main():
    evaluate = True
    calib_ds, test_ds = create_tf_dataset(args.path)
    detector_trainer = DetectorTrainer(
        input_shape=(1152, 1152, 41), training=False, for_vitis=True
    )

    checkpoint_dir = "best_checkpoint"
    checkpoint_path = f"{checkpoint_dir}/bev.ckpt"
    if os.path.exists(checkpoint_dir):
        ckpt = tf.train.Checkpoint(detector_trainer)
        ckpt.restore(checkpoint_path)

    detector = detector_trainer.detector

    from tensorflow_model_optimization.quantization.keras import (
        vitis_quantize,
        vitis_inspect,
    )

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
    # quantizer.dump_quantize_strategy(dump_file="my_quantize_strategy.json", verbose=0)
    quantized_model = quantizer.quantize_model(
        calib_dataset=calib_ds,
        calib_steps=50,
        calib_batch_size=1,
        ignore_layers=["activation"],
        # input_symmetry=False,
        # weight_symmetry=False,
        # bias_symmetry=False,
        convert_sigmoid_to_hard_sigmoid=False,
        # replace_relu6=True,
        include_cle=False,
        # convert_relu_to_relu6=True,
        # weight_per_channel=True,
        # bias_per_channel=True,
        # weight_method=1,
        # activation_per_channel=True,
        # include_fast_ft=True,
        # fast_ft_epochs=10,
        # separate_conv_act=False
    )
    quantized_model.save("quantized_model.h5")
    w = tf.summary.create_file_writer("tb_logs")
    with w.as_default():
        for n, (lidar, bbs) in enumerate(test_ds):
            print(n, lidar.shape[0])
            quantized_pred = quantized_model.predict(lidar)
            pred = detector.predict(lidar)
            # pred = tf.concat([quantized_pred, pred], -1)
            # print(pred[0, :16, :16])
            # break
            summary_image = tf.concat([pred, quantized_pred], 2)
            summary_image = tf.concat(
                [summary_image, tf.zeros([*summary_image.shape[:-1], 1])], -1
            )
            tf.summary.image("summary", summary_image, step=n)
    # print(n, pred)
    if evaluate:
        detector.compile(
            loss=ce,
            metrics=ce,
        )
        detector.evaluate(test_ds)
        quantized_model.compile(
            loss=ce,
            metrics=ce,
        )
        quantized_model.evaluate(test_ds)


if __name__ == "__main__":
    main()
