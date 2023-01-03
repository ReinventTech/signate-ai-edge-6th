import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D,
    DepthwiseConv2D,
    Conv2DTranspose,
    LayerNormalization as LN,
    BatchNormalization as BN,
    ReLU,
    Dropout,
)


def dw_conv2d(ksize, strides, activation, x):
    y = DepthwiseConv2D(
        ksize,
        strides,
        padding="same",
        # kernel_regularizer=tf.keras.regularizers.L1L2(1e-3, 1e-3),
        # bias_regularizer=tf.keras.regularizers.l2(1e-3),
        # activity_regularizer=tf.keras.regularizers.l2(1e-2),
    )(x)
    if activation is not None:
        y = activation(y)
        # y = LN()(y)
        # y = BN(momentum=0.9)(y)
    return y


def sep_conv2d(channels, ksize, strides, activation, x):
    y = DepthwiseConv2D(
        ksize,
        strides,
        padding="same",
        # kernel_regularizer=tf.keras.regularizers.L1L2(1e-2, 1e-2),
        # bias_regularizer=tf.keras.regularizers.l2(1e-2),
        # activity_regularizer=tf.keras.regularizers.l2(1e-2),
    )(x)
    y = ReLU()(y)
    # y = BN(momentum=0.9)(y)
    # y = LN()(y)
    y = conv2d(channels, 1, (1, 1), activation, y)
    # y = BN(momentum=0.9)(y)
    # y = LN()(y)
    return y


def conv2d(channels, ksize, strides, activation, x):
    y = Conv2D(
        channels,
        ksize,
        strides,
        padding="same",
        # kernel_regularizer=tf.keras.regularizers.L1L2(1e-2, 1e-2),
        # bias_regularizer=tf.keras.regularizers.l2(1e-2),
        # activity_regularizer=tf.keras.regularizers.l2(1e-2),
    )(x)
    if activation is not None:
        y = activation(y)
        # y = BN(momentum=0.9)(y)
        # y = LN()(y)
    return y


def conv2d_transpose(channels, ksize, strides, activation, x):
    y = Conv2DTranspose(
        channels,
        ksize,
        strides,
        padding="same",
        # kernel_regularizer=tf.keras.regularizers.L1L2(1e-2, 1e-2),
        # bias_regularizer=tf.keras.regularizers.l2(1e-2),
        # activity_regularizer=tf.keras.regularizers.l2(1e-2),
    )(x)
    if activation is not None:
        y = activation(y)
        # y = BN(momentum=0.9)(y)
        # y = LN()(y)
    return y


def resblock(x, channels, ksize=3, dw=False, training=True):
    s = x
    if dw:
        y = sep_conv2d(channels, ksize, (1, 1), ReLU(), x)
        y = sep_conv2d(channels, ksize, (1, 1), None, y)
    else:
        y = conv2d(channels, ksize, (1, 1), ReLU(), x)
        y = conv2d(channels, ksize, (1, 1), None, y)
    y = tf.keras.layers.Add()([s, y])
    y = ReLU()(y)
    return y


def resblock_v2(x, channels, ksize=3, dw=False, training=True):
    s = x
    y = conv2d(channels * 2, 1, (1, 1), ReLU(6), x)
    y = dw_conv2d(ksize, (1, 1), ReLU(6), y)
    y = conv2d(channels, 1, (1, 1), None, y)
    y = tf.keras.layers.Add()([s, y])
    y = ReLU()(y)
    return y


def sequential(x, layers, training=True):
    for layer in layers:
        x = layer(x, training=training)
    return x


def hard_sigmoid(x):
    return ReLU(6.0)(x + 3.0) * (1.0 / 6.0)


@tf.custom_gradient
def clip(x, min_value, max_value):
    y = tf.clip_by_value(x, min_value, max_value)

    @tf.function
    def backward(w):
        return w

    return y, backward


def get_detector_functional_model(x, training=True, for_vitis=False):
    deep = True
    # s = sep_sep_conv2d(32, 7, (2, 2), ReLU(), x)
    s = sep_conv2d(32, 7, (2, 2), ReLU(), x)
    x1 = resblock_v2(s, 32, 3, False, training=training)
    x1 = resblock_v2(x1, 32, 3, False, training=training)
    # x1 = tf.keras.layers.Add()([x1, s])
    # s = sep_sep_conv2d(64, 5, (2, 2), ReLU(), x1)
    s = sep_conv2d(64, 5, (2, 2), ReLU(), x1)
    x2 = resblock_v2(s, 64, 3, training=training)
    x2 = resblock_v2(x2, 64, 3, training=training)
    # x2 = tf.keras.layers.Add()([x2, s])
    # x2 = sep_conv2d(64, 3, (1, 1), ReLU(), x2)
    # if training:
    # x2 = tf.quantization.quantize_and_dequantize(x2, 0, 1)

    s = sep_conv2d(128, 3, (2, 2), ReLU(), x2)
    x3 = resblock_v2(s, 128, 3, training=training)
    x3 = resblock_v2(x3, 128, 3, training=training)
    # x3 = tf.keras.layers.Add()([x3, s])
    s = sep_conv2d(256, 3, (2, 2), ReLU(), x3)
    x4 = resblock_v2(s, 256, 3, False, training=training)
    x4 = resblock_v2(x4, 256, 3, False, training=training)
    # x4 = tf.keras.layers.Add()([x4, s])
    x4 = sep_conv2d(256, 3, (1, 1), ReLU(), x4)

    if deep:
        s = sep_conv2d(512, 3, (2, 2), ReLU(), x4)
        x5 = resblock_v2(s, 512, 3, False, training=training)
        x5 = resblock_v2(x5, 512, 3, False, training=training)
        # x5 = tf.keras.layers.Add()([x5, s])
        s = sep_conv2d(1024, 3, (2, 2), ReLU(), x5)
        x6 = resblock_v2(s, 1024, 3, False, training=training)
        x6 = resblock_v2(x6, 1024, 3, False, training=training)
        # x6 = tf.keras.layers.Add()([x6, s])
        x6 = sep_conv2d(1024, 3, (1, 1), ReLU(), x6)

        s = conv2d_transpose(512, 3, (2, 2), ReLU(), x6)
        s = tf.keras.layers.Concatenate(-1)([s, x5])
        y = resblock_v2(s, 1024, 3, False, training=training)
        y = resblock_v2(y, 1024, 3, False, training=training)
        # y = tf.keras.layers.Add()([y, s])
        s = conv2d_transpose(256, 3, (2, 2), ReLU(), y)
        s = tf.keras.layers.Concatenate(-1)([s, x4])
        y = resblock_v2(s, 512, 3, False, training=training)
        y = resblock_v2(y, 512, 3, False, training=training)
        # y = tf.keras.layers.Add()([y, s])
        y2 = sep_conv2d(256, 3, (1, 1), ReLU(), y)
    else:
        y2 = x4

    s = conv2d_transpose(128, 3, (2, 2), ReLU(), y2)
    s = tf.keras.layers.Concatenate(-1)([s, x3])
    y = resblock_v2(s, 256, 3, False, training=training)
    y = resblock_v2(y, 256, 3, False, training=training)
    # y = tf.keras.layers.Add()([y, s])
    s = conv2d_transpose(64, 3, (2, 2), ReLU(), y)
    s = tf.keras.layers.Concatenate(-1)([s, x2])
    y = resblock_v2(s, 128, 3, training=training)
    y = resblock_v2(y, 128, 3, training=training)
    # y = tf.keras.layers.Add()([y, s])
    y4 = sep_conv2d(64, 3, (1, 1), ReLU(), y)

    s = conv2d_transpose(32, 3, (2, 2), ReLU(), y4)
    s = tf.keras.layers.Concatenate(-1)([s, x1])

    y = resblock_v2(s, 64, 3, False, training=training)
    y = resblock_v2(y, 64, 3, False, training=training)
    # y = tf.keras.layers.Add()([y, s])

    s = conv2d_transpose(16, 3, (2, 2), ReLU(), y)
    y = resblock_v2(s, 16, 3, False, training=training)
    y = resblock_v2(y, 16, 3, False, training=training)
    # y = tf.keras.layers.Add()([y, s])

    y = conv2d(2, 3, (1, 1), None, y)
    # y = hard_sigmoid(y)
    # if not for_vitis:
    y = tf.keras.layers.Activation("sigmoid")(y)

    return y


class DetectorTrainer(Model):
    def __init__(
        self,
        input_shape=(None, None, 41),
        training=True,
        for_vitis=False,
        name="detector",
    ):
        super(DetectorTrainer, self).__init__(name=name)
        detector_input = tf.keras.layers.Input(input_shape)
        detector_output = get_detector_functional_model(
            detector_input, training, for_vitis
        )
        self.detector = tf.keras.Model(
            inputs=[detector_input], outputs=[detector_output], name="detector"
        )
        self.step = tf.Variable(0, dtype=tf.int64, trainable=False)

    def image_summary(self, x, y, g):
        s = y
        # lidar, orientation = x[..., :-3], x[..., -3:]
        # orientation = tf.concat(
        # [orientation[..., :1], (orientation[..., 1:] + 1) / 2], -1
        # )
        lidar, orientation = x[..., :-3], x[..., -1:]
        orientation = tf.tile(orientation, [1, 1, 1, 3])
        x = tf.reduce_max(lidar, -1)
        x = tf.stack([x, x, x], -1)
        y = tf.concat([y, tf.zeros((*y.shape[:3], 1))], -1)
        m = y > 0.7
        fy = y * tf.cast(m, tf.float32)
        pedestrian_ccs = tfa.image.connected_components(m[..., 0])
        vehicle_ccs = tfa.image.connected_components(m[..., 1])
        filtered_pedestrian_ccs = tf.zeros(pedestrian_ccs.shape, tf.bool)
        filtered_vehicle_ccs = tf.zeros(vehicle_ccs.shape, tf.bool)
        pedestrian_centroid_image = tf.zeros(pedestrian_ccs.shape, tf.float32)
        vehicle_centroid_image = tf.zeros(vehicle_ccs.shape, tf.float32)
        for n in range(1, 9):
            pedestrian_cc = pedestrian_ccs == n
            pedestrian_filter = (
                tf.reduce_sum(tf.cast(pedestrian_cc, tf.int32), (1, 2), keepdims=True)
                > 5
            )
            pedestrian_cc = tf.math.logical_and(pedestrian_cc, pedestrian_filter)
            filtered_pedestrian_ccs = tf.math.logical_or(
                filtered_pedestrian_ccs, pedestrian_cc
            )

            vehicle_cc = vehicle_ccs == n
            vehicle_filter = (
                tf.reduce_sum(tf.cast(vehicle_cc, tf.int32), (1, 2), keepdims=True) > 8
            )
            vehicle_cc = tf.math.logical_and(vehicle_cc, vehicle_filter)
            filtered_vehicle_ccs = tf.math.logical_or(filtered_vehicle_ccs, vehicle_cc)

            xs, ys = tf.meshgrid(tf.range(m.shape[2]), tf.range(m.shape[1]))
            yxs = tf.stack([ys, xs], -1)

            pedestrian_centroid = tf.expand_dims(yxs, 0) * tf.cast(
                tf.expand_dims(pedestrian_cc, -1), tf.int32
            )
            pedestrian_centroid = tf.cast(
                tf.math.round(
                    tf.cond(
                        tf.reduce_sum(tf.cast(pedestrian_cc, tf.int32)) > 0,
                        lambda: tf.reduce_sum(
                            tf.cast(pedestrian_centroid, tf.float32), (1, 2)
                        )
                        / tf.reduce_sum(tf.cast(pedestrian_cc, tf.float32)),
                        lambda: tf.reduce_sum(
                            tf.cast(pedestrian_centroid, tf.float32), (1, 2)
                        ),
                    )
                ),
                tf.int32,
            )
            pedestrian_centroid = tf.expand_dims(yxs, 0) == tf.reshape(
                pedestrian_centroid, (-1, 1, 1, 2)
            )
            pedestrian_centroid = tf.reduce_min(
                tf.cast(pedestrian_centroid, tf.float32), -1
            ) * tf.cast(
                tf.reduce_sum(tf.cast(pedestrian_cc, tf.int32), (1, 2), keepdims=True)
                > 0,
                tf.float32,
            )
            pedestrian_centroid_image = tf.math.maximum(
                pedestrian_centroid_image, pedestrian_centroid
            )

            vehicle_centroid = tf.expand_dims(yxs, 0) * tf.cast(
                tf.expand_dims(vehicle_cc, -1), tf.int32
            )
            vehicle_centroid = tf.cast(
                tf.math.round(
                    tf.cond(
                        tf.reduce_sum(tf.cast(vehicle_cc, tf.int32)) > 0,
                        lambda: tf.reduce_sum(
                            tf.cast(vehicle_centroid, tf.float32), (1, 2)
                        )
                        / tf.reduce_sum(tf.cast(vehicle_cc, tf.float32)),
                        lambda: tf.reduce_sum(
                            tf.cast(vehicle_centroid, tf.float32), (1, 2)
                        ),
                    )
                ),
                tf.int32,
            )
            vehicle_centroid = tf.expand_dims(yxs, 0) == tf.reshape(
                vehicle_centroid, (-1, 1, 1, 2)
            )
            vehicle_centroid = tf.reduce_min(
                tf.cast(vehicle_centroid, tf.float32), -1
            ) * tf.cast(
                tf.reduce_sum(tf.cast(vehicle_cc, tf.int32), (1, 2), keepdims=True) > 0,
                tf.float32,
            )
            vehicle_centroid_image = tf.math.maximum(
                vehicle_centroid_image, vehicle_centroid
            )

        filtered_ccs = tf.stack([filtered_pedestrian_ccs, filtered_vehicle_ccs], -1)
        filtered_ccs = tf.concat(
            [filtered_ccs, tf.zeros((m.shape[0], m.shape[1], m.shape[2], 1), tf.bool)],
            -1,
        )
        filtered_ccs = tf.cast(filtered_ccs, tf.float32)

        centroid_image = tf.stack(
            [
                pedestrian_centroid_image,
                vehicle_centroid_image,
                tf.zeros((m.shape[0], m.shape[1], m.shape[2]), tf.float32),
            ],
            -1,
        )
        centroid_image = (
            centroid_image
            + tf.pad(centroid_image[:, 2:, :, :], [[0, 0], [0, 2], [0, 0], [0, 0]])
            + tf.pad(centroid_image[:, 1:, :, :], [[0, 0], [0, 1], [0, 0], [0, 0]])
            + tf.pad(centroid_image[:, :-1, :, :], [[0, 0], [1, 0], [0, 0], [0, 0]])
            + tf.pad(centroid_image[:, :-2:, :, :], [[0, 0], [2, 0], [0, 0], [0, 0]])
        )
        centroid_image = (
            centroid_image
            + tf.pad(centroid_image[:, :, 2:, :], [[0, 0], [0, 0], [0, 2], [0, 0]])
            + tf.pad(centroid_image[:, :, 1:, :], [[0, 0], [0, 0], [0, 1], [0, 0]])
            + tf.pad(centroid_image[:, :, :-1, :], [[0, 0], [0, 0], [1, 0], [0, 0]])
            + tf.pad(centroid_image[:, :, :-2:, :], [[0, 0], [0, 0], [2, 0], [0, 0]])
        )
        centroid_image = tf.math.minimum(centroid_image, 1)

        g = tf.concat([g, tf.zeros((*y.shape[:3], 1))], -1)
        image = tf.concat(
            # [x, y, fy, centroid_image, g, orientation],
            [
                tf.pad(
                    x[:, 1:-1, 1:-1, :],
                    [[0, 0], [1, 1], [1, 1], [0, 0]],
                    constant_values=1,
                ),
                tf.pad(
                    y[:, 1:-1, 1:-1, :],
                    [[0, 0], [1, 1], [1, 1], [0, 0]],
                    constant_values=1,
                ),
                tf.pad(
                    fy[:, 1:-1, 1:-1, :],
                    [[0, 0], [1, 1], [1, 1], [0, 0]],
                    constant_values=1,
                ),
                tf.pad(
                    centroid_image[:, 1:-1, 1:-1, :],
                    [[0, 0], [1, 1], [1, 1], [0, 0]],
                    constant_values=1,
                ),
                tf.pad(
                    g[:, 1:-1, 1:-1, :],
                    [[0, 0], [1, 1], [1, 1], [0, 0]],
                    constant_values=1,
                ),
                tf.pad(
                    orientation[:, 1:-1, 1:-1, :],
                    [[0, 0], [1, 1], [1, 1], [0, 0]],
                    constant_values=1,
                ),
            ],
            axis=2,
        )
        tf.summary.image(
            "input/pred/filtered/centroid/gt/orientation",
            image,
            max_outputs=16,
            step=self.step,
        )
        return s

    def call(self, x, training=True):
        x, g = x[..., :-2], x[..., -2:]
        y = self.detector(x)
        y = tf.cond(
            self.step % 100 == 0, lambda: self.image_summary(x, y, g), lambda: y
        )
        self.step.assign_add(1)
        return y
