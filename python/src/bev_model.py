import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D,
    DepthwiseConv2D,
    Conv2DTranspose,
    # LayerNormalization as LN,
    # BatchNormalization as BN,
    ReLU,
    # Dropout,
    GlobalAveragePooling2D,
    UpSampling2D,
    Dense,
    Multiply,
)


def dw_conv2d(ksize, strides, activation, x):
    y = DepthwiseConv2D(
        ksize,
        strides,
        padding="same",
    )(x)
    if activation is not None:
        y = activation(y)
    return y


def sep_conv2d(channels, ksize, strides, activation, x):
    y = DepthwiseConv2D(
        ksize,
        strides,
        padding="same",
    )(x)
    y = ReLU(6)(y)
    y = conv2d(channels, 1, (1, 1), activation, y)
    return y


def conv2d(channels, ksize, strides, activation, x):
    y = Conv2D(
        channels,
        ksize,
        strides,
        padding="same",
    )(x)
    if activation is not None:
        y = activation(y)
    return y


def conv2d_transpose(channels, ksize, strides, activation, x):
    y = Conv2DTranspose(
        channels,
        ksize,
        strides,
        padding="same",
    )(x)
    if activation is not None:
        y = activation(y)
    return y


def resblock(x, channels, ksize=3, dw=False, training=True, activation="relu"):
    s = x
    if dw:
        y = sep_conv2d(channels, ksize, (1, 1), ReLU(6), x)
        y = sep_conv2d(channels, ksize, (1, 1), None, y)
    else:
        y = conv2d(channels, ksize, (1, 1), ReLU(6), x)
        y = conv2d(channels, ksize, (1, 1), None, y)
    y = tf.keras.layers.Add()([s, y])
    y = ReLU()(y)
    return y


def resblock_v2(x, channels, ksize=3, dw=False, training=True, activation="relu"):
    s = x
    y = conv2d(channels * 2, 1, (1, 1), ReLU(6), x)
    y = dw_conv2d(ksize, (1, 1), ReLU(6), y)
    y = conv2d(channels, 1, (1, 1), None, y)
    y = tf.keras.layers.Add()([s, y])
    y = ReLU()(y)
    return y


def resblock_v3(x, channels, ksize=3, dw=False, training=True, activation="relu"):
    s = x
    y = conv2d(
        channels * 2, 1, (1, 1), ReLU(6) if activation == "relu" else hard_swish, x
    )
    y = dw_conv2d(ksize, (1, 1), ReLU(6) if activation == "relu" else hard_swish, y)
    m = GlobalAveragePooling2D(keepdims=True)(y)
    m = Dense(channels)(m)
    if activation == "relu":
        m = ReLU()(m)
    else:
        m = hard_swish(m)
    m = Dense(channels * 2)(m)
    m = hard_sigmoid(m)
    m = tf.tile(m, [1, tf.shape(s)[1], tf.shape(s)[2], 1])
    # m = UpSampling2D(tf.shape(s)[1:3])(m)
    y = Multiply()([y, m])
    y = conv2d(channels, 1, (1, 1), None, y)
    y = tf.keras.layers.Add()([s, y])
    if activation == "relu":
        y = ReLU()(y)
    else:
        y = hard_swish(y)
    return y


def sequential(x, layers, training=True):
    for layer in layers:
        x = layer(x, training=training)
    return x


def hard_sigmoid(x):
    return ReLU(6.0)(x + 3.0) * (1.0 / 6.0)


def hard_swish(x):
    return Multiply()([x, hard_sigmoid(x)])


@tf.custom_gradient
def clip(x, min_value, max_value):
    y = tf.clip_by_value(x, min_value, max_value)

    @tf.function
    def backward(w):
        return w

    return y, backward


def get_pedestrian_detector_functional_model_small(x1, x2):
    x = UpSampling2D((2, 2))(x2)
    x = sep_conv2d(24, 3, (1, 1), ReLU(6), x)
    x = tf.keras.layers.Concatenate(-1)([x, x1])
    x = resblock_v2(x, 48, 3, False)
    x = UpSampling2D((2, 2))(x)
    x = sep_conv2d(8, 3, (1, 1), ReLU(6), x)
    x = resblock_v2(x, 8, 3, False)
    x = resblock_v2(x, 8, 3, False)
    x = conv2d(1, 3, (1, 1), None, x)
    return x


def get_base_detector_layers_small():
    return [
        "conv2d",
        "conv2d_1",
        "conv2d_2",
        "conv2d_3",
        "conv2d_4",
        "conv2d_5",
        "depthwise_conv2d",
        "depthwise_conv2d_1",
        "depthwise_conv2d_2",
    ]


def get_pedestrian_detector_layers_small():
    return [
        "conv2d_6",
        "conv2d_7",
        "conv2d_8",
        "conv2d_9",
        "conv2d_10",
        "conv2d_11",
        "conv2d_12",
        "conv2d_13",
        "conv2d_14",
        "depthwise_conv2d_3",
        "depthwise_conv2d_4",
        "depthwise_conv2d_5",
        "depthwise_conv2d_6",
        "depthwise_conv2d_7",
    ]


def get_pedestrian_detector_functional_model_large(x):
    x = conv2d(32, 7, (2, 2), ReLU(), x)
    x1 = resblock_v2(x, 32, 3, False)
    x = conv2d(64, 5, (2, 2), ReLU(), x1)
    x2 = resblock_v2(x, 64, 3)
    x = conv2d_transpose(32, 3, (2, 2), ReLU(), x2)
    x = tf.keras.layers.Concatenate(-1)([x, x1])
    x = resblock_v2(x, 64, 3, False)
    x = conv2d_transpose(16, 3, (2, 2), ReLU(), x)
    x = resblock_v2(x, 16, 3, False)
    x = conv2d(1, 3, (1, 1), None, x)
    return x


def get_base_detector_layers_large():
    return [
        "conv2d",
        "conv2d_1",
        "conv2d_2",
        "conv2d_3",
        "conv2d_4",
        "conv2d_5",
        "depthwise_conv2d",
        "depthwise_conv2d_1",
        "depthwise_conv2d_2",
    ]


def get_pedestrian_detector_layers_large():
    return [
        "conv2d",
        "conv2d_1",
        "depthwise_conv2d",
        "conv2d_2",
        "conv2d_3",
        "conv2d_4",
        "depthwise_conv2d_1",
        "conv2d_5",
        "conv2d_transpose",
        "conv2d_6",
        "depthwise_conv2d_2",
        "conv2d_7",
        "conv2d_transpose_1",
        "conv2d_8",
        "depthwise_conv2d_3",
        "conv2d_9",
        "conv2d_10",
    ]


def get_pedestrian_detector_layers(model_size="small"):
    if model_size == "small":
        return get_pedestrian_detector_layers_small()
    else:
        return get_pedestrian_detector_layers_large()


def get_base_detector_layers(model_size="small"):
    if model_size == "small":
        return get_base_detector_layers_small()
    else:
        return get_base_detector_layers_large()


def get_vehicle_detector_functional_model_small(x1, x2):
    x = sep_conv2d(96, 3, (2, 2), ReLU(6), x2)
    x3 = resblock_v2(x, 96, 3)
    x = sep_conv2d(192, 3, (2, 2), ReLU(6), x3)
    x = resblock_v2(x, 192, 3, False, activation="relu")
    x4 = sep_conv2d(192, 3, (1, 1), ReLU(6), x)

    x = UpSampling2D((2, 2))(x4)
    x = sep_conv2d(96, 3, (1, 1), ReLU(6), x)
    x = tf.keras.layers.Concatenate(-1)([x, x3])
    x = sep_conv2d(96, 3, (1, 1), ReLU(6), x)
    x = resblock_v2(x, 96, 3, False, activation="relu")
    x = UpSampling2D((2, 2))(x)
    x = sep_conv2d(48, 3, (1, 1), ReLU(6), x)
    x = tf.keras.layers.Concatenate(-1)([x, x2])
    x = sep_conv2d(48, 3, (1, 1), ReLU(6), x)
    x = resblock_v2(x, 48, 3)

    x = UpSampling2D((2, 2))(x)
    x = sep_conv2d(24, 3, (1, 1), ReLU(6), x)
    x = resblock_v2(x, 24, 3, False)
    x = resblock_v2(x, 24, 3, False)

    x = UpSampling2D((2, 2))(x)
    x = sep_conv2d(8, 3, (1, 1), ReLU(6), x)
    x = resblock_v2(x, 8, 3, False)
    x = resblock_v2(x, 8, 3, False)
    x = conv2d(1, 3, (1, 1), None, x)

    return x


def get_vehicle_detector_functional_model_large(x, include_last_activation=False):
    x = conv2d(32, 7, (2, 2), ReLU(), x)
    x1 = resblock_v2(x, 32, 3, False)
    x = conv2d(64, 5, (2, 2), ReLU(), x1)
    x2 = resblock_v2(x, 64, 3)

    x = conv2d(128, 3, (2, 2), ReLU(), x2)
    x3 = resblock_v2(x, 128, 3)
    x = conv2d(256, 3, (2, 2), ReLU(), x3)
    x4 = resblock_v2(x, 256, 3, False, activation="relu")
    x4 = conv2d(256, 3, (1, 1), ReLU(), x4)

    x = conv2d_transpose(128, 3, (2, 2), ReLU(), x4)
    x = tf.keras.layers.Concatenate(-1)([x, x3])
    x = resblock_v2(x, 256, 3, False, activation="relu")
    x = conv2d_transpose(64, 3, (2, 2), ReLU(), x)
    x = tf.keras.layers.Concatenate(-1)([x, x2])
    x = resblock_v2(x, 128, 3)
    x = conv2d(64, 3, (1, 1), ReLU(), x)

    x = conv2d_transpose(32, 3, (2, 2), ReLU(), x)
    x = tf.keras.layers.Concatenate(-1)([x, x1])
    x = resblock_v2(x, 64, 3, False)
    x = conv2d_transpose(16, 3, (2, 2), ReLU(), x)
    x = resblock_v2(x, 16, 3, False)
    x = conv2d(1, 3, (1, 1), None, x)

    if include_last_activation:
        x = tf.keras.layers.Activation("sigmoid")(x)
    return x


def get_detector_functional_model_small(x):
    x = conv2d(24, 5, (2, 2), ReLU(6), x)
    x1 = resblock_v2(x, 24, 3, False)
    x2 = sep_conv2d(48, 5, (2, 2), ReLU(6), x1)
    x2 = resblock_v2(x2, 48, 3)
    y_pedestrian = get_pedestrian_detector_functional_model_small(x1, x2)
    y_vehicle = get_vehicle_detector_functional_model_small(x1, x2)
    y = tf.keras.layers.Concatenate(-1)([y_pedestrian, y_vehicle])
    y = tf.keras.layers.Activation("sigmoid")(y)
    return y


def get_detector_functional_model_large(x):
    y_pedestrian = get_pedestrian_detector_functional_model_large(x)
    y_vehicle = get_vehicle_detector_functional_model_large(x)
    y = tf.keras.layers.Concatenate(-1)([y_pedestrian, y_vehicle])
    y = tf.keras.layers.Activation("sigmoid")(y)
    return y


class DetectorTrainer(Model):
    def __init__(
        self,
        input_shape=(None, None, 24),
        training=True,
        for_vitis=False,
        model_size="small",
        name="detector",
    ):
        super(DetectorTrainer, self).__init__(name=name)
        detector_input = tf.keras.layers.Input(input_shape)
        if model_size == "small":
            detector_output = get_detector_functional_model_small(detector_input)
        else:
            detector_output = get_detector_functional_model_large(detector_input)
        self.detector = tf.keras.Model(
            inputs=[detector_input], outputs=[detector_output], name="detector"
        )
        self.step = tf.Variable(0, dtype=tf.int64, trainable=False)

    def image_summary(self, x, y, g):
        s = y
        lidar = x
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
            # [x, y, fy, centroid_image, g],
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
            ],
            axis=2,
        )
        tf.summary.image(
            "input/pred/filtered/centroid/gt",
            image,
            max_outputs=16,
            step=self.step,
        )
        return s

    def call(self, x, training=True):
        x, g = x[..., :-2], x[..., -2:]
        y = self.detector(x, training=training)
        y = tf.cond(
            self.step % 100 == 0, lambda: self.image_summary(x, y, g), lambda: y
        )
        self.step.assign_add(1)
        return y
