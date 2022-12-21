import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    LayerNormalization as LN,
)


class ResBlock(tf.keras.Model):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.channels = channels
        self.conv1 = tf.keras.layers.Conv2D(
            self.channels, 3, padding="same", activation="relu"
        )
        self.conv2 = tf.keras.layers.Conv2D(
            self.channels, 3, padding="same", activation="relu"
        )
        self.ln1 = LN()
        self.ln2 = LN()

    def call(self, inputs, training=True):
        x = inputs
        y = self.conv1(x)
        y = self.ln1(y, training=training)
        y = self.conv2(y)
        y = self.ln2(y, training=training)
        return x + y


class Detector(Model):
    def __init__(self, name="detector"):
        super(Detector, self).__init__(name=name)
        self.step = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.downsample1 = Sequential(
            [Conv2D(32, 7, (2, 2), padding="same", activation="relu"), LN()]
        )
        self.resblocks1 = Sequential(
            [
                ResBlock(32),
                ResBlock(32),
            ]
        )
        self.downsample2 = Sequential(
            [Conv2D(64, 5, (2, 2), padding="same", activation="relu"), LN()]
        )
        self.resblocks2 = Sequential(
            [
                ResBlock(64),
                ResBlock(64),
            ]
        )
        self.conv1 = Sequential(
            [Conv2D(64, 3, padding="same", activation="relu"), LN()]
        )

        self.downsample3 = Sequential(
            [Conv2D(128, 3, (2, 2), padding="same", activation="relu"), LN()]
        )
        self.resblocks3 = Sequential(
            [
                ResBlock(128),
                ResBlock(128),
            ]
        )
        self.downsample4 = Sequential(
            [Conv2D(256, 3, (2, 2), padding="same", activation="relu"), LN()]
        )
        self.resblocks4 = Sequential(
            [
                ResBlock(256),
                ResBlock(256),
            ]
        )
        self.conv2 = Sequential(
            [Conv2D(256, 3, padding="same", activation="relu"), LN()]
        )

        self.downsample5 = Sequential(
            [Conv2D(512, 3, (2, 2), padding="same", activation="relu"), LN()]
        )
        self.resblocks5 = Sequential(
            [
                ResBlock(512),
                ResBlock(512),
            ]
        )
        self.downsample6 = Sequential(
            [Conv2D(1024, 3, (2, 2), padding="same", activation="relu"), LN()]
        )
        self.resblocks6 = Sequential(
            [
                ResBlock(1024),
                ResBlock(1024),
            ]
        )
        self.conv3 = Sequential(
            [Conv2D(1024, 3, padding="same", activation="relu"), LN()]
        )

        self.upsample1 = Sequential(
            [
                Conv2DTranspose(512, 3, (2, 2), padding="same", activation="relu"),
                LN(),
            ]
        )
        self.resblocks7 = Sequential(
            [
                ResBlock(1024),
                ResBlock(1024),
            ]
        )
        self.upsample2 = Sequential(
            [
                Conv2DTranspose(256, 3, (2, 2), padding="same", activation="relu"),
                LN(),
            ]
        )
        self.resblocks8 = Sequential(
            [
                ResBlock(512),
                ResBlock(512),
            ]
        )
        self.conv4 = Sequential(
            [Conv2D(256, 3, padding="same", activation="relu"), LN()]
        )

        self.upsample3 = Sequential(
            [
                Conv2DTranspose(128, 3, (2, 2), padding="same", activation="relu"),
                LN(),
            ]
        )
        self.resblocks9 = Sequential(
            [
                ResBlock(256),
                ResBlock(256),
            ]
        )
        self.upsample4 = Sequential(
            [
                Conv2DTranspose(64, 3, (2, 2), padding="same", activation="relu"),
                LN(),
            ]
        )
        self.resblocks10 = Sequential(
            [
                ResBlock(128),
                ResBlock(128),
            ]
        )
        self.conv5 = Sequential(
            [Conv2D(64, 3, padding="same", activation="relu"), LN()]
        )

        self.upsample5 = Sequential(
            [
                Conv2DTranspose(32, 3, (2, 2), padding="same", activation="relu"),
                LN(),
            ]
        )
        self.resblocks11 = Sequential(
            [
                ResBlock(64),
                ResBlock(64),
            ]
        )
        self.upsample6 = Sequential(
            [
                Conv2DTranspose(16, 3, (2, 2), padding="same", activation="relu"),
                LN(),
            ]
        )
        self.resblocks12 = Sequential(
            [
                ResBlock(16),
                ResBlock(16),
            ]
        )
        self.conv6 = Conv2D(2, 3, padding="same", activation="sigmoid")

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
        for n in range(1, 31):
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
        self.step.assign_add(1)
        return s

    def call(self, x, training=True, summary=True):
        if summary:
            x, g = x[..., :-2], x[..., -2:]
        s = self.downsample1(x, training=training)
        x1 = self.resblocks1(s)
        x1 = x1 + s
        s = self.downsample2(x1, training=training)
        x2 = self.resblocks2(s)
        x2 = x2 + s
        x2 = self.conv1(x2)

        s = self.downsample3(x2, training=training)
        x3 = self.resblocks3(s)
        x3 = x3 + s
        s = self.downsample4(x3, training=training)
        x4 = self.resblocks4(s)
        x4 = x4 + s
        x4 = self.conv2(x4)

        s = self.downsample5(x4, training=training)
        x5 = self.resblocks5(s)
        x5 = x5 + s
        s = self.downsample6(x5, training=training)
        x6 = self.resblocks6(s)
        x6 = x6 + s
        x6 = self.conv3(x6)

        s = self.upsample1(x6, training=training)
        s = tf.concat([s, x5], -1)
        y = self.resblocks7(s)
        y1 = y + s
        s = self.upsample2(y1, training=training)
        s = tf.concat([s, x4], -1)
        y = self.resblocks8(s)
        y = y + s
        y2 = self.conv4(y)

        s = self.upsample3(y2, training=training)
        s = tf.concat([s, x3], -1)
        y = self.resblocks9(s)
        y3 = y + s
        s = self.upsample4(y3, training=training)
        s = tf.concat([s, x2], -1)
        y = self.resblocks10(s)
        y = y + s
        y4 = self.conv5(y)

        s = self.upsample5(y4, training=training)
        s = tf.concat([s, x1], -1)
        y = self.resblocks11(s)
        y5 = y + s
        s = self.upsample6(y5, training=training)
        y = self.resblocks12(s)
        y = y + s
        y = self.conv6(y)

        if summary:
            y = self.image_summary(x, y, g)
        return y
