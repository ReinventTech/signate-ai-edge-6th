from bev_model import DetectorTrainer
import tensorflow as tf
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Convert BEV model's checkpoint (.ckpt) to a saved model (.pb)"
    )
    parser.add_argument("--ckpt", type=str, help="The checkpoint path")
    parser.add_argument("--output", type=str, help="A path to output pb")
    args = parser.parse_args()
    detector_trainer = DetectorTrainer(training=False)

    checkpoint_path = args.ckpt
    ckpt = tf.train.Checkpoint(detector_trainer)
    ckpt.restore(checkpoint_path)

    detector = detector_trainer.detector
    inp = tf.keras.layers.Input((1152, 1152, 41))
    # outp = detector(inp, training=False)
    outp = detector(inp)
    model = tf.keras.Model(inputs=[inp], outputs=[outp], name="bev")
    # model.trainable = False
    # model.compile()

    # class BEV(tf.keras.Model):
    # def __init__(self, detector, name="bev"):
    # super(BEV, self).__init__(name=name)
    # self.detector = detector

    # @tf.function(
    # input_signature=[
    # tf.TensorSpec(shape=[None, None, None, 40 + 1], dtype=tf.float32)
    # ]
    # )
    # def call(self, x):
    # x = detector(x, training=False, summary=False)
    # return x

    # model = BEV(detector)

    tf.saved_model.save(model, args.output)


if __name__ == "__main__":
    main()
