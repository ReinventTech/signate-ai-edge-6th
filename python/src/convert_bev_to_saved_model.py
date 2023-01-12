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
    inp = tf.keras.layers.Input((1152, 1152, 24))
    outp = detector(inp)
    model = tf.keras.Model(inputs=[inp], outputs=[outp], name="bev")

    tf.saved_model.save(model, args.output)


if __name__ == "__main__":
    main()
