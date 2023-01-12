from bev_model import DetectorTrainer, get_pedestrian_detector_layers
import tensorflow as tf
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Merge two BEV models' checkpoints for pedestrian/vehicle (.ckpt) and convert it to a saved model (.pb)"
    )
    parser.add_argument("--pckpt", type=str, help="The best pedestrian checkpoint path")
    parser.add_argument("--vckpt", type=str, help="The best vehicle checkpoint path")
    parser.add_argument("--output", type=str, help="A path to output pb")
    parser.add_argument(
        "--model-size", type=str, default="small", help="Model size (small or large)"
    )
    args = parser.parse_args()

    detector_trainer = DetectorTrainer(training=False)
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
    inp = tf.keras.layers.Input((1152, 1152, 24))
    outp = detector(inp)
    model = tf.keras.Model(inputs=[inp], outputs=[outp], name="bev")

    tf.saved_model.save(model, args.output)


if __name__ == "__main__":
    main()
