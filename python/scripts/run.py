import sys
import json
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exec-path", help="/path/to/src")
    parser.add_argument("--test-meta-path", help="/path/to/meta_data.json")
    parser.add_argument("--test-data-dir", help="/path/to/train/3d_labels")
    parser.add_argument("--result-path", default="./run_result.json")

    args = parser.parse_args()

    return args


def main():
    # parse the arguments
    args = parse_args()
    exec_path = os.path.abspath(args.exec_path)
    test_meta_path = os.path.abspath(args.test_meta_path)
    result_path = os.path.abspath(args.result_path)
    test_data_dir = os.path.abspath(args.test_data_dir)

    # load the meta data
    with open(test_meta_path) as f:
        test_meta = json.load(f)

    # change the working directory
    os.chdir(exec_path)
    cwd = os.getcwd()
    print("\nMoved to {}".format(cwd))
    model_path = os.path.join("..", "models")

    # load the model
    sys.path.append(cwd)
    from predictor import ScoringService

    print("\nLoading the model...", end="\r")
    model_flag = ScoringService.get_model(model_path)
    if model_flag:
        print("Loaded the model.   ")
    else:
        print("Could not load the model.")
        return None

    # run all and save the result
    result = {}
    # count = 0
    for scene_id, frames in test_meta.items():
        print(scene_id)
        for frame in frames:
            frame["cam_path"] = os.path.join(test_data_dir, frame["cam_path"])
            frame["lidar_path"] = os.path.join(test_data_dir, frame["lidar_path"])
            output = ScoringService.predict(frame)
            result.update(output)
            # count += 1
            # if count > 1:
                # break

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
