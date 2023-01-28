import sys
import json
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-meta-path", help="/path/to/meta_data.json")
    parser.add_argument("--test-data-dir", help="/path/to/train/3d_labels")

    args = parser.parse_args()

    return args


def main():
    # parse the arguments
    args = parse_args()
    # exec_path = os.path.abspath(args.exec_path)
    test_meta_path = os.path.abspath(args.test_meta_path)
    test_data_dir = os.path.abspath(args.test_data_dir)

    # load the meta data
    with open(test_meta_path) as f:
        test_meta = json.load(f)

    # change the working directory
    # os.chdir(exec_path)
    # cwd = os.getcwd()
    # print("\nMoved to {}".format(cwd), file=sys.stderr)

    # load the model
    # sys.path.append(cwd)

    # run all and save the result
    for scene_id, frames in test_meta.items():
        print(scene_id, file=sys.stderr)
        for frame in frames:
            frame["cam_path"] = os.path.join(test_data_dir, frame["cam_path"])
            frame["lidar_path"] = os.path.join(test_data_dir, frame["lidar_path"])
            print(
                frame["test_key"],
                frame["lidar_path"],
                frame["cam_ego_pose"]["translation"][0],
                frame["cam_ego_pose"]["translation"][1],
                frame["cam_ego_pose"]["translation"][2],
                frame["cam_ego_pose"]["rotation"][0],
                frame["cam_ego_pose"]["rotation"][1],
                frame["cam_ego_pose"]["rotation"][2],
                frame["cam_ego_pose"]["rotation"][3],
            )


if __name__ == "__main__":
    main()
