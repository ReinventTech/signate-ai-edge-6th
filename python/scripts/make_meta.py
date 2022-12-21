import os
import json
import argparse
import numpy as np
from nuscenes.nuscenes import NuScenes


def make_test(nusc, test_scene_names, category_names):
    meta_data = {}
    annotations = {}
    nbr_samples = 0
    scene_ids = {
        test_scene_name: "{:02d}".format(i)
        for i, test_scene_name in enumerate(test_scene_names)
    }
    for scene in nusc.scene:
        if scene["name"] in test_scene_names:
            print(scene["name"], scene["nbr_samples"])
            nbr_samples += scene["nbr_samples"]
            sample_token = scene["first_sample_token"]
            scene_id = scene_ids[scene["name"]]
            meta_data[scene_id] = []
            frame_id = 0
            while sample_token != "":
                sample = nusc.get("sample", sample_token)
                time_stamp = sample["timestamp"]
                scene_token = sample["scene_token"]
                frame_name = "{:03d}".format(frame_id)
                test_key = "{}_{}".format(scene_id, frame_name)

                # sensor data
                lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
                cam = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
                lidar_path = lidar["filename"]
                cam_path = cam["filename"]
                cam_ego_pose = nusc.get(
                    "ego_pose", cam["ego_pose_token"]
                )  # rotation, translation
                cam_ego_pose_f = {
                    "translation": cam_ego_pose["translation"],
                    "rotation": cam_ego_pose["rotation"],
                }
                lidar_ego_pose = nusc.get(
                    "ego_pose", lidar["ego_pose_token"]
                )  # rotation, translation
                lidar_ego_pose_f = {
                    "translation": lidar_ego_pose["translation"],
                    "rotation": lidar_ego_pose["rotation"],
                }
                cam_calibration = nusc.get(
                    "calibrated_sensor", cam["calibrated_sensor_token"]
                )
                cam_calibration_f = {
                    "translation": cam_calibration["translation"],
                    "rotation": cam_calibration["rotation"],
                    "camera_intrinsic": cam_calibration["camera_intrinsic"],
                }
                lidar_calibration = nusc.get(
                    "calibrated_sensor", lidar["calibrated_sensor_token"]
                )
                lidar_calibration_f = {
                    "translation": lidar_calibration["translation"],
                    "rotation": lidar_calibration["rotation"],
                    "camera_intrinsic": lidar_calibration["camera_intrinsic"],
                }

                meta_data[scene_id].append(
                    {
                        "test_key": test_key,
                        "cam_path": cam_path,
                        "lidar_path": lidar_path,
                        "time_stamp": time_stamp,
                        "cam_ego_pose": cam_ego_pose_f,
                        "lidar_ego_pose": lidar_ego_pose_f,
                        "cam_calibration": cam_calibration_f,
                        "lidar_calibration": lidar_calibration_f,
                    }
                )

                # annoation
                annotation = {}
                for ann in sample["anns"]:
                    annotation_data = nusc.get("sample_annotation", ann)
                    category_name = annotation_data["category_name"]
                    new_name = None
                    for c, o in category_names.items():
                        if category_name in o:
                            new_name = c
                    if new_name is not None:
                        flag = True
                        translation = annotation_data["translation"]
                        dist = np.linalg.norm(
                            np.array(translation[:2])
                            - np.array(lidar_ego_pose["translation"][:2])
                        )
                        num_lidar_pts = annotation_data["num_lidar_pts"]
                        if new_name == "pedestrian":
                            if dist > 40:
                                flag = False
                        else:
                            if dist > 50:
                                flag = False
                        if num_lidar_pts == 0:
                            flag = False
                        if flag:
                            if new_name not in annotation:
                                annotation[new_name] = []
                            annotation[new_name].append(
                                [translation[0], translation[1], 1]
                            )
                annotations[test_key] = annotation

                # update
                sample_token = sample["next"]
                frame_id += 1

    print("total number of samples:", nbr_samples)
    return meta_data, annotations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", help="/path/to/train/3d_labels")
    parser.add_argument("--output-path", default=".")

    args = parser.parse_args()

    return args


def main():
    # parse the arguments
    args = parse_args()
    data_dir = args.data_dir
    meta_data_path = os.path.join(args.output_path, "meta_data.json")
    annotations_path = os.path.join(args.output_path, "ans.json")

    # set scene names for validation
    test_scene_names = [
        # "scene-0001",  # 2, 12 x3
        # # "scene-0043", #
        # "scene-0004",  # 4, 20 x6
        # "scene-0005",  # 2, 39 x13
        # "scene-0008",  # 5, 20 x3
        # # "scene-0007" #
        # "scene-0009",  # 10, 79 x15
        # # "scene-0010" #
        # "scene-0011",
        # "scene-0020",  # 14, 123 x29  # 171, 129 x27
        # "scene-0021",  # 3, 41 x8
        # # "scene-0022" #
        # "scene-0024",
        # "scene-0025",  # 2, 8 x7  # 5, 31 x22
        # # "scene-0026"  # 3, 4 x63
        # "scene-0027",  # 12, 15 x7
        # "scene-0028" #
        # "scene-0029"  #
        # "scene-0030"  #
        # "scene-0031"  # 2, 15 x16
        # "scene-0032"  #
        # "scene-0033"  #
        # "scene-0034"  # 3, 8 x15
        # "scene-0041" #
        # "scene-0042" #
        # "scene-0045"  #
        # "scene-0048"  # 1043, 600 x153
        # "scene-0049" # skip
        # "scene-0050" # skip
        # "scene-0051"  #
        # "scene-0054" #
        # "scene-0058"  # skip
        "scene-0109"  # 1004, 3633 x183
    ]  # list of scenes for validation

    # set test categories
    category_names = {
        "pedestrian": [
            "human.pedestrian.construction_worker",
            "human.pedestrian.adult",
        ],
        "vehicle": ["vehicle.car"],
    }

    # make meta data and ground truth
    nusc = NuScenes(version="v1.0-trainval", dataroot=data_dir, verbose=True)
    meta_data, annotations = make_test(nusc, test_scene_names, category_names)
    with open(meta_data_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f)
    with open(annotations_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f)


if __name__ == "__main__":
    main()
