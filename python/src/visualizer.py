import cv2
import quaternion
import numpy as np


def visualize_predictions(
    pedestrian_preds, vehicle_preds, summary_image, ego_pose, ans
):

    ego_txyz = np.float32(ego_pose["translation"])
    ego_qt = quaternion.as_quat_array(ego_pose["rotation"])

    # pred_image = np.zeros((1152, 1152, 3), np.uint8)
    pred_image = summary_image[:, :1152].copy()
    gt_image = summary_image[:, :1152].copy()
    for n, p in enumerate(ans["pedestrian"]):
        xyz = np.array([p[0], p[1], 0])
        xyz = xyz - ego_txyz
        xyz = quaternion.rotate_vectors(ego_qt.inverse(), xyz)
        xyz[0] = xyz[0] * 10 + 512 + 64
        xyz[1] = -xyz[1] * 10 + 512 + 64
        xyz = np.clip(np.round(xyz), 0, 1152).astype(int)
        gt_image = cv2.circle(
            gt_image, (xyz[0], xyz[1]), 1, (255, 0, 0), 3, cv2.LINE_AA
        )
        pred_image = cv2.circle(
            pred_image, (xyz[0], xyz[1]), 7, (0, 255, 255), 2, cv2.LINE_AA
        )
    for n, p in enumerate(ans["vehicle"]):
        xyz = np.array([p[0], p[1], 0])
        xyz = xyz - ego_txyz
        xyz = quaternion.rotate_vectors(ego_qt.inverse(), xyz)
        xyz[0] = xyz[0] * 10 + 512 + 64
        xyz[1] = -xyz[1] * 10 + 512 + 64
        xyz = np.clip(np.round(xyz), 0, 1152).astype(int)
        gt_image = cv2.circle(
            gt_image, (xyz[0], xyz[1]), 1, (0, 255, 0), 3, cv2.LINE_AA
        )
        pred_image = cv2.circle(
            pred_image, (xyz[0], xyz[1]), 7, (255, 0, 255), 2, cv2.LINE_AA
        )
    for n, p in enumerate(pedestrian_preds):
        color = (
            (255, 0, 0) if p[2] >= 0.5 else (144, 0, 0) if p[2] >= 0.2 else (96, 0, 0)
        )
        thickness = 4 if p[2] >= 0.5 else 3 if p[2] >= 0.2 else 2
        xyz = np.array([p[0], p[1], 0])
        xyz = xyz - ego_txyz
        xyz = quaternion.rotate_vectors(ego_qt.inverse(), xyz)
        xyz[0] = xyz[0] * 10 + 512 + 64
        xyz[1] = -xyz[1] * 10 + 512 + 64
        xyz = np.clip(np.round(xyz), 0, 1152).astype(int)
        pred_image = cv2.circle(
            pred_image, (xyz[0], xyz[1]), 1, color, thickness, cv2.LINE_AA
        )
        pred_image = cv2.putText(
            pred_image,
            f"{n}:{int(p[2]*100)}",
            (xyz[0] + 3, xyz[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            color,
            1,
            cv2.LINE_AA,
        )
    for n, p in enumerate(vehicle_preds):
        color = (
            (0, 255, 0) if p[2] >= 0.5 else (0, 144, 0) if p[2] >= 0.2 else (0, 96, 0)
        )
        thickness = 4 if p[2] >= 0.5 else 3 if p[2] >= 0.2 else 2
        xyz = np.array([p[0], p[1], 0])
        xyz = xyz - ego_txyz
        xyz = quaternion.rotate_vectors(ego_qt.inverse(), xyz)
        xyz[0] = xyz[0] * 10 + 512 + 64
        xyz[1] = -xyz[1] * 10 + 512 + 64
        xyz = np.clip(np.round(xyz), 0, 1152).astype(int)
        pred_image = cv2.circle(
            pred_image, (xyz[0], xyz[1]), 1, color, thickness, cv2.LINE_AA
        )
        pred_image = cv2.putText(
            pred_image,
            f"{n}:{int(p[2]*100)}",
            (xyz[0] + 3, xyz[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            color,
            1,
            cv2.LINE_AA,
        )

    # summary_image = np.concatenate([summary_image, pred_image, gt_image], 1)
    summary_image = np.concatenate([summary_image[:, 1152:], pred_image], 1)

    summary_image = cv2.cvtColor(summary_image, cv2.COLOR_BGR2RGB)
    return summary_image
