"""It tracking camera position via E and IMU

Example run:
    1. kitti-dataset.
        python test.py \
        -d /mnt/hdd/data/data_odometry_gray/dataset/sequences/00/image_0/ \
        -f /mnt/hdd/data/data_odometry_poses/dataset/poses/00.txt
"""
import os
import time
import argparse

import cv2
import numpy as np

from visual_odometry import PinholeCamera, VisualOdometry


def get_parser():
    """It parses and returns the command line arguments.
    """
    parser = argparse.ArgumentParser(description="Visual Odometry")
    parser.add_argument("-d",
                        "--image-dir",
                        required=True,
                        help="Path to the image directory.")
    parser.add_argument("-f",
                        "--pose-file",
                        required=True,
                        help="Path to the pose file. ")
    return parser


def run(vo, image_dir):
    """It runs the updating step and plots trajectory on a cv2 canvas.
    Args:
        vo (object): VisualOdometry object.
        image_dir (str): Path to the directory which stores images.
    Return:
        None: It saves the trajectory on a image file.
    """
    traj = np.zeros((600, 600, 3), dtype=np.uint8)

    for img_id in range(4541):
        img_file = image_dir + str(img_id).zfill(6) + '.png'
        img = cv2.imread(img_file, 0)

        start_time = time.time()
        vo.update(img, img_id)
        processing_time = time.time() - start_time
        print(f"{os.path.basename(img_file)} processed in {processing_time}")

        cur_t = vo.cur_t
        if img_id > 2:
            x, y, z = cur_t[0], cur_t[1], cur_t[2]
        else:
            x, y, z = 0., 0., 0.
        draw_x, draw_y = int(x) + 290, int(z) + 90
        true_x, true_y = int(vo.trueX) + 290, int(vo.trueZ) + 90

        cv2.circle(traj, (draw_x, draw_y), 1,
                   (img_id * 255 / 4540, 255 - img_id * 255 / 4540, 0), 1)
        cv2.circle(traj, (true_x, true_y), 1, (0, 0, 255), 2)
        cv2.rectangle(traj, (10, 20), (600, 60), (0, 0, 0), -1)
        text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
        cv2.putText(traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 255), 1, 8)

        cv2.imshow('Road facing camera', img)
        # cv2.moveWindow('Trajectory', 800, 800)
        cv2.imshow('Trajectory', traj)
        cv2.waitKey(1)

    cv2.imwrite('map.png', traj)


if __name__ == "__main__":
    args = get_parser().parse_args()
    print("Command Line Args:", args)

    cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
    vo = VisualOdometry(cam, args.pose_file)

    run(vo, args.image_dir)
