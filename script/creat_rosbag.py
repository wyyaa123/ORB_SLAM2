import os
import numpy as np

import rosbag 
import rospy
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Vector3
from cv_bridge import CvBridge
import cv2 as cv
from tqdm import tqdm

def ReadImage(file_dir):
    "Here assume the name of image is the timestamp"
    file_names_left = sorted(os.listdir(file_dir))
    def get_timestamp(file_name):
        return np.float64(file_name[:-4])
    timestamps = map(get_timestamp, file_names_left)
    print("Total add %i images!"%(np.size(file_names_left)))
    return file_names_left, list(timestamps)

def ReadIMU(file_path):
    '''return IMU data and timestamp of IMU'''
    file = open(file_path, 'r')
    all = file.readlines()
    timestamp = []
    imu_data = []
    index = 0
    for f in all:
        index = index + 1
        line = f.rstrip('\n').split(' ') # here maybe ',' or ' '
        timestamp.append(line[0])
        imu_data.append(line[1:7])
    print("Total add %i imus!"%(index))
    return imu_data, list(timestamp)

def CreateBag():
    left_image_dir = "datasets/Real/D435I/2026-04-24-08-20-35/camera_infra1_image_rect_raw/"
    left_image_sem_dir = "datasets/Real/D435I/2026-04-24-08-20-35/camera_infra1_semantic/merged/"
    right_image_dir = "datasets/Real/D435I/2026-04-24-08-20-35/camera_infra2_image_rect_raw/"
    right_image_sem_dir = "datasets/Real/D435I/2026-04-24-08-20-35/camera_infra2_semantic/merged/"
    imu_path = 'datasets/Real/D435I/2026-04-24-08-20-35/camera_imu.txt'
    depth_image_dir = "datasets/Real/D435I/2026-04-24-08-20-35/depth/"

    bag = rosbag.Bag("data.bag", 'w')
    file_names_left, imgstamp_left = ReadImage(left_image_dir)
    file_names_left_sem, imgstamp_left_sem = ReadImage(left_image_sem_dir)
    file_names_right, imgstamp_right = ReadImage(right_image_dir)
    file_names_right_sem, imgstamp_right_sem = ReadImage(right_image_sem_dir)
    file_names_depth, imgstamp_depth = ReadImage(depth_image_dir)
    # print(file_names_left)
    # print(timestamp)
    imu_data, imustamp = ReadIMU(imu_path)
    # print(imu_data)
    # print(timestamp)
    print('working!')

    for i in tqdm(range(len(file_names_left)), desc="infra1 image"):
        img = Image()
        img = CvBridge().cv2_to_imgmsg(cv.imread(left_image_dir + file_names_left[i], cv.IMREAD_GRAYSCALE))
        img.header.frame_id = "camera"
        img.header.stamp = rospy.Time(imgstamp_left[i] * 1e-9)
        img.encoding = "8UC1"
        bag.write("/camera/infra1/image_rect_raw", img, img.header.stamp)

    for i in tqdm(range(len(file_names_left_sem)), desc="infra1 semantic"):
        img = Image()
        img = CvBridge().cv2_to_imgmsg(cv.imread(left_image_sem_dir + file_names_left_sem[i], cv.IMREAD_GRAYSCALE))
        img.header.frame_id = "camera"
        img.header.stamp = rospy.Time(imgstamp_left_sem[i] * 1e-9)
        img.encoding = "8UC1"
        bag.write("/camera/infra1/semantic", img, img.header.stamp)

    for i in tqdm(range(len(file_names_right)), desc="infra2 image"):
        img = Image()
        img = CvBridge().cv2_to_imgmsg(cv.imread(right_image_dir + file_names_right[i], cv.IMREAD_GRAYSCALE))
        img.header.frame_id = "camera"
        img.header.stamp = rospy.Time(imgstamp_right[i] * 1e-9)
        img.encoding = "8UC1"
        bag.write("/camera/infra2/image_rect_raw", img, img.header.stamp)

    for i in tqdm(range(len(file_names_right_sem)), desc="infra2 semantic"):
        img = Image()
        img = CvBridge().cv2_to_imgmsg(cv.imread(right_image_sem_dir + file_names_right_sem[i], cv.IMREAD_GRAYSCALE))
        img.header.frame_id = "camera"
        img.header.stamp = rospy.Time(imgstamp_right_sem[i] * 1e-9)
        img.encoding = "8UC1"
        bag.write("/camera/infra2/semantic", img, img.header.stamp)

    for i in tqdm(range(len(file_names_depth)), desc="depth image"):
        img = Image()
        img = CvBridge().cv2_to_imgmsg(cv.imread(depth_image_dir + file_names_depth[i], cv.IMREAD_UNCHANGED))
        img.header.frame_id = "camera"
        img.header.stamp = rospy.Time(imgstamp_depth[i] * 1e-9)
        img.encoding = "16UC1"
        bag.write("/camera/depth/image_rect_raw", img, img.header.stamp)

    for i in tqdm(range(1, len(imu_data)), desc="imu"):
        # print(i)
        imu = Imu()
        angular_v = Vector3()
        linear_a = Vector3()
        angular_v.x = float(imu_data[i][0])
        angular_v.y = float(imu_data[i][1])
        angular_v.z = float(imu_data[i][2])
        linear_a.x = float(imu_data[i][3])
        linear_a.y = float(imu_data[i][4])
        linear_a.z = float(imu_data[i][5])
        imuStamp = rospy.rostime.Time.from_sec(float(imustamp[i]) * 1e-9)  # according to the timestamp unit
        imu.header.stamp=imuStamp
        imu.angular_velocity = angular_v
        imu.linear_acceleration = linear_a

        bag.write("/camera/imu",imu,imuStamp)

    bag.close()
    print('Done!')


if __name__ == "__main__":
    # bag = rosbag.Bag("", 'w')
    # bag.write("/camera/imu", Image(), )
    CreateBag()