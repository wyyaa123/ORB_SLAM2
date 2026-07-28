# -*- encoding: utf-8 -*-
'''
@File    :   img_extract.py
@Time    :   2023/09/01 11:32:18
@Author  :   orCate 
@Version :   1.0
@Contact :   8631143542@qq.com
'''

import os

import cv2 as cv
import numpy as np
import rosbag
import rospy
import tqdm
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu


class parseBag():
    def __init__(self, bag_file, save_path):
        self.bridge = CvBridge()
        self.save_path = save_path
        self.imu_files = {}
        self.image_dirs = {}
        self.imu_count = 0
        self.image_count = 0
        self.processed_count = 0

        with rosbag.Bag(bag_file, 'r') as bag:   # 要读取的bag文件
            topic_type_map = self._get_topic_type_map(bag)
            bar = tqdm.tqdm(bag.read_messages(), total=bag.get_message_count(), desc="parsing")
            for topic, msg, t in bar:
                msg_type = topic_type_map.get(topic, '')
                self.processed_count += 1

                if msg_type == 'sensor_msgs/Imu':
                    self._save_imu(topic, msg)
                    self.imu_count += 1
                elif msg_type == 'sensor_msgs/Image' or msg_type == 'sensor_msgs/CompressedImage':
                    if self._save_image(topic, msg, msg_type):
                        self.image_count += 1
                    else:
                        bar.set_description_str('failed!')

                if self.processed_count % 500 == 0:
                    bar.set_postfix(imu=self.imu_count, image=self.image_count)

            bar.set_postfix(imu=self.imu_count, image=self.image_count)

        for imu_file in self.imu_files.values():
            imu_file.close()

    def _topic_to_name(self, topic):
        return topic.lstrip('/').replace('/', '_')

    def _get_topic_type_map(self, bag):
        topic_info = bag.get_type_and_topic_info()
        return {topic: info.msg_type for topic, info in topic_info.topics.items()}

    def _get_timestamp_ns(self, msg):
        return msg.header.stamp.to_nsec()

    def _get_imu_file(self, topic):
        if topic not in self.imu_files:
            # imu_file_path = os.path.join(self.save_path, self._topic_to_name(topic) + '.txt')
            imu_file_path = os.path.join(self.save_path, self._topic_to_name(topic) + '.csv')
            os.makedirs(os.path.dirname(imu_file_path), exist_ok=True)
            self.imu_files[topic] = open(imu_file_path, 'w', encoding='utf-8')
        return self.imu_files[topic]

    def _get_image_dir(self, topic):
        if topic not in self.image_dirs:
            topic_dir = os.path.join(self.save_path, self._topic_to_name(topic))
            os.makedirs(topic_dir, exist_ok=True)
            self.image_dirs[topic] = topic_dir
        return self.image_dirs[topic]

    def _save_imu(self, topic, msg):
        imu_msg: Imu = msg
        timestamp = self._get_timestamp_ns(imu_msg)

        gyro_x = imu_msg.angular_velocity.x
        gyro_y = imu_msg.angular_velocity.y
        gyro_z = imu_msg.angular_velocity.z
        acc_x = imu_msg.linear_acceleration.x
        acc_y = imu_msg.linear_acceleration.y
        acc_z = imu_msg.linear_acceleration.z

        imu_data = [timestamp, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z]
        imu_data = ''.join(str(data) + ',' for data in imu_data) + '\n'
        self._get_imu_file(topic).write(imu_data)

    def _save_image(self, topic, msg, msg_type):
        image_dir = self._get_image_dir(topic)
        timestamp = self._get_timestamp_ns(msg)
        image_path = os.path.join(image_dir, str(timestamp) + '.png')

        if msg_type == 'sensor_msgs/Image':
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            except CvBridgeError as e: 
                print(e)
                return False
            if msg.encoding == 'rgb8':
                cv_image = cv.cvtColor(cv_image, cv.COLOR_RGB2BGR)
            elif msg.encoding == 'rgba8':
                cv_image = cv.cvtColor(cv_image, cv.COLOR_RGBA2BGR)
        elif msg_type == 'sensor_msgs/CompressedImage':
            try:
                np_buffer = np.frombuffer(msg.data, dtype=np.uint8)
                cv_image = cv.imdecode(np_buffer, cv.IMREAD_UNCHANGED)
            except Exception as e:
                print(e)
                return False
        else:
            return False

        return cv.imwrite(image_path, cv_image)


if __name__ == '__main__':
    try:
        parse = parseBag("datasets/Real/D435I/wallpainting_2026-07-27-14-57-50.bag", "datasets/Real/D435I/wallpainting")
    except rospy.ROSInterruptException:
        pass