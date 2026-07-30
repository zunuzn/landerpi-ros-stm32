#!/usr/bin/env python3
import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from ros_robot_controller_msgs.msg import WheelEncoderState


class EncoderOdomPublisher(Node):
    def __init__(self):
        super().__init__('encoder_odom_publisher')

        self.declare_parameter('wheel_diameter', 0.065)
        self.declare_parameter('wheelbase', 0.1368)
        self.declare_parameter('track_width', 0.1446)
        self.declare_parameter('ticks_per_revolution', 1040)
        self.declare_parameter('wheel_signs', [-1, -1, 1, 1])
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('base_frame_id', 'base_footprint')
        self.declare_parameter('max_dt', 0.5)
        self.declare_parameter('max_wheel_delta', 0.5)

        self.wheel_diameter = float(self.get_parameter('wheel_diameter').value)
        self.wheelbase = float(self.get_parameter('wheelbase').value)
        self.track_width = float(self.get_parameter('track_width').value)
        self.ticks_per_revolution = int(self.get_parameter('ticks_per_revolution').value)
        self.wheel_signs = list(self.get_parameter('wheel_signs').value)
        self.odom_frame_id = self.get_parameter('odom_frame_id').value
        self.base_frame_id = self.get_parameter('base_frame_id').value
        self.max_dt = float(self.get_parameter('max_dt').value)
        self.max_wheel_delta = float(self.get_parameter('max_wheel_delta').value)

        if len(self.wheel_signs) != 4:
            raise ValueError('wheel_signs must contain exactly four values')
        if self.ticks_per_revolution <= 0 or self.wheel_diameter <= 0.0:
            raise ValueError('wheel dimensions and ticks_per_revolution must be positive')
        if self.wheelbase <= 0.0 or self.track_width <= 0.0:
            raise ValueError('wheelbase and track_width must be positive')
        if self.max_dt <= 0.0 or self.max_wheel_delta <= 0.0:
            raise ValueError('max_dt and max_wheel_delta must be positive')

        self.circumference = math.pi * self.wheel_diameter
        self.rotation_radius = (self.wheelbase + self.track_width) / 2.0
        self.last_ticks = None
        self.last_stamp = None
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.initialized = False

        self.publisher = self.create_publisher(Odometry, '/encoder_odom', 10)
        self.create_subscription(WheelEncoderState, '/wheel_encoder/state', self.callback, 20)
        self.get_logger().info(
            'waiting for first /wheel_encoder/state sample to initialize /encoder_odom')

    def mecanum_twist(self, wheel_values):
        v1, v2, v3, v4 = wheel_values
        return (
            (v1 + v2 + v3 + v4) / 4.0,
            (-v1 + v2 + v3 - v4) / 4.0,
            (-v1 - v2 + v3 + v4) / (4.0 * self.rotation_radius),
        )

    def callback(self, msg):
        stamp = Time.from_msg(msg.header.stamp)
        ticks = list(msg.ticks)

        if len(ticks) != 4 or len(msg.rps) != 4:
            self.get_logger().warn('invalid wheel encoder sample length; expected 4 ticks and 4 rps')
            return

        if self.last_ticks is None:
            self.last_ticks = ticks
            self.last_stamp = stamp
            self.initialized = True
            self.get_logger().info(
                '/encoder_odom initialized at x=0.0 y=0.0 yaw=0.0; '
                'first encoder sample used as baseline')
            self.publish_odom(msg.header.stamp, 0.0, 0.0, 0.0)
            return

        dt = (stamp - self.last_stamp).nanoseconds / 1e9
        if dt <= 0.0:
            self.get_logger().warn('dropped wheel encoder sample with non-increasing timestamp')
            return
        if dt > self.max_dt:
            self.get_logger().warn(
                'dropped wheel encoder sample after large dt %.3fs; resetting encoder baseline' % dt)
            self.last_ticks = ticks
            self.last_stamp = stamp
            self.publish_odom(msg.header.stamp, 0.0, 0.0, 0.0)
            return

        wheel_distances = [
            (ticks[index] - self.last_ticks[index])
            * self.wheel_signs[index]
            * self.circumference
            / self.ticks_per_revolution
            for index in range(4)
        ]
        max_delta = max(abs(distance) for distance in wheel_distances)
        if max_delta > self.max_wheel_delta:
            self.get_logger().warn(
                'dropped wheel encoder sample with large wheel delta %.3fm; '
                'resetting encoder baseline' % max_delta)
            self.last_ticks = ticks
            self.last_stamp = stamp
            self.publish_odom(msg.header.stamp, 0.0, 0.0, 0.0)
            return

        delta_x, delta_y, delta_yaw = self.mecanum_twist(wheel_distances)
        yaw_midpoint = self.yaw + delta_yaw / 2.0

        self.x += delta_x * math.cos(yaw_midpoint) - delta_y * math.sin(yaw_midpoint)
        self.y += delta_x * math.sin(yaw_midpoint) + delta_y * math.cos(yaw_midpoint)
        self.yaw += delta_yaw

        wheel_speeds = [
            msg.rps[index] * self.wheel_signs[index] * self.circumference
            for index in range(4)
        ]
        vx, vy, wz = self.mecanum_twist(wheel_speeds)
        self.publish_odom(msg.header.stamp, vx, vy, wz)

        self.last_ticks = ticks
        self.last_stamp = stamp

    def publish_odom(self, stamp, vx, vy, wz):
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame_id
        odom.child_frame_id = self.base_frame_id
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.z = math.sin(self.yaw / 2.0)
        odom.pose.pose.orientation.w = math.cos(self.yaw / 2.0)
        odom.pose.covariance[0] = 0.02
        odom.pose.covariance[7] = 0.02
        odom.pose.covariance[14] = 1e6
        odom.pose.covariance[21] = 1e6
        odom.pose.covariance[28] = 1e6
        odom.pose.covariance[35] = 0.05
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.angular.z = wz
        odom.twist.covariance[0] = 0.04
        odom.twist.covariance[7] = 0.04
        odom.twist.covariance[14] = 1e6
        odom.twist.covariance[21] = 1e6
        odom.twist.covariance[28] = 1e6
        odom.twist.covariance[35] = 0.1
        self.publisher.publish(odom)


def main():
    rclpy.init()
    node = EncoderOdomPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
