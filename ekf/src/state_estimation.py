#!/usr/bin/env python

import rospy
import numpy as np
import random as rnd
import tf
from math import atan2, sqrt, pow
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PoseStamped, Quaternion
from nav_msgs.msg import Odometry, OccupancyGrid
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion, quaternion_from_euler
# NOTE: using simplified Pose msg from turtlesim.msg instead of geometry_msg.msg.Pose

M_PI=3.14159265358979323846

class StateEstimator:

    def __init__(self):
        rospy.init_node('turtlebot_state_estimator')
        self.rate = rospy.Rate(10)

        # State
        self.pose = PoseWithCovarianceStamped()
        self.state = np.zeros((3, 1))
        self.state_prior = np.zeros_like(self.state)
        # self.vel = Twist() # current control input: [v_x, v_y, w]
        self.u = np.zeros((2, 1))
        self.meas =  np.zeros_like(self.state) # latest measurement from IPS sensor/odom
        self.delta_t = 0.0
        self.delta_r = 0.0
        self.cov_prior = np.eye(3)
        self.cov = np.eye(3) # covariance of true belief
        self.cov_delta = np.eye(3) # covariance of measurement noise (Q)
        self.cov_eps = np.eye(3) # covariance of process noise (R)
        self.ips_prev_stamp = None
        self.odom_prev_stamp = None
        self.delta = 0.0
        self.noise_t = 0.5
        self.noise_r = 0.1
        # TODO: read simulation flag from rosparam
        self.sim = False

        # Input
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.pose_pub = rospy.Publisher('/pose', PoseStamped, queue_size=1)
        self.indoor_pub = rospy.Publisher('/inpose', PoseWithCovarianceStamped, queue_size=10)
        self.pose_ekf_pub = rospy.Publisher('/pose_ekf', PoseWithCovarianceStamped, queue_size=10)
        if self.sim:
            rospy.Subscriber("/gazebo/model_states", ModelStates, self.pose_callback)
        else:
            rospy.Subscriber("/indoor_pos", PoseWithCovarianceStamped, self.indoor_pose_callback, queue_size=10)
        # rospy.Subscriber("/cmd_vel_mux/input/teleop", Twist, self.cmd_vel_callback) 

        # Outputs
        # self.vel_pub = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist, queue_size=10)
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback, ())
    
    # cmd_vel callback
    # Read control inputs from teleop node ???
    def cmd_vel_callback(self, data):
        self.u[0] = np.sqrt(data.linear.x*data.linear.x + data.linear.y*data.linear.y)
        self.u[1] = data.angular.z

    # Odom callback
    # TODO: when using the pose message you most likely want to take relative transforms not the total integrated pose
    def odom_callback(self, data):
        # Stamp update
        # if self.odom_prev_stamp is None:
        #     self.odom_prev_stamp = data.header.stamp.secs
        if (self.meas == False).all():
            self.meas[0] = data.pose.pose.position.x
            self.meas[1] = data.pose.pose.position.y
            (roll, pitch, yaw) = euler_from_quaternion([data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])
            self.meas[2] = yaw
        else:
            x = data.pose.pose.position.x + np.random.normal(0, self.noise_t*self.noise_t)
            y = data.pose.pose.position.y + np.random.normal(0, self.noise_t*self.noise_t)
            self.delta_t = np.sqrt((x - self.meas[0])**2 + (y - self.meas[1])**2)# + np.random.normal(0, self.noise_t*self.noise_t)
            self.delta_r = np.arctan2((y - self.meas[1]), (x - self.meas[0]))# + np.random.normal(0, self.noise_r*self.noise_r)
            self.meas[0] = x
            self.meas[1] = y
            (roll, pitch, yaw) = euler_from_quaternion([data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])
            self.meas[2]= yaw
            # TRY: using velocity from odom Twist message
            self.u[0] = np.sqrt(data.twist.twist.linear.x**2 + data.twist.twist.linear.y**2)
            self.u[1] = data.twist.twist.angular.z
        # Assign static covariance values
        # self.cov_delta = np.diag(data.pose.covariance[:3])
        self.cov_delta = np.diag([0.1, 0.1, 0.1])
        # self.cov_delta = np.array(data.pose.covariance)
        # self.cov_delta = np.reshape(self.cov_eps, (3, 3))

        # self.delta = data.header.stamp.secs - self.odom_prev_stamp
        # self.odom_prev_stamp = data.header.stamp.secs

    # IPS callback
    # NOTE: pose of all objects in gazebo world is being published to model_states
    #       e.g. jackal corresponds to /gazebo/model_states/pose[17]
    def pose_callback(self, data):
        # No covariance for IPS in sim, so update ???
        # Stamp update
        # stamp = rospy.Time.now().to_sec()
        # if self.ips_prev_stamp is None:
            # self.ips_prev_stamp = data.header.stamp
            # self.ips_prev_stamp = stamp
        if (self.meas == False).all():
            self.meas[0] = data.pose[9].position.x
            self.meas[1] = data.pose[9].position.y
            (roll, pitch, yaw) = euler_from_quaternion([data.pose[9].orientation.x, data.pose[9].orientation.y, data.pose[9].orientation.z, data.pose[9].orientation.w])
            self.meas[2] = yaw
        else:
            x = data.pose[9].position.x + np.random.normal(0, self.noise_t*self.noise_t)
            y = data.pose[9].position.y + np.random.normal(0, self.noise_t*self.noise_t)
            self.delta_t = np.sqrt((x - self.meas[0])**2 + (y - self.meas[1])**2)# + np.random.normal(0, self.noise_t*self.noise_t)
            self.delta_r = np.arctan2((y - self.meas[1]), (x - self.meas[0]))# + np.random.normal(0, self.noise_r*self.noise_r)
            self.meas[0] = x
            self.meas[1] = y
            (roll, pitch, yaw) = euler_from_quaternion([data.pose[9].orientation.x, data.pose[9].orientation.y, data.pose[9].orientation.z, data.pose[9].orientation.w])
            self.meas[2] = yaw
        # Assign static covariance values   
        # self.cov_delta = np.diag(data.pose.covariance[:3])
        self.cov_delta = np.diag([0.1, 0.1, 0.1])
        # self.delta = data.header.stamp - self.ips_prev_stamp
        # self.ips_prev_stamp = data.header.stamp
        # self.delta = stamp - self.ips_prev_stamp
        # self.ips_prev_stamp = stamp

        # Republish IPS pose to map frame as GROUND TRUTH
        curpose = PoseStamped()
        curpose.pose = data.pose[9]
        curpose.header.frame_id = "/map"
        self.pose_pub.publish(curpose)


    # Indoor pose
    def indoor_pose_callback(self, data):
                # No covariance for IPS in sim, so update ???
        # Stamp update
        # stamp = rospy.Time.now().to_sec()
        # if self.ips_prev_stamp is None:
        #     self.ips_prev_stamp = data.header.stamp
        if (self.meas == False).all():
            self.meas[0] = data.pose[9].position.x
            self.meas[1] = data.pose[9].position.y
            (roll, pitch, yaw) = euler_from_quaternion([data.pose[9].orientation.x, data.pose[9].orientation.y, data.pose[9].orientation.z, data.pose[9].orientation.w])
            self.meas[2] = yaw
        else:
            x = data.pose.pose.position.x
            y = data.pose.pose.position.y
            self.delta_t = np.sqrt((x - self.meas[0])**2 + (y - self.meas[1])**2)# + np.random.normal(0, self.noise_t*self.noise_t)
            self.delta_r = np.arctan2((y - self.meas[1]), (x - self.meas[0]))# + np.random.normal(0, self.noise_r*self.noise_r)
            self.meas[0] = x
            self.meas[1] = y
            (roll, pitch, yaw) = euler_from_quaternion([data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])
            self.meas[2] = yaw
        # Assign static covariance values   
        # self.cov_delta = np.diag(data.covariance[:3])
        # self.cov_delta = np.diag([0.1, 0.1, 0.1])
        # self.delta = data.header.stamp - self.ips_prev_stamp
        # self.ips_prev_stamp = data.header.stamp

        # Republish indoor pose to map frame as GROUND TRUTH
        curpose = PoseWithCovarianceStamped()
        curpose.pose.pose = data.pose.pose
        curpose.pose.covariance = data.pose.covariance
        curpose.header.frame_id = "/map"
        self.indoor_pub.publish(curpose)
        

    # Map callback
    def map_callback(self, msg, args):
        # This function is called when a new map is received
        # you probably want to save the map into a form which is easy to work with
        pass

    def linearize_motion(self):
        # Jacobian of G was determined analytically
        G = np.eye(3)
        # G[0,-1] = -self.delta * self.u[0]*np.sin(self.state_prior[2])
        # G[1,-1] = self.delta * self.u[0]*np.cos(self.state_prior[2])
        G[0,-1] = -self.delta_t*np.cos(self.state_prior[2])
        G[1,-1] = self.delta_t*np.sin(self.state_prior[2])
        return G
  
    def linearize_meas(self):
        # Jacobian of H was determined analytically (identity matrix)
        H = np.eye(3)
        return H

    
    def run(self):
        # # (1) predict means
        # self.state_prior[0] = self.state[0] + self.delta*self.u[0]*np.cos(self.state[2])
        # self.state_prior[1] = self.state[1] + self.delta*self.u[0]*np.sin(self.state[2])
        # self.state_prior[2] = self.state[2] + self.delta*self.u[1]
        self.state_prior[0] = self.state[0] + self.delta_t*np.cos(self.state[2])# + np.random.normal(0, self.noise_t*self.noise_t)
        self.state_prior[1] = self.state[1] + self.delta_t*np.sin(self.state[2])# + np.random.normal(0, self.noise_t*self.noise_t)
        self.state_prior[2] = self.state[2] + self.delta_r# + np.random.normal(0, self.noise_r*self.noise_r)
        # (2) predict covariance
        G = self.linearize_motion()
        self.cov_prior = G.dot(self.cov.dot(G.T)) + self.cov_eps
        # (3) compute kalman gain
        H = self.linearize_meas()
        K = np.dot(self.cov_prior.dot(H.T), np.linalg.inv(H.dot(self.cov_prior.dot(H.T)) + self.cov_delta))
        # (4) update mean
        innovation = self.meas - self.state_prior # since H is identity, may use state_prior directly
        self.state = self.state_prior + np.dot(K,innovation)
        # (5) update covariance
        diff = np.eye(3) - np.dot(K, H)
        self.cov = np.dot(diff, self.cov_prior)
        # self.cov = np.dot((np.eye(3) - np.dot(K, H)), self.cov_prior)

        # Publish corrected pose
        # rospy.loginfo("current state: [{}, {}, {}]".format(self.state[0, 0], self.state[1, 0], self.state[2, 0]))
        rospy.loginfo("diff: {}".format(diff    ))
        self.pose = PoseWithCovarianceStamped()
        self.pose.pose.pose.position.x = self.state[0, 0]
        self.pose.pose.pose.position.y = self.state[1, 0]
        self.pose.pose.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, self.state[2, 0]))
        new_cov = np.zeros((36,))
        new_cov[0] = self.cov[0, 0]
        new_cov[7] = self.cov[1, 1]
        new_cov[-1] = self.cov[2, 2]
        # new_cov[0] = 0.1
        # new_cov[7] = 0.1
        # new_cov[-1] = 0.05
        # new_cov[:9] = np.array([3, 0, 0, 0, 0, 0, 3, 0, 0])
        self.pose.pose.covariance = new_cov 
        self.pose.header.frame_id = "/odom"

        self.pose_ekf_pub.publish(self.pose)
        self.rate.sleep()

if __name__ == '__main__':

    try:
        estimator = StateEstimator()
        while not rospy.is_shutdown():
            estimator.run()
        
    except rospy.ROSInterruptException:
        pass

