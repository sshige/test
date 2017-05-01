#!/usr/bin/env python

import numpy as np
import rospy
import tf
import tf.transformations
import geometry_msgs.msg
import tf_conversions.posemath as pm

# sample object frame
obj_pose_ = geometry_msgs.msg.Pose()
obj_pose_.position.x = 123.0
obj_pose_.orientation.w = 1.0
obj_frame_ = pm.fromMsg(obj_pose_)

# rotation around y
conv_matrix_ = tf.transformations.rotation_matrix(np.pi / 2.0, [0, 1, 0])
conv_frame_ = pm.fromMatrix(conv_matrix_)

# rotate object frame
conved_obj_frame_ = obj_frame_ * conv_frame_
print(conved_obj_frame_.p) # pos
print(conved_obj_frame_.M) # rotation
# or
conved_obj_frame_ = conv_frame_ * obj_frame_
print(conved_obj_frame_.p) # pos
print(conved_obj_frame_.M) # rotation
