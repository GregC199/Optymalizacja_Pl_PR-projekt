#!/usr/bin/env python

#########################################################################################################
import rospy
from geometry_msgs.msg import Twist,Pose
from nav_msgs.msg import Odometry
import numpy as np
import math
import sympy as sym
from sympy import *
from std_msgs.msg._Float64MultiArray import Float64MultiArray
from setuptools.command import rotate
from tensorflow.lite.python.schema_py_generated import AbsOptionsStart
import time
#########################################################################################################


#########################################################################################################
global next_move  #iterator do list kolejnych 
#######################################################################
#Initialize ROS node
rospy.init_node('Path_Planning_PPot', anonymous=True) #Node Path planning "Pola Potencjalne" (eng. potential field)
#######################################################################
PPot_Param = [rospy.get_param("~K_att"),rospy.get_param("~K_rep")] #Reinforcement attraction, repulsion
tau = rospy.get_param("~sampling_time") #Sampling Time
rob_mass = rospy.get_param("~rob_mass") #Robot Mass (Turtlebot 3 Waffle_pi)

Rob_rate = 10
#######################################################################
#ROS publisher code for velocity
pub1 = rospy.Publisher('/PPot_Des_Pos', Pose, queue_size=10) #Publisher "pub1" to publish at topic "/PPot_Des_Pos" to send message with destination
Des_Pos_msg = Pose() #Identify msg variable for sending destination
rate = rospy.Rate(Rob_rate) #Rate of publishing msg 10hz
#######################################################################
#Hard coded placement of obstacles
Obs_Pos_x = [ -7.5,-5.5,-3.5,-1.5,  5.0,5.0,  -8.5, -7.5, -5.5, -4.5, -3.5, -1.5, -1.5, 2.5, 4.5, 3.5, 4.5, 7.5 ]
Obs_Pos_y = [ -6.0,-6.0,-6.0,-6.0,  -7.5,-4.5,  1.5, 7.5, 4.5, 7.5, 1.5, 2.5, 6.5, 0.5, 6.5, 1.5, 2.5, 4.5 ]
#Hard coded length of obstacles
Obs_len_x = [ 0.5,0.5,0.5,0.5,  3.0, 3.0,  0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5 ]
Obs_len_y = [ 2.0,2.0,2.0,2.0,  0.5,0.5,  0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5 ]

#Algorithm parameters
dsafe=0.4 # Safe distance
dprewarn=0.75 # Prewarning distance
dnonlimit=1.5 # Increased speed distance


#######################################################################

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

def rotPath(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

def astar(start2, end2):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""
    
    start1 = (int(start2[0]),int(start2[0]))
    end1 = (int(end2[0]),int(end2[1]))
    print("start1 funkcja",start1, "end funkcja",end1)
    
    maze = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    
    helper_num = 10

    end_temp = [0,0]
    start_temp = [0,0]
    if end1[0] < 0 and end1[1] < 0:
        print("1")
        end_temp[0] = (end1[0] + helper_num)
        end_temp[1] = end1[1] + helper_num
    elif end1[0] > 0 and end1[1] > 0:
        print("2")
        end_temp[0] = end1[0] + helper_num
        end_temp[1] = end1[1] + helper_num        
    elif end1[0] > 0 and end1[1] < 0:
        print("3")
        end_temp[1] = (end1[1] - helper_num)*-1
        end_temp[0] = (end1[0] - helper_num)*-1
    elif end1[0] < 0 and end1[1] > 0:
        print("4")
        end_temp[0] = (end1[0] - helper_num)*-1
        end_temp[1] = end1[1] - helper_num
    elif end1[0] == 0 and end1[1] == 0:
        end_temp[0] = 10
        end_temp[1] = 10   
    print("astar temp ned",end_temp)
    

    if start1[0] < 0 and start1[1] < 0:
        print("1")
        start_temp[0] = (start1[0] + helper_num)
        start_temp[1] = start1[1] + helper_num
    elif start1[0] > 0 and start1[1] > 0:
        print("2")
        start_temp[0] = start1[0] + helper_num
        start_temp[1] = start1[1] + helper_num        
    elif start1[0] > 0 and start1[1] < 0:
        print("3")
        start_temp[1] = (start1[1] - helper_num)*-1
        start_temp[0] = (start1[0] - helper_num)*-1
    elif start1[0] < 0 and start1[1] > 0:
        print("4")
        start_temp[0] = (start1[0] - helper_num)*-1
        start_temp[1] = start1[1] - helper_num
    elif start1[0] == 0 and start1[1] == 0:
        start_temp[0] = 10
        start_temp[1] = 10   
    print("astar temp start",start_temp) 
    
    start = (start_temp[0],start_temp[1])
    end = (end_temp[0],end_temp[1])
    
    # Create start_temp and end_temp node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)

#######################################################################
'''
Transformation function for Euler to quaternion orientation.

Transforms Yaw, Pitch and Roll angles to x,y,z,w quaternion values.
'''
def euler_to_quaternion(yaw, pitch, roll):
    x = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    y = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    z = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    w = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [x, y, z, w]
#######################################################################

#######################################################################
'''
Transformation function for quaternion to Euler orientation.

Transforms x,y,z,w quaternion values to Yaw, Pitch and Roll angles.
'''
def quaternion_to_euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)       #
    t1 = +1.0 - 2.0 * (x * x + y * y) #
    roll = math.atan2(t0, t1)         #Calculation for Roll angle
    t2 = +2.0 * (w * y - z * x)       #
    t2 = +1.0 if t2 > +1.0 else t2    #
    t2 = -1.0 if t2 < -1.0 else t2    #
    pitch = math.asin(t2)             #Calculation for Pitch angle
    t3 = +2.0 * (w * z + x * y)       #
    t4 = +1.0 - 2.0 * (y * y + z * z) #
    yaw = math.atan2(t3, t4)          #Calculation for Yaw angle
    return [yaw, pitch, roll]
#######################################################################

#######################################################################
#ROS subscriber code for robot position
flag_cont = 0	              #Initialize flag by zero
pos_msg = Pose()	          #Identify msg variable for reading robot position
position = np.zeros((1,6))    #Create zeros array 1x6 for reading position
Velocity_msg = Twist()        #Identify msg variable for reading robot velocity
velocity = np.zeros((1,6))    #Create zeros array 1x6 for reading velocity
#######################################################################
#######################################################################
'''
Callback function for feedback the robot current position and velocity.

Callback function is called when a new message is received by the subscriber sub2.
'''
def callback(data):
    global pos_msg	#Identify msg variable created as global variable
    global sub2		#Identify a subscriber as global variable
    global flag_cont #Identify a flag for control runtime as global variable
    global position  #Identify a position of robot and orientation global variable
    global Velocity_msg #Identify a msg for robot velocity global variable
    global velocity #Identify velocity global variable
    
    msg = data
    pos_msg.position.x = round(msg.pose.pose.position.x, 4)		    #Round the value of x to 4 decimal places
    pos_msg.position.y = round(msg.pose.pose.position.y, 4)		    #Round the value of y to 4 decimal places
    pos_msg.position.z = round(msg.pose.pose.position.z, 4)		    #Round the value of z to 4 decimal places
    pos_msg.orientation.x = round(msg.pose.pose.orientation.x, 4)	#Round the value of theta to 4 decimal places
    pos_msg.orientation.y = round(msg.pose.pose.orientation.y, 4)	#Round the value of theta to 4 decimal places
    pos_msg.orientation.z = round(msg.pose.pose.orientation.z, 4)	#Round the value of theta to 4 decimal places
    pos_msg.orientation.w = round(msg.pose.pose.orientation.w, 4)	#Round the value of theta to 4 decimal places
    [yaw, pitch, roll] = quaternion_to_euler(pos_msg.orientation.x, pos_msg.orientation.y, pos_msg.orientation.z, pos_msg.orientation.w)
    position = [pos_msg.position.x,pos_msg.position.y,pos_msg.position.z,yaw, pitch, roll]
    Velocity_msg.linear.x = round(msg.twist.twist.linear.x, 4)
    Velocity_msg.linear.y = round(msg.twist.twist.linear.y, 4)
    Velocity_msg.linear.z = round(msg.twist.twist.linear.z, 4)
    Velocity_msg.angular.x = round(msg.twist.twist.angular.x, 4)
    Velocity_msg.angular.y = round(msg.twist.twist.angular.y, 4)
    Velocity_msg.angular.z = round(msg.twist.twist.angular.z, 4)
    velocity = [Velocity_msg.linear.x,Velocity_msg.linear.y,Velocity_msg.linear.z,Velocity_msg.angular.x,Velocity_msg.angular.y,Velocity_msg.angular.z]
    flag_cont = 1 #Set the flag to read

sub2 = rospy.Subscriber('/odom', Odometry, callback) #Identify the subscriber "sub2" to subscribe topic "/odom" - robot odometry
#######################################################################

#######################################################################
#ROS subscriber code for initial position
pos_msg_0 = Pose()	#Identify msg variable of initial robot position
position_0 = np.zeros((1,6)) #Identify variable for initial robot position
flag_initial = 0 #Set initial flag for 0 - waiting
Velocity_msg_0 = Twist() #Identify msg variable of initial robot velocity
velocity_0 = np.zeros((1,6)) #Identify variable for initial robot velocity
#######################################################################
#######################################################################
'''
Initial callback function for setting the initial robot position and velocity.

Callback function which is called when a new message is received by the subscriber sub1.
 '''
def callback_Init(data):
    global pos_msg_0		#Identify initial position and velocity msg variable created as global variable
    global sub1			#Identify a subscriber as global variable
    global flag_initial 	#Identify flag for initian read created as global variable
    global position_0   #Identify a initial position of robot and orientation global variable
    global Velocity_msg_0 #Identify a initial msg for robot velocity global variable
    global velocity_0 #Identify initial velocity global variable
    
    msg = data
    pos_msg_0.position.x = round(msg.pose.pose.position.x, 4)		#Round the value of x to 4 decimal places
    pos_msg_0.position.y = round(msg.pose.pose.position.y, 4)		#Round the value of y to 4 decimal places
    pos_msg_0.position.z = round(msg.pose.pose.position.z, 4)		#Round the value of y to 4 decimal places
    pos_msg_0.orientation.x = round(msg.pose.pose.orientation.x, 4)	#Round the value of theta to 4 decimal places
    pos_msg_0.orientation.y = round(msg.pose.pose.orientation.y, 4)	#Round the value of theta to 4 decimal places
    pos_msg_0.orientation.z = round(msg.pose.pose.orientation.z, 4)	#Round the value of theta to 4 decimal places
    pos_msg_0.orientation.w = round(msg.pose.pose.orientation.w, 4)	#Round the value of theta to 4 decimal places
    [yaw, pitch, roll] = quaternion_to_euler(pos_msg.orientation.x, pos_msg.orientation.y, pos_msg.orientation.z, pos_msg.orientation.w)
    position_0 = [pos_msg.position.x,pos_msg.position.y,pos_msg.position.z,yaw, pitch, roll]
    Velocity_msg_0.linear.x = round(msg.twist.twist.linear.x, 4)
    Velocity_msg_0.linear.y = round(msg.twist.twist.linear.y, 4)
    Velocity_msg_0.linear.z = round(msg.twist.twist.linear.z, 4)
    Velocity_msg_0.angular.x = round(msg.twist.twist.angular.x, 4)
    Velocity_msg_0.angular.y = round(msg.twist.twist.angular.y, 4)
    Velocity_msg_0.angular.z = round(msg.twist.twist.angular.z, 4)
    velocity_0 = [Velocity_msg_0.linear.x,Velocity_msg_0.linear.y,Velocity_msg_0.linear.z,Velocity_msg_0.angular.x,Velocity_msg_0.angular.y,Velocity_msg_0.angular.z]
    flag_initial = 1 #Set the initial flag to done
    sub1.unregister()				#Unsubsribe "sub1" client

sub1 = rospy.Subscriber('/odom', Odometry, callback_Init) #Identify the subscriber "sub1" to subscribe topic "/odom" for robot odometry
#######################################################################
#######################################################################
'''
Stop code here till subscribe the first msg of the robot position.
'''
while flag_initial == 0:
    pass
#######################################################################

#######################################################################
#Define the initial pose and velocity of the vehicle
Rob_pos_0 = [position_0[0],position_0[1],position_0[3]]
Roc_vel_0 = [velocity_0[0],velocity_0[5]]

#Write the initial x and y robot position to x_p and y_p variables
x_p = Rob_pos_0[0]
y_p = Rob_pos_0[1]

#Write the initial x and y-axis robot velocity to vel_p_x and vel_p_y variables
vel_p_x = Roc_vel_0[0]*cos(Rob_pos_0[2])
vel_p_y = Roc_vel_0[0]*sin(Rob_pos_0[2])
#######################################################################
#########################################################################################################

#########################################################################################################
#######################################################################
'''
PPot Inputs.
Goal reading function.
'''
#Define goal position variable
Goal_Pos = [0.0, 0.0]
def get_Goal(data):
    global Goal_Pos #Identify goal position global variable
    global sub_goal #Identify a subscriber as global variable
    Goal_Pos  = data.data
    #print(data.data)
       
sub_goal = rospy.Subscriber('goal', Float64MultiArray, get_Goal)      #Identify the subscriber "sub1" to subscribe topic containing goal position
#######################################################################
#######################################################################
'''
PPot calculation function.

Calculates the attraction and repulsion forces for current robot position.
The attraction force is reinforced with parabolic paramter to flatten the force.
'''
#######################################################################
def PPot_Fn(Rob_pos,Goal_pos,Obs_pos_x,Obs_pos_y,PPot_Param,dsafe,Obs_len_x,Obs_len_y):
    
    #Calculation of distance between robot and goal
    d_goal = sqrt((Rob_pos[0]-Goal_pos[0])**2+(Rob_pos[1]-Goal_pos[1])**2)
    
    #Initial calculation of attraction value
    Fx_att_val = -(0.05+1.6/(d_goal+0.165))*PPot_Param[0]*(Rob_pos[0]-Goal_pos[0])
    Fy_att_val = -(0.05+1.6/(d_goal+0.165))*PPot_Param[0]*(Rob_pos[1]-Goal_pos[1])

    #Definitions of initial repulsion values
    Fx_rep_val=0.0
    Fy_rep_val=0.0
    
    j=0     #While loop iterator of obstacles
    
    flag_safe = 0 #Flag for crossing of robot safe distance
    
    
    d_obs_min = 20.0 #Definition of starting minimal obstacle distance
    
    #Nonlimited speed linearization parameters
    a = 0.0
    b = 0.0
    
    #While loop of repulsion forces calculations
    while j<18:
        
        #Initialization of repulsion values
        Fx_rep_val_t = 0.0
        Fy_rep_val_t = 0.0
        
        #Calculation of distance between robot and center of obstacle for x and y-axis
        d_obs_val_x = sqrt((Rob_pos[0]-Obs_pos_x[j])**2)
        d_obs_val_y = sqrt((Rob_pos[1]-Obs_pos_y[j])**2)
        
        #Supplementation of calculation with obstacle walls length
        if(d_obs_val_x < Obs_len_x[j]):
            if(d_obs_val_y < Obs_len_y[j]):
                d_obs_rob = 0.0
            else:
                d_obs_rob = sqrt((d_obs_val_y-Obs_len_y[j])**2)
        else:
            if(d_obs_val_y < Obs_len_y[j]):
                d_obs_rob = sqrt((d_obs_val_x-Obs_len_x[j])**2)
            else:
                d_obs_rob = sqrt((d_obs_val_x-Obs_len_x[j])**2+(d_obs_val_y-Obs_len_y[j])**2)
        
        #Replacement of previous minimal distance between robot and obstacle
        if d_obs_rob < d_obs_min:
            d_obs_min = d_obs_rob
        
        #Calculation of repulsion forces of number 'j' obstacle in the prewarning range 
        if d_obs_rob < dprewarn:
            Fx_rep_val_t = 1.5*PPot_Param[1]*(1-(d_obs_rob/dprewarn))*((Rob_pos[0]-Obs_pos_x[j])/(d_obs_rob**3))
            Fy_rep_val_t = 1.5*PPot_Param[1]*(1-(d_obs_rob/dprewarn))*((Rob_pos[1]-Obs_pos_y[j])/(d_obs_rob**3))                                                       
        elif (d_obs_rob < dsafe):
            flag_safe = 1 #Setting up for the crossing of safe distance by robot
            
            #Further reinforcement of repulsion forces after crossing safe distance
            Fx_rep_val_t = 6.0*Fx_rep_val_t
            Fy_rep_val_t = 6.0*Fy_rep_val_t
        else:
            Fx_rep_val_t = 0.0 #Setting repulsion forces to 0 that arent in the prewarning range
            Fy_rep_val_t = 0.0
            
        #Increasing repulsion force with "j" obstacle repulsion
        Fx_rep_val += Fx_rep_val_t
        Fy_rep_val += Fy_rep_val_t
        
        j +=1 #Increment "j" number of obstacle
        
    #Letting the robot to dive in the repulsion field if it fits
    if ((d_obs_min - 0.25) >  d_goal):
        Fx_rep_val = Fx_rep_val*d_goal*0.5
        Fy_rep_val = Fy_rep_val*d_goal*0.5
    
    #Robot forces change depending on safe distance crossing    
    if flag_safe == 1:
        Fx_net_val = (Fx_att_val + Fx_rep_val)/3.0
        Fy_net_val = (Fy_att_val + Fy_rep_val)/3.0
    else:
        Fx_net_val = Fx_att_val + Fx_rep_val
        Fy_net_val = Fy_att_val + Fy_rep_val
        
    #Reinforcement of robot forces when there is no risk of hitting obstacle
    if (d_obs_min > dnonlimit):
        Fx_net_val = Fx_net_val*4.0
        Fy_net_val = Fy_net_val*4.0
    elif ((d_obs_min <= dnonlimit) and (d_obs_min > dprewarn)):
        #Linearization of forces to match the increased speed distance
        a = -3.0 / ( dprewarn - dnonlimit )
        b = 4.0 - a * dnonlimit
        Fx_net_val = Fx_net_val*(d_obs_min*a+b)
        Fy_net_val = Fy_net_val*(d_obs_min*a+b)
        
    #Decreasion of forces to stop robot scrabbling close to goal point
    if (d_goal < 0.05):
        Fx_net_val = Fx_net_val*0.5
        Fy_net_val = Fy_net_val*0.5
        
    F_xy_net = [Fx_net_val,Fy_net_val]
    return F_xy_net
#######################################################################
#########################################################################################################

#########################################################################################################
#Simulation While Loop
'''
start = (0,0)
end = (-9,-9)
robot_path = astar(start, end)


for touple, index in enumerate(robot_path): 
    point_np = np.asarray(robot_path[touple])
    point_np = rotPath(point_np)
    point_np[0] = point_np[0] - 10
    point_np[1] = point_np[1] - 10
    robot_path[touple] = tuple(point_np)'''
    
#path_length = len(robot_path)
flaga_astar = 1
next_move = 0
Temp_Goal = [0,0]
Goal_Pos[0] = 0
Goal_Pos[1] = 0
path_length = 0
Old_goal = [0,0] 
robot_path = []
while 1:

    next_move = 0
    path_length = 0
    robot_path.clear()
    
    print("Robot path CLEAR ",robot_path)

    time.sleep(1)
    
    if Goal_Pos[0] != 0 and Goal_Pos[1] != 0 and Old_goal[0] != Goal_Pos[0] and Old_goal[1] != Goal_Pos[1]:
        #start = (position[0],position[1])
        #end = (Goal_Pos[0], Goal_Pos[1]) 
        robot_path = astar((position[0],position[1]), (Goal_Pos[0], Goal_Pos[1]) )
        path_length = len(robot_path)
        Old_goal[0] = Goal_Pos[0]
        Old_goal[1] = Goal_Pos[1]
        print("sciezka robota",robot_path)
        

    print("start",(position[0],position[1]))
    print("end",(Goal_Pos[0], Goal_Pos[1]))
    

    
    next_move = 0
        
    
    if path_length > 2:
        for touple, index in enumerate(robot_path): 
            point_np = np.asarray(robot_path[touple])
            point_np = rotPath(point_np)
            point_np[0] = point_np[0] - 10
            point_np[1] = point_np[1] - 10
            robot_path[touple] = tuple(point_np)  
            print("sciezka robota po przelozeniu",robot_path)
        while 1 and not rospy.is_shutdown():
            #New data aviable flag check - control runtime flag
            if flag_cont == 1:
                
                #Obtaining of robot current position and velocity
                Rob_pos = [position[0],position[1],position[3]]
                Rob_vel = [velocity[0],velocity[5]]
                
                if next_move < 1:
                    Temp_Goal[0] = robot_path[next_move+1][0]
                    Temp_Goal[1] = robot_path[next_move+1][1]
                
                print("Aktualna pozycja gola:", Temp_Goal)  
                rob_goal_dist_x = abs(position[0] - Temp_Goal[0])
                rob_goal_dist_y = abs(position[1] - Temp_Goal[1])
                print("Pozycja",position)
                
                print(rob_goal_dist_x)
                print(rob_goal_dist_y)
                
                print("nextmove",next_move)
                print("path length", len(robot_path))
                
                if next_move >= len(robot_path) - 1:
                    next_move = 0    
                    break
                
                elif rob_goal_dist_x < 0.9 and rob_goal_dist_y < 0.9 and next_move != len(robot_path)-1:
                    print("WSZEDŁEM")
                    Temp_Goal[0] = robot_path[next_move+1][0]
                    Temp_Goal[1] = robot_path[next_move+1][1]
                    next_move = next_move + 1

                    
                #PPot function call
                F_xy_net = PPot_Fn(Rob_pos,Temp_Goal,Obs_Pos_x,Obs_Pos_y,PPot_Param,dsafe,Obs_len_x,Obs_len_y)
                
                #Calculation of distance and direction of movement
                F_net = float(sqrt(F_xy_net[0]**2 + F_xy_net[1]**2))
                F_net_direct = float(atan2(F_xy_net[1], F_xy_net[0]))
                
                #Calculate the desired robot position from the PPot
                vel_c_x = vel_p_x + (F_xy_net[0]/rob_mass)*tau
                vel_c_y = vel_p_y + (F_xy_net[1]/rob_mass)*tau
                x_des = x_p + vel_c_x*tau
                y_des = y_p + vel_c_y*tau
                
                #Writing the new destination and orientation Rob_pos_des variable
                Rob_pos_des = [x_des,y_des,F_net_direct]
                
                #Update the previous robot states for the next iteration
                vel_p_x = Rob_vel[0]*cos(Rob_pos[2])
                vel_p_y = Rob_vel[0]*sin(Rob_pos[2])
                x_p = Rob_pos[0]
                y_p = Rob_pos[1]
                flag_cont = 0
        
            else:
                Rob_pos_des = Rob_pos
                
            Des_Pos_msg.position.x = Rob_pos_des[0]
            Des_Pos_msg.position.y = Rob_pos_des[1] #Writing quaternion values to destination msg
            Des_Pos_msg.position.z = 0
            [qx_des, qy_des, qz_des, qw_des] = euler_to_quaternion(Rob_pos_des[2], 0, 0) #Transformation of Yaw robot angle to quaternion orientation
            Des_Pos_msg.orientation.x = qx_des  #Writing quaternion values to destination msg
            Des_Pos_msg.orientation.y = qy_des  #Writing quaternion values to destination msg
            Des_Pos_msg.orientation.z = qz_des  #Writing quaternion values to destination msg
            Des_Pos_msg.orientation.w = qw_des  #Writing quaternion values to destination msg
        
            pub1.publish(Des_Pos_msg)	#Publish msg of destination position
            rate.sleep()		        #Sleep with rate
            
    time.sleep(1)
        #########################################################################################################
