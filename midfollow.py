#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import sys
import cv2
import numpy as np
from PIL import Image as PImage
import time
import math
from math import *
import rospy
from std_msgs.msg import Int32, Float32
from sensor_msgs.msg import  Image, Joy
from motion_control import *
from image_ros_msgs.msg import BoundingBox, BoundingBoxes
from geometry_msgs.msg import  Twist
from Rosrobot import rosrobot
global flag0
global flag1
global flag2
global flaga
global flagb
global flagc
global flagd
global flage
global flagf
global flagg
global flagh
global flagi
global flagj
global flagp1
global flagp2
flag0=True
flag1 = False
flag2 = False

flaga = True
flagb = False
flagc = False
flagd = False
flage = False
flagf = False
flagg = False
flagh = False
flagi = False
flagj = False
flagp1 = True
flagp2 = False

global huandao
huandao=False

def mid(follow, mask, img):
    height = follow.shape[0]  # 输入图像高度
    width = follow.shape[1]  # 输入图像宽度

    half = int(width / 2)  # 输入图像中线

    # 从下往上扫描赛道,最下端取图片中线为分割线
    for y in range(height - 1, -1, -1):

        if y == height - 1:  # 刚开始从底部扫描时
            left = 0
            right = width - 1
            left_scale = 0.5  # 初始赛道追踪范围
            right_scale = 0.5  # 初始赛道追踪范围
        elif left == 0 and right == width - 1:  # 下层没有扫描到赛道时
            left_scale = 0.25  # 赛道追踪范围
            right_scale = 0.25  # 赛道追踪范围
        elif left == 0:  # 仅左下层没有扫描到赛道时
            left_scale = 0.25  # 赛道追踪范围
            right_scale = 0.2  # 赛道追踪范围
        elif right == width - 1:  # 仅右下层没有扫描到赛道时
            left_scale = 0.2  # 赛道追踪范围
            right_scale = 0.25  # 赛道追踪范围
        else:
            left_scale = 0.2  # 赛道追踪范围
            right_scale = 0.2  # 赛道追踪范围

        # 根据下层左线位置和scale，设置左线扫描范围
        left_range = mask[y][max(0, left - int(left_scale * width)):min(left + int(left_scale * width), width - 1)]
        # 根据下层右线位置和scale，设置右线扫描范围
        right_range = mask[y][max(0, right - int(right_scale * width)):min(right + int(right_scale * width), width - 1)]

        # 左侧规定范围内未找到赛道
        if (left_range == np.zeros_like(left_range)).all():
            left = left  # 取图片最左端为左线
        else:
            left = int(
                (max(0, left - int(left_scale * width)) + np.average(
                    np.where(left_range == 255))) * 0.4 + left * 0.6)  # 取左侧规定范围内检测到赛道像素平均位置为左线

        # 右侧规定范围内未找到赛道
        if (right_range == np.zeros_like(right_range)).all():
            right = right  # 取图片最右端为右线
        else:
            right = int(
                (max(0, right - int(right_scale * width)) + np.average(
                    np.where(right_range == 255))) * 0.4 + right * 0.6)  # 取右侧规定范围内检测到赛道像素平均位置为右线

        mid = int((left + right) / 2)  # 计算中点

        follow[y, mid] = 255  # 画出拟合中线,实际使用时为提高性能可省略
        # img[y, max(0, left - int(left_scale * width)):min(left + int(left_scale * width), width - 1)] = [0, 0, 255]
        # img[y, max(0, right - int(right_scale * width)):min(right + int(right_scale * width), width - 1)] = [0, 0, 255]

        if y == int((360 / 480) * follow.shape[0]):  # 设置指定提取中点的纵轴位置
            mid_output = mid
    cv2.circle(follow, (mid_output, int((360 / 480) * follow.shape[0])), 5, 255, -1)  # opencv为(x,y),画出指定提取中点

    error = (half - mid_output) / width * 640  # 计算图片中点与指定提取中点的误差

    return follow, error, img  # error为正数右转,为负数左转
def cal(mask):
    global flag0
    global flag1
    global flag2
    global flaga
    global flagb
    global flagc
    global flagd
    global flage
    global flagf
    global flagg
    global flagh
    global flagi
    global flagj
    global flagp1
    global flagp2
    # print(cal1(mask)[1], cal2(mask)[1], cal3(mask)[1])
    # 阶段一（刚进入任务）》阶段二（开始转圈）
    if flagp1:
        if cal1(mask)[1] < -50/6.4 and flag0:
            flag1 = True
            flag0=False
            print(0)
        if flag1 == True:
            if cal1(mask)[1] > -20/6.4:
                flag2 = True
                flag1 = False
                print(1)
        if flag2 == True:
            if cal1(mask)[1] <-50/6.4:
                flagp1 = False
                flagp2 = True

    # 阶段二（转圈）》阶段三（返回基地）
    if flagp2:

        if cal1(mask)[1] < -50/6.4 and flaga:
            flagb = True
            flaga = False
#            print(0000000000000)
        if flagb == True:
            if cal1(mask)[1] > -20/6.4:
                flagc = True
                flagb = False
#                print(111111111111)
        if flagc == True:
            if cal1(mask)[1] < -80/6.4:
                flagd = True
                flagc = False
#                print(2222222)
        if flagd == True:
            if cal1(mask)[1] > -20/6.4:
                flage = True
                flagd = False
#                print(3333333333)
        if flage == True:
            if cal1(mask)[1] < -70/6.4:
                flagf = True
                flage = False
#                print(444444444444444)
        if flagf == True:
            if cal1(mask)[1] > -20/6.4:
                flagg = True
                flagf = False
#                print(5555555555555)
        if flagg == True:
            if cal1(mask)[1] < -80/6.4:
                flagh = True
                flagg = False
#                print(666666666666)
        if flagh == True:
            if cal1(mask)[1] > -20/6.4:
                flagi=True
                flagh=False
#                print(777777)
        if flagi == True:
            if cal1(mask)[1] <-90/6.4 and cal2(mask)[1]<0:
                flagj=True
                flagi=False
        if flagj==True:
            if cal1(mask)[1] > -85/6.4 and cal1(mask)[1] < 50/6.4:
                flagp2=False
    # 对应阶段返回输出
    if flagp1:
        return cal2(mask)
    elif flagp2:
        return cal3(mask)
    else:
        return cal2(mask)


# 原图
def cal1(mask):
    follow = mask.copy()

    height = 100  # 输入图像高度
    width = 100  # 输入图像宽度

    half = 50

    # 扫描过程
    for y in range(height - 1, -1, -1):

        left_range = follow[y][0:half]
        right_range = follow[y][half:width - 1]

        # 左侧规定范围内未找到赛道
        if (left_range == np.zeros_like(left_range)).all():
            left = 0
        else:
            left = int(np.where(left_range == 255)[0][-1])

        # 右侧规定范围内未找到赛道
        if (right_range == np.zeros_like(right_range)).all():
            right = width - 1
        else:
            right = half + int(np.where(right_range == 255)[0][0])

        mid = int((left + right) / 2)  # 计算中点

        follow[y, mid] = 255  # 画出拟合中线,实际使用时为提高性能可省略

        if y == int(360/480*100):  # 设置指定提取中点的纵轴位置
            mid_output = mid

    cv2.circle(follow, (mid_output, int(360/480*100)), 5, 255, -1)  # opencv为(x,y),画出指定提取中点

    error = half - mid_output  # 计算图片中点与指定提取中点的误差
    error=error*7
    return follow, error


# 补右车道
def cal2(mask):
    follow = mask.copy()
    cv2.line(follow, [91, 133], [56, 0], 255, 4)


    height = 100  # 输入图像高度
    width = 100  # 输入图像宽度
    
    half = 50

    # 扫描过程
    for y in range(height - 1, -1, -1):

        left_range = follow[y][0:half]
        right_range = follow[y][half:width - 1]

        # 左侧规定范围内未找到赛道
        if (left_range == np.zeros_like(left_range)).all():
            left = 0
        else:
            left = int(np.where(left_range == 255)[0][-1])

        # 右侧规定范围内未找到赛道
        if (right_range == np.zeros_like(right_range)).all():
            right = width - 1
        else:
            right = half + int(np.where(right_range == 255)[0][0])

        mid = int((left + right) / 2)  # 计算中点

        follow[y, mid] = 255  # 画出拟合中线,实际使用时为提高性能可省略

        if y == int(360/480*100):  # 设置指定提取中点的纵轴位置
            mid_output = mid

    cv2.circle(follow, (mid_output, int(360/480*100)), 5, 255, -1)  # opencv为(x,y),画出指定提取中点


    error = half - mid_output  # 计算图片中点与指定提取中点的误差
    error=error*9
    return follow, error



# 补外环岛
def cal3(mask):
    follow = mask.copy()
    cv2.circle(follow, (102, 125), 90, 255, 6)


    height = 100  # 输入图像高度
    width =100  # 输入图像宽度

    half = 50
    # 扫描过程
    for y in range(height - 1, -1, -1):

        left_range = follow[y][0:half]
        right_range = follow[y][half:width - 1]

        # 左侧规定范围内未找到赛道
        if (left_range == np.zeros_like(left_range)).all():
            left = 0
        else:
            left = int(np.where(left_range == 255)[0][-1])

        # 右侧规定范围内未找到赛道
        if (right_range == np.zeros_like(right_range)).all():
            right = width - 1
        else:
            right = half + int(np.where(right_range == 255)[0][0])

        mid = int((left + right) / 2)  # 计算中点

        follow[y, mid] = 255  # 画出拟合中线,实际使用时为提高性能可省略

        if y == int(360/480*100):  # 设置指定提取中点的纵轴位置
            mid_output = mid

    cv2.circle(follow, (mid_output, int(360/480*100)), 5, 255, -1)  # opencv为(x,y),画出指定提取中点

    error = half - mid_output  # 计算图片中点与指定提取中点的误差
    error=error*5.5
    return follow, error
    
class CONTROLER:
    def __init__(self):
        #PID相关参数
        self.target_point = 0.0
        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.Itime=243
        self.last_control_time = time.time() * self.Itime
        self.current_time = self.last_control_time
        self.last_error = 0.0        
        self.P = 0.6
        self.I = 1.0
        self.D = 0.05
        self.pid_out_rate = 0.0000000000007
        self.pid_out_power =7
        self.windup_guard = 25.0

        #图像相关
        self.RC_buff_length = 5
        self.history_left_lane = [0] * self.RC_buff_length  # 惯性滤波缓存区
        self.history_right_lane = [0] * self.RC_buff_length  # 惯性滤波缓存区
        self.history_angle = [float(0.0)] * self.RC_buff_length  # 惯性滤波缓存区


        self.Frame_image = None
        #巡线相关
        self.camera_correction_done = True    # 摄像头校准标志
        self.last_direction = 0                 # 上次转弯的方向,debug用
        self.follow_type = True                 #巡线开关
        self.first_stop =True
        self.vo_flag_1=False
        self.vo_flag_2=False
        self.vo_flag_3=False

        self.first_tower = False
        self.second_tower = False
        self.sleep_time = 0.5

        #速度相关
        self.max_speed = 0.4
        self.min_speed = 0.3
        self.sign_speed = self.max_speed
        self.vel = Twist()
        self.pid_cmd_vel_x=0
        self.pid_cmd_vel_z=0
        self.scale_1 = 4
        self.scale_2 = 3.6
        #ros相关

        rospy.init_node('follow_line', anonymous=False)
        self.cam_sub=rospy.Subscriber("/usb_cam/image_raw", Image, self.Image_callback)      #获取图像
        self.follow_sub=rospy.Subscriber('/follow_type', Int32, self.follow_type_callback )     #获取巡线状态
        self.vel_pub = rospy.Publisher("/cmd_vel",Twist,queue_size = 1000)      #发布速度
        rospy.Subscriber('/control_type', Int32, self.control_type_callback)
        rospy.Subscriber('/odom', Odometry, self.get_odom)
        self.coor =motion_control()
        rospy.on_shutdown(self.cancel)
    
    def control_type_callback(self,data):
        if data.data == 4:
            self.P = 1.0
            self.I = 1.3
            self.D = 0.2
            self.scale_1 = 5
            self.scale_2 = 5
        if not self.first_tower and data.data == 3:
            self.P = 1.25
            self.I = 1.47
            self.D = 0.0
            self.scale_1 = 3
            self.scale_2 = 8
#            print(0)
            self.first_tower = True
        if self.first_tower and not self.second_tower and data.data == 3:
            self.P = 1.25
            self.I = 1.4
            self.D = 0.0
            self.scale_1 = 3
            self.scale_2 = 5
#            print(0)
            self.second_tower = True
        if data.data == 7:  #渔舟唱晚
            self.P = 1.3
            self.I = 1.45
            self.D = 0.0
            self.scale_1 = 3
            self.scale_2 = 5
            self.sleep_time = 0.3


        
    def get_odom(self,msg):  #获取里程计的当前的位姿（7元素）
        self.orientationx = msg.pose.pose.position.x
        self.orientationy = msg.pose.pose.position.y

    def cancel(self):
        self.send_VelOrder(0,0,0)

        self.cam_sub.unregister()
        self.follow_sub.unregister()
        self.vel_pub.unregister()


    def get_servo_angle(self):
        try:
            """
            :return:舵机数值
            """
            self.output = self.output ** int(self.pid_out_power)
            angle = self.pid_out_rate * self.output
            #print("pid_power_out=", self.output, "angle=", angle)
            return angle
        except:
            s = sys.exc_info()
            print("Error '%s' happened on line %d" % (s[1], s[2].tb_lineno))

    def update(self, feedback_value):
        """Calculates PID value for given reference feedback
        .. math::
            u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}
        """
        error = self.target_point - feedback_value
        error = error
        self.current_time = time.time() * self.Itime  # ms
        delta_time = self.current_time - self.last_control_time
        delta_error = error - self.last_error

        self.PTerm = self.P * error
        self.ITerm += error * delta_time

        # 超调震荡窗口
        if (self.ITerm < -self.windup_guard):
            self.ITerm = -self.windup_guard
        elif (self.ITerm > self.windup_guard):
            self.ITerm = self.windup_guard

        self.DTerm = 0.0
        if delta_time > 0:
            self.DTerm = delta_error / delta_time

        # Remember last time and last error for next calculation
        self.last_control_time = self.current_time
        self.last_error = error

        self.output = self.PTerm + (self.I * self.ITerm) + (self.D * self.DTerm)

    def put_left_lane(self, data):
        for i in range(self.RC_buff_length - 1):
            self.history_left_lane[i] = self.history_left_lane[i + 1]
        self.history_left_lane[self.RC_buff_length - 1] = data

    def put_right_lane(self, data):
        for i in range(self.RC_buff_length - 1):
            self.history_right_lane[i] = self.history_right_lane[i + 1]
        self.history_right_lane[self.RC_buff_length - 1] = data

    def put_angle(self, data):
        for i in range(len(self.history_angle)-1):
            self.history_angle[i] = self.history_angle[i + 1]
        self.history_angle[-1] = data

    def get_history_turn(self):
        # 舵机角度<0为左转
        angle = sum(self.history_angle) / len(self.history_angle)
        if angle < 0:
            return 1
        elif angle > 0:
            return 2
        else:
            return 0
      
    def get_history_angle(self):
        return sum(self.history_angle) / len(self.history_angle)

    def get_left_lane_history(self):
        return sum(self.history_left_lane[:self.RC_buff_length])

    def get_right_lane_history(self):
        return sum(self.history_right_lane[:self.RC_buff_length])

    def call_back_30hz(self, cv_image):
        try:

            global flag1
            global flag2
            global flaga
            global flagb
            global flagc
            global flagd
            global flage
            global flagf
            global flagg
            global flagh
            global flagi
            global flagj
            global flagp1
            global flagp2
            global huandao

            huandao=False
    

            if not self.camera_correction_done:
                line_low = 0.99
                line_up = 0.85
                # 相机校准过程中，图像不缩放，画出中线以供校准
#                H = cv_image.shape[0]
#                W = cv_image.shape[1]
#                cv2.line(cv_image, pt1=(0, int(H * line_low)), pt2=(W, int(H * line_low)), color=(0, 255, 0), thickness=2)
#                cv2.line(cv_image, pt1=(0, int(H * line_up)), pt2=(W, int(H * line_up)), color=(0, 255, 0), thickness=2)
#                cv2.line(cv_image, pt1=(W // 2, 0), pt2=(W // 2, H), color=(0, 255, 0), thickness=2)
                cv2.imshow("lanes", cv_image)
                if  cv2.waitKey(1) == 32:
                    self.camera_correction_done = True
                    cv2.destroyAllWindows()
            else:
#                tt=time.time()
                img = cv_image
#                cv2.imshow("x",cv_image)
                
                if huandao==False:
                    img = cv2.resize(img, (100,100))
                    # HSV阈值分割
                    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(img_hsv, np.array([43, 60, 90]), np.array([62, 255, 255]))

                    follow = mask.copy()
                    follow , error, img = mid(follow, mask,img)
                
                    error=error-21
                    bais = error*0.3
#                    cv2.imshow("follow",follow)

                else:
                    img = cv2.resize(img, (100,100))
                # HSV阈值分割
                    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(img_hsv, np.array([43, 60, 90]), np.array([62, 255, 255]))

                    follow = mask.copy()
                    follow,error=cal(mask)
                    error=error-21
                    bais = error*0.3
#                    cv2.imshow("mask",mask)
#                    cv2.imshow("follow",follow)

#                if  cv2.waitKey(1) == 32:pass


#                print("------------------------------sbais:",bais)


                self.update(bais)
                angle_value = self.get_servo_angle()*self.scale_1
                self.put_angle(angle_value)



                angle_value = self.get_history_angle()  # 取惯性滤波后的角度值
                if angle_value>1 :angle_value=1
                if angle_value <-1:angle_value=-1
                #print(angle_value)
                
                # 正数右转，负数左转
                self.last_direction  = pid.get_history_turn()
                self.pid_cmd_vel_x   =  self.sign_speed 
                self.pid_cmd_vel_z   =  -angle_value

                if self.follow_type : 
                    self.send_VelOrder(self.pid_cmd_vel_x,0,self.pid_cmd_vel_z)
                    self.first_stop =True
                if self.follow_type ==0:
                    if self.first_stop: 
                        self.send_VelOrder(0,0,0)
                        time.sleep(0.5)
                        self.send_VelOrder(0,0,0)
                        self.first_stop = False
                    pass
#                print(time.time()-tt)

        except:
            s = sys.exc_info()
            print("Error '%s' happened on line %d" % (s[1], s[2].tb_lineno))

    
    def follow_type_callback(self,data):
        if data.data == 3:
            self.sign_speed =self.min_speed               #慢速巡线
            data.data = 1

        if data.data == 4:                                #快速巡线
            self.sign_speed =self.max_speed
            data.data = 1

        if data.data == 5:                                #第一次识别漩涡
            self.sign_speed =0.28
            self.vo_flag_1 = True
            self.start_x=self.orientationx
            self.start_y=self.orientationy
            data.data = 1

        if data.data == 6:                                 #第一次识别漩涡
            self.sign_speed =0.28
            self.vo_flag_2= True
            self.start_x=self.orientationx
            self.start_y=self.orientationy
            data.data = 1

        if data.data == 7:                                 #第一次识别漩涡
            self.sign_speed =0.28
            self.vo_flag_3 = True
            self.start_x=self.orientationx
            self.start_y=self.orientationy
            data.data = 1
        self.follow_type=data.data

    def send_VelOrder(self,v_x,v_y,v_w):      
            self.vel.linear.x = v_x
            self.vel.linear.y = v_y
            self.vel.angular.z = v_w
            self.vel_pub.publish(self.vel)

    def Image_callback(self, image):
        if not isinstance(image, Image): return
        self.boundingBoxes = BoundingBoxes()
        self.boundingBoxes.header = image.header
        self.boundingBoxes.image_header = image.header
        self.getImageStatus = True
        self.color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(
            image.height, image.width, -1)
        self.Frame_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
    
    def main(self):
        while not rospy.is_shutdown():
            if not isinstance(self.Frame_image, np.ndarray): 
                continue 

            self.call_back_30hz(self.Frame_image)



if __name__=="__main__":
	car = rosrobot()
	car.set_pwm_servo_all(90,0,60,0)
	pid = CONTROLER()
	pid.main()
	print("done")
