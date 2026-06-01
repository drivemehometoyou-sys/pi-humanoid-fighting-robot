import cv2
import numpy as np
import threading
import apriltag
import uptech
import time
import servo_test as st
import threading
import closed_loop_controller
from simple_pid import PID

up=uptech.UpTech()
clc1 = closed_loop_controller.ClosedLoopController()

class ApriltagDetect:
    def __init__(self):
        self.target_id = 10  # 目标ID初始化
        self.at_detector = apriltag.Detector(apriltag.DetectorOptions(families='tag36h11 tag25h9'))
        self.enemytag = 1  # 敌方二维码ID
        self.friendtag = 0  # 我方二维码ID

    def get_apriltag_area(self, tag):
        """计算二维码的面积"""
        corners = tag.corners.astype(int)
        s_1 = abs((corners[1][0] + corners[2][0]) / 2 - (corners[0][0] + corners[3][0]) / 2) * \
              abs((corners[2][1] + corners[3][1]) / 2 - (corners[0][1] + corners[1][1]) / 2)
        s_2 = abs((corners[0][0] + corners[1][0]) / 2 - (corners[2][0] + corners[3][0]) / 2) * \
              abs((corners[1][1] + corners[2][1]) / 2 - (corners[0][1] + corners[3][1]) / 2)
        return max(s_1, s_2)

    def detect_frame(self, frame):
        """检测当前帧中的二维码"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = self.at_detector.detect(gray)
        if tags:
            for tag in tags:
                self.draw_bounding_box(frame, tag)  # 为所有二维码打框
            return [(self.calculate_offset(frame.shape, tag), tag) for tag in tags]
        return []

    def calculate_offset(self, frame_shape, tag):
        """计算二维码中心与图像中心的偏移量"""
        frame_center = np.array([frame_shape[1] // 2, frame_shape[0] // 2])
        tag_center = np.mean(tag.corners, axis=0).astype(int)
        offset = tag_center - frame_center
        return int(offset[0]), int(offset[1])

    def classify_target(self, tag_id):
        """判断目标是敌方还是友方"""
        if tag_id == self.enemytag:
            return 1  # 敌方
        elif tag_id == self.friendtag:
            return 2  # 友方
        return 0  # 未识别

    def draw_bounding_box(self, frame, tag):
        """在图像上框出二维码"""
        corners = tag.corners.astype(int)
        cv2.polylines(frame, [corners], isClosed=True, color=(0, 255, 0), thickness=2)

    def get_largest_enemy_tag_info(self, tags):
        """获取面积最大的敌方二维码的偏移量和面积"""
        largest_enemy_tag = None
        max_area = 0
        largest_offset = (0, 0)

        for offset, tag in tags:
            if self.classify_target(tag.tag_id) == 1:  # 判断是否为敌方二维码
                area = self.get_apriltag_area(tag)  # 计算面积
                if area > max_area:
                    max_area = area
                    largest_enemy_tag = tag
                    largest_offset = offset

        if largest_enemy_tag:
            return largest_offset, max_area  # 返回偏移量和面积
        else:
            return None, 0  # 如果没有找到敌方二维码，则返回None


# 摄像头部分不变
class Camera:
    def __init__(self, camera):
        self.frame = None
        self.ret = False
        self.camera = camera
        self.openflag = False
        self.cap = None

    def open(self):
        """打开摄像头"""
        try:
            self.cap = cv2.VideoCapture(self.camera)
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open camera")

            self.cap.set(3, 640)
            self.cap.set(4, 320)
            self.openflag = True
            threading.Thread(target=self.queryframe, args=()).start()

        except Exception as e:
            print(f"摄像头打开失败: {e}")

    def queryframe(self):
        """不断读取摄像头帧"""
        while self.openflag:
            self.ret, self.frame = self.cap.read()

    def getframe(self):
        """返回读取状态和当前帧"""
        return self.ret, self.frame

    def close(self):
        """关闭摄像头"""
        self.openflag = False
        if self.cap:
            self.cap.release()


def Camera_New():
    global ENER_FLAG
    global largest_enemy_offset
    global largest_enemy_offset1
    global largest_enemy_area
    global largest_enemy_area1
    
    camera_index = 0
    camera = Camera(camera_index)
    detector = ApriltagDetect()
    largest_enemy_offset1=0
    largest_enemy_area1=0
    try:
        camera.open()
        threading.Event().wait(1)
        while True:
            ret, frame = camera.getframe()
            if not ret:
                print("未能读取帧")
                break

            tag_offsets = detector.detect_frame(frame)

            # 获取面积最大的敌方二维码的偏移量和面积
            largest_enemy_offset, largest_enemy_area = detector.get_largest_enemy_tag_info(tag_offsets)

            if largest_enemy_offset is not None:
                ENER_FLAG=1
                largest_enemy_offset1=largest_enemy_offset[0]
                largest_enemy_area1=largest_enemy_area
                print(f"最大敌方二维码的偏移量: {largest_enemy_offset}, 面积: {largest_enemy_area}")
            else:
                ENER_FLAG=0
                largest_enemy_offset1=0
                

            # 绘制图像中心点
            frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
            cv2.circle(frame, frame_center, 5, (255, 0, 0), -1)
            cv2.putText(frame, "(0, 0)", frame_center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            for offset, tag in tag_offsets:
                tag_id = tag.tag_id
                if detector.classify_target(tag_id) > 0:
                    tag_center = np.mean(tag.corners, axis=0).astype(int)
                    cv2.line(frame, frame_center, tuple(tag_center), (0, 255, 0), 2)

                cv2.putText(frame, f"ID: {tag_id}, Offset: ({offset[0]}, {offset[1]})", tuple(tag.corners[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Processed Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"发生错误: {e}")

    finally:
        camera.close()
        cv2.destroyAllWindows()  # 确保关闭所有窗口

def read_io():

    global io_data

    io_all_input = up.ADC_IO_GetAllInputLevel()
    io_array = '{:08b}'.format(io_all_input)  #:08b transfer to 8bit Binary 
    io_data.clear()
    for index, value in enumerate(io_array):       #without extra index add
        io = (int)(value)     
        io_data.append(io)  
    io_data =io_data[::-1]  #reverse the order
def io_update1():  #普通数据更新模式
    read_io()
def io_update2():  #加入线程后数据更新模式
    while(1):
        read_io()
        time.sleep(0.05)

def GetIO():      #读取ADC口数据
    global ADC_DATA    ##设定全局变量
    # global io_data
    ADC_DATA=up.ADC_Get_All_Channle()
    
def ADC_MEDIAN():  #读取中值
    global ADC_DATA
    sum=[0,0,0,0,0,0,0,0,0,0]
    total=np.zeros((30,10)) 
    for i in range(0,30, 1):
        GetIO()   
        total[i]=ADC_DATA
    sum=np.median(total,axis=0)
    return sum
def ADC_update():
    global ADC_DATE
    global Robot_State
    global bias_F
    global bias_B
    while(1):
        ADC_DATE=ADC_MEDIAN()
        if Robot_State==2:
            bias_F=300-ADC_DATE[0]
            bias_B=300-ADC_DATE[1]
        ADC_DATE[0]=ADC_DATE[0]+bias_F
        ADC_DATE[1]=ADC_DATE[1]+bias_B
        time.sleep(0.02)

def MotorSetSpeed(left,right):  #设置左右电机速度，只能输入整数
    clc1.set_motor_speed(1, -left)
    time.sleep(0.03)
    clc1.set_motor_speed(2, right)
    time.sleep(0.03)
    clc1.set_motor_speed(4, right)
    time.sleep(0.03)
    clc1.set_motor_speed(3, -left)
    time.sleep(0.03)

def MotorSpeed():

    while(1):
        MotorSetSpeed(-500,500) 


def GO_UP():
    MotorSetSpeed(1000,800)
    time.sleep(3)  
    MotorSetSpeed(0,0)

def Patrol():
    global ADC_DATE
    if(ADC_DATE[0]<ADC_DATE[1]-10):
        MotorSetSpeed(600,0)
    else:
        MotorSetSpeed(1200,1200)
    time.sleep(0.3) 

def main():
    global ADC_DATE
    global Robot_State
    global bias_F
    global bias_B
    global ENER_FLAG
    global io_data
    

    ADC_DATE=[0,0,0,0,0,0,0,0,0,0]
    io_data=[0,0,0,0,0,0,0,0,0,0]
    Robot_State=0
    bias_F=0
    bias_B=0
    ENER_FLAG=0
    
    
    
    up.__init__()
    
    
    up.ADC_IO_Open()#打开adc io
    up.CDS_Open()
    up.MPU6500_Open()
    clc1.__init__()
    
     
    t1 = threading.Thread(target=Camera_New)     
    t1.start()
    t2 = threading.Thread(target=ADC_update)     
    t2.start()
    t3 = threading.Thread(target=io_update2)     
    t3.start()
    #while(True):
    #    print('1 is {:.1f},2 is {:.1f},3 is {:.1f},4 is {:.1f},5 is {:.1f},6 is {:.1f}'.format(up.CDS_GetCurPos(1),up.CDS_GetCurPos(2),up.CDS_GetCurPos(3),up.CDS_GetCurPos(4),up.CDS_GetCurPos(5),up.CDS_GetCurPos(6)))
    while(True):
        print('the ADC1 is {:.1f},the ADC2 is {:.1f}'.format(ADC_DATE[0],ADC_DATE[1]))
        if  (io_data[2] == 0 and io_data[6] == 0 and Robot_State==0):
            Robot_State=1
            print("Ready for calibration")
            time.sleep(3)

        if  (io_data[2] == 0 and io_data[6] == 0 and Robot_State==1):
            Robot_State = 2
            print("calibration")
            time.sleep(3)
            
        if  (io_data[2] == 0 and io_data[6] == 0 and Robot_State==2):
            Robot_State = 3
            print("Done of calibration")
            time.sleep(3)
        if  (io_data[2] == 0 and io_data[6] == 0 and Robot_State==3):
            Robot_State = 4
            print("Up")
            GO_UP()
        if  (Robot_State == 4):
            Patrol()
            print("Patrol")
        if  (ENER_FLAG==1):
            Robot_State=5
            GoFor_AP()
        if  (Robot_State==5 and ENER_FLAG==0):
            Robot_State=4


        
            

if __name__ == "__main__":
    main()  # 执行主函数
####################


 