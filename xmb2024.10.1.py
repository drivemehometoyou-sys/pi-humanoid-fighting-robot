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
def cds_chushi():    #
    up.CDS_SetAngle(2,258,800) 
    time.sleep(0.4) 
    up.CDS_SetAngle(3,800,800)   
    time.sleep(0.4) 
    up.CDS_SetAngle(4,420,800)   
    time.sleep(0.4) 
    up.CDS_SetAngle(6,427,800) 
    time.sleep(0.4) 
    up.CDS_SetAngle(1,478,800)
    time.sleep(0.4) 
    up.CDS_SetAngle(5,553,800) 
    time.sleep(0.4) 
def Atack():
    up.CDS_SetAngle(1,759,800)#大抬
    up.CDS_SetAngle(5,302,800) 
    time.sleep(0.3) 
    up.CDS_SetAngle(4,836,800)   
    up.CDS_SetAngle(6,822,800) 
    time.sleep(0.3) 
    up.CDS_SetAngle(2,213,800) 
    up.CDS_SetAngle(3,840,800)    
    time.sleep(0.3) 
    up.CDS_SetAngle(1,650,800)#小抬
    up.CDS_SetAngle(5,450,800) 
    time.sleep(0.3)
    MotorSetSpeed(-900,-900)
    cds_chushi()
    MotorSetSpeed(0,0)

    
def Atack_2():

    up.CDS_SetAngle(5,200,800)#小抬
    up.CDS_SetAngle(1,800,800) 
    time.sleep(0.4) 
    up.CDS_SetAngle(4,836,800)   
    up.CDS_SetAngle(6,822,800) 
    time.sleep(0.3) 
    up.CDS_SetAngle(2,203,800) 
    up.CDS_SetAngle(3,650,800)   
    time.sleep(0.3) 
    up.CDS_SetAngle(2,400,800) #da
    up.CDS_SetAngle(3,853,800)    #xiao up.CDS_SetAngle(3,853,800)  
    time.sleep(0.4) 

    
def Q_D():
    MotorSetSpeed(0,0)
    up.CDS_SetAngle(5,200,300)#小抬
    up.CDS_SetAngle(1,800,300) 
    time.sleep(0.4)
    up.CDS_SetAngle(4,420,800)   
    up.CDS_SetAngle(6,427,800) 
    time.sleep(1.5) 
    up.CDS_SetAngle(4,836,300)#小抬
    up.CDS_SetAngle(6,822,300) 
    time.sleep(1.5) 
    up.CDS_SetAngle(1,584,300)#小抬
    up.CDS_SetAngle(5,442,300) 
    time.sleep(1.5) 
    cds_chushi()

    
def H_D():

    MotorSetSpeed(0,0)
    up.CDS_SetAngle(2,868,300)#小抬
    up.CDS_SetAngle(3,162,300) 
    time.sleep(1.5) 
    up.CDS_SetAngle(5,200,300)#小抬
    up.CDS_SetAngle(1,800,300) 
    time.sleep(0.4)
    up.CDS_SetAngle(4,420,800)   
    up.CDS_SetAngle(6,427,800) 
    time.sleep(1.5) 
    up.CDS_SetAngle(4,836,300)#小抬
    up.CDS_SetAngle(6,822,300) 
    time.sleep(1.5) 
    up.CDS_SetAngle(1,584,300)#小抬
    up.CDS_SetAngle(5,442,300) 
    time.sleep(1.5) 
    cds_chushi()
    
def GO_UP():
    MotorSetSpeed(3800,3400)
    time.sleep(0.8)  
    MotorSetSpeed(3800,1800)
    up.CDS_SetAngle(2,400,800) 
    up.CDS_SetAngle(3,650,800)  
    time.sleep(0.5)  
    MotorSetSpeed(4000,4000)
    time.sleep(1) 
def Patrol():
    global io_data
    global ADC_DATE
    global Flag
    if(io_data[0]==0 or io_data[1]==0 or io_data[2]==0 or io_data[3]==0 or io_data[4]==0 or io_data[5]==0 or io_data[6]==0 or io_data[7]==0 ):
        Flag=1
        if(io_data[0]==0):
            MotorSetSpeed(100,100)
            Atack_2()
        elif(io_data[4]==0):
            MotorSetSpeed(-900,900)
            Atack_2()
        elif(io_data[3]==0):
            MotorSetSpeed(900,-900)
            Atack_2()
        elif(io_data[5]==0):
            MotorSetSpeed(-900,900)
            Atack_2()
        elif(io_data[2]==0):
            MotorSetSpeed(750,-750)   
            Atack_2()
        elif(io_data[6]==0):
            MotorSetSpeed(-750,750)
            Atack_2()
        elif(io_data[1]==0):
            MotorSetSpeed(250,-250)
            Atack_2()
        elif(io_data[7]==0):
            MotorSetSpeed(-250,250)
            Atack_2()
    else:
        if(ADC_DATE[0]+25<ADC_DATE[1] and ADC_DATE[1]<585):
            MotorSetSpeed(0,800)
        else:
            MotorSetSpeed(2000,2000)
    time.sleep(0.3)   
def GoFor_AP():    
    global io_data
    global largest_enemy_offset
    global largest_enemy_area
    global ENER_FLAG1
    global largest_enemy_offset1
    if   (largest_enemy_offset1<-100):
        MotorSetSpeed(300,600)
    elif(largest_enemy_offset1> 100 ):
        MotorSetSpeed(600,300)
    else:
        if(io_data[0]==1 and io_data[1]==1 and io_data[7]==1):
            MotorSetSpeed(2200,2200)
        else:
            MotorSetSpeed(400,400)

        ENER_FLAG1=0

    time.sleep(0.3)
    
def _TIM_():
    global Counter
    global Counter1
    global Counter1_Flag
    global Robot_State
    while(True):
        if(Robot_State>3):

            Counter+=1
        if(Counter1_Flag==1):

            Counter1+=1
        time.sleep(0.1)
        
        
def main():
    up.__init__()
    time.sleep(1)
    up.CDS_Open()
    time.sleep(1)
    up.CDS_SetMode(1, 1)  # 设置电机 1 为电机模式
    time.sleep(1)
    up.CDS_SetMode(2, 1)  # 设置电机 2 为电机模式
    # global largest_enemy_area1
    # global ADC_DATE
    # global Robot_State
    # global bias_F
    # global bias_B
    # global ENER_FLAG
    # global ENER_FLAG1
    # global io_data
    # global Flag
    # global Counter
    # global Counter1
    # global Counter1_Flag
    # global largest_enemy_offset1
    # global AP_Counter
    # ADC_DATE=[0,0,0,0,0,0,0,0,0,0]
    # io_data=[0,0,0,0,0,0,0,0,0,0]
    # Robot_State=0
    # bias_F=0
    # bias_B=0
    # ENER_FLAG=0
    # ENER_FLAG1=0
    # Flag=0
    # Counter=0
    # Counter1=0
    # Counter1_Flag=0
    # AP_Counter=0
    # st.cds_init()
    # #电机初始化
    # clc1 = closed_loop_controller.ClosedLoopController()
    # clc1.__init__()
    # time.sleep(0.1)
    # #IO初始化
    # up=uptech.UpTech()
    # up.__init__()
    # time.sleep(0.1)
    # up.ADC_IO_Open()
    
    # #LED初始化
    # up.ADC_Led_SetColor(1,0X000000)
    # up.ADC_Led_SetColor(0,0X000000)
    # #LCD初始化
    # up.LCD_Open(2)
    # time.sleep(0.1)
    # up.ADC_IO_Open()
    # fore_color = up.COLOR_GREEN
    # back_color = up.COLOR_BLACK
    # #设置LCD字体背景颜色
    # up.LCD_SetForeColor(fore_color)
    # up.LCD_SetBackColor(back_color)
    # up.LCD_FillScreen(back_color)
    # up.LCD_SetFont(up.FONT_10X16)
    # time.sleep(0.1)
    # #光电管初始
    # cds_chushi()
    # cds_chushi()
    # time.sleep(0.5)
    # cds_chushi()
    # MotorSetSpeed(0,0)
    # print("初始化完成")
    
    
    
    # # t2 = threading.Thread(target=ADC_update)     
    # # t2.start()
    # # t4 = threading.Thread(target=io_update2)     
    # # t4.start()
    # # t6 = threading.Thread(target=Camera_New)     
    # # t6.start()
    # # t9 = threading.Thread(target=_TIM_)     
    # # t9.start()
    # time.sleep(1)
    # cds_chushi()
    # time.sleep(1)
    # up.CDS_SetMode(1,1)
    # time.sleep(1)
    # up.CDS_SetMode(2,1)
    # time.sleep(1)
    #while(True):
    #    Atack()
    #    print('1 is {:.1f}'.format(Counter))
    #    print('1 is {:.1f},2 is {:.1f}'.format(ADC_DATE[2],ADC_DATE[3]))
    #    print('1 is {:.1f},2 is {:.1f},3 is {:.1f},4 is {:.1f},5 is {:.1f},6 is {:.1f}'.format(io_data[0],io_data[1],io_data[2],io_data[3],io_data[4],io_data[5]))
    #while(True):
    #    Atack_2()
    
    #    print('1 is {:.1f},2 is {:.1f},3 is {:.1f},4 is {:.1f},5 is {:.1f},6 is {:.1f}'.format(up.CDS_GetCurPos(1),up.CDS_GetCurPos(2),up.CDS_GetCurPos(3),up.CDS_GetCurPos(4),up.CDS_GetCurPos(5),up.CDS_GetCurPos(6)))
    while True:
        print("运行中")
        up.CDS_SetSpeed(1, 500)  # 设置电机1速度为500
        time.sleep(0.3)
        up.CDS_SetSpeed(2, 500)  # 如果你也想让电机2转动
        time.sleep(10)
        # #print('the ADC1 is {:.1f},the ADC2 is {:.1f}'.format(ADC_DATE[0],ADC_DATE[1]))
        # if(Robot_State>3 and Counter>30 and (ADC_DATE[2]<1000 or ADC_DATE[3]<1000)):
        #     Robot_State=6
        #     time.sleep(0.1)
        #     if(ADC_DATE[2]<1000):
        #         Q_D()
        #     elif(ADC_DATE[3]<1000):
        #         H_D()
        # if(Robot_State==6 and Counter>30 and (ADC_DATE[2]>1000 and ADC_DATE[3]>1000)):
        #     Robot_State=4

                
        # if  (Robot_State==5 and ENER_FLAG==0 and Counter>30):
        #     MotorSetSpeed(0,0)
        #     time.sleep(0.3)
        #     if(ENER_FLAG==0):
        #         print("l_AP")
        #         print('1 is {:.1f}'.format(Counter))

        #         if(ENER_FLAG==0):
        #             AP_Counter=0
        #             Atack()
        #             MotorSetSpeed(-1400,-1400)
        #             cds_chushi()
        #             up.CDS_SetAngle(1,478,800)
        #             up.CDS_SetAngle(5,553,800) 
        #             time.sleep(0.3) 
        #             MotorSetSpeed(0,0)
        #             time.sleep(0.3)
        #         if(ENER_FLAG==0):
        #             Robot_State=4
        # if(Robot_State<=3):
        #     print('1 is {:.1f},2 is {:.1f}'.format(ADC_DATE[0],ADC_DATE[1]))
        # if  (io_data[2] == 0 and io_data[6] == 0 and Robot_State==0):
        #     Robot_State=1
        #     print("Ready for calibration")
        #     time.sleep(1)

        # if  (io_data[2] == 0 and io_data[6] == 0 and Robot_State==1):
        #     Robot_State = 2
        #     print("calibration")
        #     time.sleep(1)
            
        # if  (io_data[2] == 0 and io_data[6] == 0 and Robot_State==2):
        #     Robot_State = 3
        #     print("Done of calibration")
        #     time.sleep(1)
        # if  (io_data[2] == 0 and io_data[6] == 0 and Robot_State==3):
        #     Robot_State = 4
        #     print("Up")
        #     GO_UP()
        # if  (ENER_FLAG==1 and Robot_State > 3 and largest_enemy_area1>190):
        #     Robot_State=5
        #     if(Flag==1):
        #         MotorSetSpeed(0,0)
        #         cds_chushi()
        #         Flag=0
        #     GoFor_AP()
        # elif  (ENER_FLAG==1 and Robot_State > 3 and largest_enemy_area1<190):
        #     Robot_State=4
        # if  (Robot_State == 4):
        #     print('1 is {:.1f},2 is {:.1f}'.format(ADC_DATE[0],ADC_DATE[1]))
        #     Patrol()
        #     print("Patrol")


        
            

if __name__ == "__main__":
    main()  # 执行主函数
####################