import cv2
import numpy as np
import threading
import apriltag
import uptech
import time
import servo_test as st
import closed_loop_controller
# from simple_pid import PID

up = uptech.UpTech()
clc1 = closed_loop_controller.ClosedLoopController()

# openCV在这里面的任务：占用硬件（VideoCapture） -> 设置长宽（set） -> 咔嚓拍照（read） -> 释放硬件（release）
class Camera:
    def __init__(self, camera_index):
        self.frame = None           # 用来存放拍下来的“照片”（图像矩阵）
        self.ret = False            # 照片有没有拍成功的标志（布尔值）
        self.camera_index = camera_index # 摄像头的硬件编号（通常是 0）
        self.open_flag = False      # 摄影师是否在上班的标志位
        self.cap = None             # 摄像头的实体对象（也就是手里的照相机）

    def open(self):
        """打开摄像头"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)  # 1：获取硬件控制权
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open camera")

            self.cap.set(3, 640) # 2：设置分辨率参数
            self.cap.set(4, 320)
            self.open_flag = True
            threading.Thread(target=self.query_frame, args=()).start() # 开辟独立线程读取画面

        except Exception as e:
            print(f"摄像头打开失败: {e}")

    def query_frame(self):
        """不断读取摄像头帧"""
        # 死循环中疯狂读取硬件！
        # ret: 是否成功读到 (True/False)
        # frame: 真正的图片三维矩阵 (B, G, R)
        while self.open_flag:
            self.ret, self.frame = self.cap.read()   # 3：按下快门   frame:Numpy 三维数字矩阵：320行 × 640列 × 3个颜色通道(BGR)

    def get_frame(self): # 与主脑的接口
        """返回读取状态和当前帧"""
        return self.ret, self.frame

    def close(self):
        """关闭摄像头"""
        self.open_flag = False
        if self.cap:
            self.cap.release()  # 4：释放硬件归还系统


class AprilTagDetector:
    def __init__(self):
        self.target_id = 10  # 目标ID初始化
        self.at_detector = apriltag.Detector(apriltag.DetectorOptions(families='tag36h11 tag25h9')) # 创建 AprilTag 探测器对象
        self.enemy_tag = 1   # 敌方二维码ID
        self.friend_tag = 0  # 我方二维码ID

    def detect_frame(self, frame):
        """检测当前帧中的二维码"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 把三维彩色图，变成二维灰度图
        tags = self.at_detector.detect(gray) # 让引擎去灰度图里找标签
        if tags:
            # tag 对象
            # tag.tag_id：这个标签的 ID 号码。
            # tag.corners：一个包含 4 个坐标点的小矩阵，代表这个标签在画面里的左下、右下、右上、左上四个角点的准确像素坐标。
            # tag.center：标签中心点的准确像素坐标。
            for tag in tags:
                self.draw_bounding_box(frame, tag)  # 为所有二维码打框 (在原彩色图上画绿色方框)

            # 返回一个列表，里面打包了：(这颗标签的坐标偏移量, 标签本体信息)
            return [(self.calculate_offset(frame.shape, tag), tag) for tag in tags] 
        return [] # 如果什么都没找到，返回空列表

    def get_apriltag_area(self, tag):
        """计算二维码的面积"""
        corners = tag.corners.astype(np.float32)
        return cv2.contourArea(corners)

    def calculate_offset(self, frame_shape, tag):
        """计算二维码中心与图像中心的偏移量"""
        frame_center = np.array([frame_shape[1] // 2, frame_shape[0] // 2])  # 画面中心：宽度的一半，高度的一半
        tag_center = tag.center.astype(int)
        offset = tag_center - frame_center        
        return int(offset[0]), int(offset[1])

    def classify_target(self, tag_id):
        """判断目标是敌方还是友方"""
        if tag_id == self.enemy_tag:
            return 1  # 敌方
        elif tag_id == self.friend_tag:
            return 2  # 友方
        return 0  # 未识别

    def draw_bounding_box(self, frame, tag):
        """在图像上框出二维码"""
        # 1. 提取出 4 个角点坐标，并确保它们是整数（画图不能在半个像素上画）
        corners = tag.corners.astype(int)
        # 2. 调用 OpenCV 的画多边形函数
        cv2.polylines(frame, [corners], isClosed=True, color=(0, 255, 0), thickness=2)

    def get_largest_enemy_tag_info(self, tags):
        """获取面积最大的敌方二维码的偏移量和面积"""
        largest_enemy_tag = None
        max_area = 0
        largest_offset = (0, 0)

        for offset, tag in tags:
            if self.classify_target(tag.tag_id) == 1:  # 判断是否为敌方二维码
                area = self.get_apriltag_area(tag)     # 计算面积
                if area > max_area:
                    max_area = area
                    largest_enemy_tag = tag
                    largest_offset = offset

        if largest_enemy_tag:
            return largest_offset, max_area  # 返回偏移量和面积
        else:
            return None, 0  # 如果没有找到敌方二维码，则返回None



### 这个线程输出 全局变量 enemy_flag 和 current_enemy_offset和 current_enemy_area

# 在函数外面/开头加一个全局变量记录丢失的帧数
lost_count = 0  
MAX_LOST_TOLERANCE = 5  # 容忍度：允许连续丢 5 帧（约 0.2 秒）不放弃

def camera_task():
    global enemy_flag
    global current_enemy_offset
    global current_enemy_area
    global lost_count

    camera_index = 0
    camera = Camera(camera_index)  # 拿起相机
    detector = AprilTagDetector()  # 开启识别引擎
    current_enemy_offset = 0
    current_enemy_area = 0
    
    try:
        camera.open()
        threading.Event().wait(1)
        while True:
            ret, frame = camera.get_frame()
            if not ret:
                print("未能读取帧")
                break

            tag_offsets = detector.detect_frame(frame)

            # 获取面积最大的敌方二维码的偏移量和面积
            largest_offset, max_area = detector.get_largest_enemy_tag_info(tag_offsets)

            if largest_offset is not None:
                lost_count = 0  # 只要看到一眼，丢失计数器立刻清零，回满血！
                enemy_flag = 1
                current_enemy_offset = largest_offset[0]
                current_enemy_area = max_area
                print(f"最大敌方二维码偏移量: {largest_offset}, 面积: {max_area}")
            else:
                lost_count += 1 # 计数器 +1，开始读秒
                if lost_count > MAX_LOST_TOLERANCE:
                # 只有连续 5 帧都没看到，才是真丢了！
                    enemy_flag = 0 
                    current_enemy_offset = 0
                else:   
                    pass
            # 实战时注释
            # # 绘制图像中心点
            # frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
            # cv2.circle(frame, frame_center, 5, (255, 0, 0), -1) # 画一个实心圆 (-1代表实心)：中心点，半径5，颜色(255, 0, 0)纯蓝
            # cv2.putText(frame, "(0, 0)", frame_center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) # 在旁边写上文字 "(0, 0)"

            # for offset, tag in tag_offsets:
            #     tag_id = tag.tag_id
            #     if detector.classify_target(tag_id) > 0:
            #         tag_center = tag.center.astype(int) # 计算标签中心
            #         cv2.line(frame, frame_center, tuple(tag_center), (0, 255, 0), 2)  # 从画面中心，画一条绿色的线，连到标签中心

            #     # 把标签的 ID 和 偏移量数字，用黄色 (0, 255, 255) 写在标签旁边
            #     cv2.putText(frame, f"ID: {tag_id}, Offset: ({offset[0]}, {offset[1]})", 
            #                 tuple(tag.corners[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Processed Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        camera.close()
        cv2.destroyAllWindows()

######################################################################################################
def read_io():
    global io_data
    io_all_input = up.ADC_IO_GetAllInputLevel()
    io_array = '{:08b}'.format(io_all_input)  # :08b 转换为8位二进制
    io_data.clear()
    for index, value in enumerate(io_array):
        io = int(value)    
        io_data.append(io)  
    io_data = io_data[::-1]  # 反转顺序


def io_update_normal():  # 普通数据更新模式
    read_io()


def io_update_task():  # 加入线程后数据更新模式
    while True:
        read_io()
        time.sleep(0.05)
######################################################################################################

def get_adc_data():      # 读取ADC口数据
    global adc_data    
    adc_data = up.ADC_Get_All_Channle()


def get_adc_median():  # 读取中值滤波数据
    global adc_data
    total = np.zeros((30, 10)) 
    for i in range(0, 30, 1):
        get_adc_data()   
        total[i] = adc_data
    sum_median = np.median(total, axis=0)  #按列求平均
    return sum_median


def adc_update_task():
    global adc_data
    global robot_state
    global bias_front
    global bias_rear
    while True:
        adc_data = get_adc_median()
        if robot_state == 2:
            bias_front = 300 - adc_data[0]
            bias_rear = 300 - adc_data[1]
        adc_data[0] = adc_data[0] + bias_front
        adc_data[1] = adc_data[1] + bias_rear
        time.sleep(0.02)
######################################################################################################### 

def set_motor_speed(left, right):  # 设置左右底盘电机速度
    clc1.set_motor_speed(1, -left)
    time.sleep(0.03)
    clc1.set_motor_speed(2, right)
    time.sleep(0.03)
    clc1.set_motor_speed(4, right)
    time.sleep(0.03)
    clc1.set_motor_speed(3, -left)
    time.sleep(0.03)   


def test_motor_speed():
    while True:
        set_motor_speed(-500, 500)
#########################################################################################################

def reset_servos():    # 舵机/机械臂初始化角度位置
    up.CDS_SetAngle(2, 258, 800) 
    time.sleep(0.4) 
    up.CDS_SetAngle(3, 800, 800)   
    time.sleep(0.4) 
    up.CDS_SetAngle(4, 420, 800)   
    time.sleep(0.4) 
    up.CDS_SetAngle(6, 427, 800) 
    time.sleep(0.4) 
    up.CDS_SetAngle(1, 478, 800)
    time.sleep(0.4) 
    up.CDS_SetAngle(5, 553, 800) 
    time.sleep(0.4) 


def attack():
    up.CDS_SetAngle(1, 759, 800)  # 大抬
    up.CDS_SetAngle(5, 302, 800) 
    time.sleep(0.3) 
    up.CDS_SetAngle(4, 836, 800)   
    up.CDS_SetAngle(6, 822, 800) 
    time.sleep(0.3) 
    up.CDS_SetAngle(2, 213, 800) 
    up.CDS_SetAngle(3, 840, 800)    
    time.sleep(0.3) 
    up.CDS_SetAngle(1, 650, 800)  # 小抬
    up.CDS_SetAngle(5, 450, 800) 
    time.sleep(0.3)
    
    set_motor_speed(-900, -900)
    reset_servos()
    set_motor_speed(0, 0)


def attack_light():
    up.CDS_SetAngle(5, 200, 800)  # 小抬
    up.CDS_SetAngle(1, 800, 800) 
    time.sleep(0.4) 
    up.CDS_SetAngle(4, 836, 800)   
    up.CDS_SetAngle(6, 822, 800) 
    time.sleep(0.3) 
    up.CDS_SetAngle(2, 203, 800) 
    up.CDS_SetAngle(3, 650, 800)   
    time.sleep(0.3) 
    up.CDS_SetAngle(2, 400, 800)  # 大
    up.CDS_SetAngle(3, 853, 800)  # 小 
    time.sleep(0.4) 


def front_avoidance():
    set_motor_speed(0, 0)
    
    up.CDS_SetAngle(5, 200, 300)  # 小抬
    up.CDS_SetAngle(1, 800, 300) 
    time.sleep(0.4)
    up.CDS_SetAngle(4, 420, 800)   
    up.CDS_SetAngle(6, 427, 800) 
    time.sleep(1.5) 
    up.CDS_SetAngle(4, 836, 300)  # 小抬
    up.CDS_SetAngle(6, 822, 300) 
    time.sleep(1.5) 
    up.CDS_SetAngle(1, 584, 300)  # 小抬
    up.CDS_SetAngle(5, 442, 300) 
    time.sleep(1.5) 
    reset_servos()


def rear_avoidance():
    set_motor_speed(0, 0)
    
    up.CDS_SetAngle(2, 868, 300)  # 小抬
    up.CDS_SetAngle(3, 162, 300) 
    time.sleep(1.5) 
    up.CDS_SetAngle(5, 200, 300)  # 小抬
    up.CDS_SetAngle(1, 800, 300) 
    time.sleep(0.4)
    up.CDS_SetAngle(4, 420, 800)   
    up.CDS_SetAngle(6, 427, 800) 
    time.sleep(1.5) 
    up.CDS_SetAngle(4, 836, 300)  # 小抬
    up.CDS_SetAngle(6, 822, 300) 
    time.sleep(1.5) 
    up.CDS_SetAngle(1, 584, 300)  # 小抬
    up.CDS_SetAngle(5, 442, 300) 
    time.sleep(1.5) 
    reset_servos()
#########################################################################################################

def go_up_ramp():
    set_motor_speed(3800, 3400)
    time.sleep(0.8)  
    set_motor_speed(3800, 1800)
    up.CDS_SetAngle(2, 400, 800) 
    up.CDS_SetAngle(3, 650, 800)  
    time.sleep(0.5)  
    set_motor_speed(4000, 4000)
    time.sleep(1) 


def patrol():
    global io_data
    global adc_data
    global obstacle_flag
    
    if 0 in io_data[:8]:
        obstacle_flag = 1
        if io_data[0] == 0:
            set_motor_speed(100, 100)
            attack_light()
        elif io_data[4] == 0:
            set_motor_speed(-900, 900)
            attack_light()
        elif io_data[3] == 0:
            set_motor_speed(900, -900)
            attack_light()
        elif io_data[5] == 0:
            set_motor_speed(-900, 900)
            attack_light()
        elif io_data[2] == 0:
            set_motor_speed(750, -750)   
            attack_light()
        elif io_data[6] == 0:
            set_motor_speed(-750, 750)
            attack_light()
        elif io_data[1] == 0:
            set_motor_speed(250, -250)
            attack_light()
        elif io_data[7] == 0:
            set_motor_speed(-250, 250)
            attack_light()
    else:
        if adc_data[0] + 25 < adc_data[1] and adc_data[1] < 585:
            set_motor_speed(0, 800)
        else:
            set_motor_speed(2000, 2000)
    time.sleep(0.3)   


def track_apriltag():    
    global io_data
    global current_enemy_offset
    global current_enemy_area
    global enemy_flag_track
    
    if current_enemy_offset < -100:
        set_motor_speed(300, 600)
    elif current_enemy_offset > 100:
        set_motor_speed(600, 300)
    else:
        if io_data[0] == 1 and io_data[1] == 1 and io_data[7] == 1:
            set_motor_speed(2200, 2200)
        else:
            set_motor_speed(400, 400)
        enemy_flag_track = 0
    time.sleep(0.3)
    

def timer_task():
    global main_counter
    global counter_1
    global counter_1_flag
    global robot_state
    
    while True:
        if robot_state > 3:
            main_counter += 1
        if counter_1_flag == 1:
            counter_1 += 1
        time.sleep(0.1)
        

def main():
    ENEMY_ATTACK_AREA = 1000
    
    global current_enemy_area
    global adc_data
    global robot_state
    global bias_front
    global bias_rear
    global enemy_flag
    global enemy_flag_track
    global io_data
    global obstacle_flag
    global main_counter
    global counter_1
    global counter_1_flag
    global current_enemy_offset
    global apriltag_counter
    global up
    global clc1

    # 变量与传感器状态初始化
    adc_data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    io_data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    robot_state = 0
    bias_front = 0
    bias_rear = 0
    enemy_flag = 0
    enemy_flag_track = 0
    obstacle_flag = 0
    main_counter = 0
    counter_1 = 0
    counter_1_flag = 0
    apriltag_counter = 0
    current_enemy_offset = 0
    current_enemy_area = 0
    
    # 舵机与电机闭环驱动初始化
    st.cds_init()
    clc1 = closed_loop_controller.ClosedLoopController()
    clc1.__init__()
    time.sleep(0.1)
    
    # 主板总线与 I/O 模块初始化
    up = uptech.UpTech()
    up.__init__()
    time.sleep(0.1)
    up.ADC_IO_Open()
    
    # LED状态灯初始化
    up.ADC_Led_SetColor(1, 0X000000)
    up.ADC_Led_SetColor(0, 0X000000)
    
    # LCD车载显示屏面板初始化
    up.LCD_Open(2)
    time.sleep(0.1)
    up.ADC_IO_Open()
    fore_color = up.COLOR_GREEN
    back_color = up.COLOR_BLACK
    up.LCD_SetForeColor(fore_color)
    up.LCD_SetBackColor(back_color)
    up.LCD_FillScreen(back_color)
    up.LCD_SetFont(up.FONT_10X16)
    time.sleep(0.1)
    
    # 动作组初始归位
    reset_servos()
    reset_servos()
    time.sleep(0.5)
    reset_servos()
    set_motor_speed(0, 0)
    print("初始化完成")
    
    # 启动异步高频后台任务线程
    t2 = threading.Thread(target=adc_update_task)     
    t2.start()
    t4 = threading.Thread(target=io_update_task)     
    t4.start()
    t6 = threading.Thread(target=camera_task)     
    t6.start()
    t9 = threading.Thread(target=timer_task)     
    t9.start()
    time.sleep(1)
    
    # 配置底盘总线电机的运行模式
    up.CDS_Open()
    time.sleep(0.1)
    up.CDS_SetMode(1, 1)  # 设置电机 1 为连续旋转电机模式
    time.sleep(0.1)
    up.CDS_SetMode(2, 1)  # 设置电机 2 为连续旋转电机模式
    time.sleep(0.1)

    # 核心有限状态机控制闭环
    while True:
        # 【安全与边界限制机制】
        if robot_state > 3 and main_counter > 30 and (adc_data[2] < 1000 or adc_data[3] < 1000):
            robot_state = 6
            time.sleep(0.1)
            if adc_data[2] < 1000:
                front_avoidance()  
            elif adc_data[3] < 1000:
                rear_avoidance()  
                
        # 避障动作结束后，若环境恢复安全，重置回常规巡航状态 4
        if robot_state == 6 and main_counter > 30 and (adc_data[2] > 1000 and adc_data[3] > 1000):
            robot_state = 4

        # 【锁定与丢失目标处理机制】
        if robot_state == 5 and enemy_flag == 0 and main_counter > 30:
            set_motor_speed(0, 0)
            time.sleep(0.3)
            if enemy_flag == 0:
                print("Lost AprilTag - Engaging Blind Attack")
                print(f"Counter is {main_counter:.1f}")
                if enemy_flag == 0:
                    apriltag_counter = 0
                    attack()  
                    set_motor_speed(-1400, -1400)
                    reset_servos()
                    up.CDS_SetAngle(1, 478, 800)
                    up.CDS_SetAngle(5, 553, 800) 
                    time.sleep(0.3) 
                    set_motor_speed(0, 0)
                    time.sleep(0.3)
                if enemy_flag == 0:
                    robot_state = 4  
                    
        # 【上场自检与标定状态机（状态0-3）】
        if robot_state <= 3:
            print(f"ADC_0 is {adc_data[0]:.1f}, ADC_1 is {adc_data[1]:.1f}")
            
        # 依据特定路面或按键 IO 信号，步进标定流程
        if io_data[2] == 0 and io_data[6] == 0 and robot_state == 0:
            robot_state = 1
            print("Ready for calibration")
            time.sleep(1)

        if io_data[2] == 0 and io_data[6] == 0 and robot_state == 1:
            robot_state = 2
            print("Calibration in progress")
            time.sleep(1)
            
        if io_data[2] == 0 and io_data[6] == 0 and robot_state == 2:
            robot_state = 3
            print("Calibration done")
            time.sleep(1)
            
        # 标定完全通过，触发 go_up_ramp 正式切入战局状态 4
        if io_data[2] == 0 and io_data[6] == 0 and robot_state == 3:
            robot_state = 4
            print("Up and Start")
            go_up_ramp()
            
        # 【视觉目标追踪条件判定】
        if enemy_flag == 1 and robot_state > 3 and current_enemy_area > ENEMY_ATTACK_AREA:
            robot_state = 5  
            if obstacle_flag == 1:
                set_motor_speed(0, 0)
                reset_servos()
                obstacle_flag = 0
            track_apriltag()  
        elif enemy_flag == 1 and robot_state > 3 and current_enemy_area < ENEMY_ATTACK_AREA:
            robot_state = 4  

        # 【状态4：常规路线寻迹巡航】
        if robot_state == 4:
            print(f"ADC_0 is {adc_data[0]:.1f}, ADC_1 is {adc_data[1]:.1f}")
            patrol()  
            print("Patrol Running")
            
        time.sleep(0.05)  


if __name__ == "__main__":
    main()