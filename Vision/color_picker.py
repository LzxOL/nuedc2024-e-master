import cv2
import numpy as np
# import serial
import time

# 配置串口参数
# port = 'COM3'  # 根据实际情况修改
# baud_rate = 9600  # 波特率

# 初始化串口
# ser = serial.Serial(port, baud_rate, timeout=1)





def nothing(x):
    pass

# 创建一个窗口
cv2.namedWindow('image')

# 创建六个滑动条，分别代表HSV颜色空间的最小和最大值
cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

# 初始化滑动条的最大值
cv2.setTrackbarPos('HMax', 'image', 179)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 255)

# 打开摄像头
cap = cv2.VideoCapture(0)
# 确保串口打开
# if ser.is_open:
#     print(f"串口 {port} 已打开，波特率为 {baud_rate}")
# else:
#     print(f"无法打开串口 {port}")

while True:
    # 读取一帧图像
    ret, frame = cap.read()
    if not ret:
        break
    # frame = cv2.erode(frame, (3, 3), 0)
    # 将图像转换到HSV颜    色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #
    # 获取滑动条的当前位置
    h_min = cv2.getTrackbarPos('HMin', 'image')
    h_max = cv2.getTrackbarPos('HMax', 'image')
    s_min = cv2.getTrackbarPos('SMin', 'image')
    s_max = cv2.getTrackbarPos('SMax', 'image')
    v_min = cv2.getTrackbarPos('VMin', 'image')
    v_max = cv2.getTrackbarPos('VMax', 'image')

    # 根据滑动条的位置调整HSV范围
    lower_hsv = np.array([h_min, s_min, v_min])
    upper_hsv = np.array([h_max, s_max, v_max])

    # 获取特定颜色范围内的图像
    hsv_black = hsv.copy()
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)


    result = cv2.bitwise_and(frame, frame, mask=mask)

    # 显示结果
    cv2.imshow('image', mask)

    # 发送标志值1
    # flag = 1
    # ser.write(flag.to_bytes(1, byteorder='big'))  # 将整数转换为单个字节并发送
    #
    # print('send',flag)
    # 检测按键事件，按下ESC退出
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 释放摄像头并销毁所有窗口
cap.release()
cv2.destroyAllWindows()
