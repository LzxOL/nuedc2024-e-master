import cv2


# 鼠标回调函数
def show_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse coordinates: ({x}, {y})")
        # 创建一个副本以在上面显示坐标
        frame_copy = frame.copy()
        cv2.putText(frame_copy, f"({x}, {y})", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Video", frame_copy)


# 打开视频流（默认是摄像头）
cap = cv2.VideoCapture(0)

# 创建一个窗口
cv2.namedWindow("Video")

# 设置鼠标回调函数
cv2.setMouseCallback("Video", show_coordinates)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 显示当前帧
    cv2.imshow("Video", frame)

    # 退出条件
    if cv2.waitKey(1) & 0xFF == 27:  # 按ESC键退出
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
