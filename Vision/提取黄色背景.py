import cv2
import numpy as np

# 打开摄像头
cap = cv2.VideoCapture(0)
dst_points = np.array([[0, 0], [300, 0], [0, 200], [300, 200]], dtype="float32")

# 定义黄色的HSV范围
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

def nothing(x):
    pass

# 创建窗口和滑动条
cv2.namedWindow("Trackbars")
cv2.createTrackbar("Canny Thresh1", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("Canny Thresh2", "Trackbars", 150, 255, nothing)

def get_center(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    return cX, cY

def threshold_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    return mask_yellow

def find_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # 获取近似多边形
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        if len(approx) == 4:
            src_points = np.array([point[0] for point in approx], dtype="float32")
            print("Corner Points:", src_points)
            return src_points
        else:
            print("Largest contour is not a quadrilateral.")
    return None

def process_frame(frame):
    mask_yellow = threshold_frame(frame)
    src_points = find_largest_contour(mask_yellow)

    if src_points is not None:
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(frame, M, (300, 200))

        canny_thresh1 = cv2.getTrackbarPos("Canny Thresh1", "Trackbars")
        canny_thresh2 = cv2.getTrackbarPos("Canny Thresh2", "Trackbars")

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)

        # 膨胀操作
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        cv2.imshow("edges", edges)

        black_centers = []
        white_centers = []

        mask_black = cv2.inRange(warped, np.array([0, 0, 0]), np.array([179, 146, 147]))
        mask_white = cv2.inRange(warped, np.array([0, 0, 173]), np.array([179, 62, 255]))

        black_contours, _ = cv2.findContours(mask_black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        white_contours, _ = cv2.findContours(mask_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in black_contours:
            if cv2.contourArea(contour) > 150 and cv2.contourArea(contour) < 300:  # 过滤掉小噪声
                perimeter = cv2.arcLength(contour, True)
                area = cv2.contourArea(contour)
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if 0.7 < circularity < 1.3:  # 判断是否为圆形
                    cX, cY = get_center(contour)
                    black_centers.append((cX, cY))
                    cv2.circle(warped, (cX, cY), 5, (0, 0, 0), -1)
                    cv2.putText(warped, f'B({cX},{cY})', (cX - 20, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        for contour in white_contours:
            if cv2.contourArea(contour) > 200 and cv2.contourArea(contour) < 300:  # 过滤掉小噪声
                perimeter = cv2.arcLength(contour, True)
                area = cv2.contourArea(contour)
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if 0.7 < circularity < 1.3:  # 判断是否为圆形
                    cX, cY = get_center(contour)
                    white_centers.append((cX, cY))
                    cv2.circle(warped, (cX, cY), 5, (255, 255, 255), -1)
                    cv2.putText(warped, f'W({cX},{cY})', (cX - 20, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imshow("Warped Frame", warped)
        cv2.imshow("Yellow Mask", mask_yellow)
        cv2.imshow("Black Mask", mask_black)
        cv2.imshow("White Mask", mask_white)

        return black_centers, white_centers, src_points

    return [], [], None

while True:
    ret, frame = cap.read()

    if not ret:
        break

    black_centers, white_centers, src_points = process_frame(frame)

    if src_points is not None:
        for point in src_points:
            point = tuple(map(int, point))  # 将坐标转换为整数元组
            cv2.circle(frame, point, 5, (0, 0, 255), -1)
        cv2.imshow("Original Frame with Corners", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
