import cv2
import numpy as np
import time
import random
import serial

ser = serial.Serial(
        port='COM9',          # 更改为你的串口端口
        baudrate=115200,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1
    )


cX_dingwei, cY_dingwei = 0, 0
# 打开摄像头
cap = cv2.VideoCapture(0)

grid_coordinates_update_flag = 0#棋盘坐标标定flag
chosed_black = 0
count_flag = 0
y_count_flag = 0#抓棋子的x标志位
catch_qizi_flag = 0 #抓棋子的标志位

place_where_num = 0 #下位机发来的选择哪个格子的标志位
biaoding_flag = 0 #
# 第四个任务
fourth_task_plack = 0#第四个任务接收下位机的放置位置的标志位
bisheng_flag = 0

light_up_flag = 0
xiaqi_flag = 0#下位机发来的下棋标志位

judge_mid_or_bian = 0


target_num = 0 #任务一的标志位


# 各种列表 ——————————————————————————————————————————————————————
# 将棋盘中的九宫格各个宫格存在 1*9 的列表中
grid_coordinates = np.zeros((1, 9, 2), dtype=int)
# 直接定义 grid_coordinates


# 定义黄色的HSV范围
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# 棋盘部分霍夫圆的坐标列表
black_centers_mid = []
white_centers_mid = []
previous_white_centers_mid = []

# 棋盘两边棋子的坐标列表


# 棋盘中已经放好了的棋子的坐标

# 初始化棋盘状态，0表示空位，1表示黑棋，2表示白棋
board = np.zeros((3, 3), dtype=int)
previous_board = np.zeros((3, 3), dtype=int)  # 用于保存之前的棋盘状态
board_qizi = np.zeros((3, 3), dtype=int)



# ——————————————————————————————————————————————————————————————————



# 四个点定义的坐标
src_points = np.array([[193, 98], [495, 103], [190, 290], [494, 293]], dtype="float32")
# 目标点，定义变换后图像的大小
dst_points = np.array([[0, 0], [300, 0], [0, 200], [300, 200]], dtype="float32")


# def get_perspective_point()
def find_background_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # 获取近似多边形
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        if len(approx) == 4:
            src_points = np.array([point[0] for point in approx], dtype="float32")
            # print("Corner Points:", src_points)
            return src_points
        else:
            print("Largest contour is not a quadrilateral.")
    return None

def biaoding(edges):
    print("once")
    centers = []
    squares = find_squares(edges)
    for square in squares:
        cX, cY = get_square_center(square)
        if 100 < cX < 200:
            centers.append((cX, cY))

            cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
            cv2.drawContours(frame, [square], 0, (0, 255, 0), 2)

    # 按 y 值分组
    centers_sorted_by_y = sorted(centers, key=lambda p: p[1])
    y_groups = []

    # 创建三组
    group_size = len(centers) // 3
    for i in range(0, len(centers_sorted_by_y), group_size):
        y_groups.append(centers_sorted_by_y[i:i + group_size])

    # 对每组内的坐标按 x 值排序
    sorted_groups = [sorted(group, key=lambda p: p[0]) for group in y_groups]

    # 合并所有排序后的组
    final_sorted_centers = [item for group in sorted_groups for item in group]

    # 将列表转换为 NumPy 数组，并调整形状为 (1, 9, 2)
    grid_coordinates = np.array(final_sorted_centers).reshape((1, 9, 2))

    print("grid_coordinates:")
    print(grid_coordinates)
    return grid_coordinates

# 黑白棋子的阈值
def threshold_frame(frame):
    # 设置黑色和白色的阈值
    black_lower = np.array([0, 65, 0])
    black_upper = np.array([179, 123, 83])


    white_lower = np.array([0, 0, 173])
    white_upper = np.array([179, 62, 255])

    DINGWEI_lower = np.array([0, 144, 128])
    DINGWEI_upper = np.array([22, 255, 255])



    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 对图像进行阈值处理
    mask_black = cv2.inRange(hsv, black_lower, black_upper)
    mask_white = cv2.inRange(hsv, white_lower, white_upper)
    mask_dingwei = cv2.inRange(hsv, DINGWEI_lower, DINGWEI_upper)

    # 仅保留感兴趣区域
    # mask_black = cv2.bitwise_and(mask_black, mask_black, mask=mask)
    # mask_white = cv2.bitwise_and(mask_white, mask_white, mask=mask)

    return mask_black, mask_white, mask_dingwei

def find_squares(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    grid_coordinates = np.zeros((3, 3, 2), dtype=int)  # 用于存储九宫格坐标
    for contour in contours:
        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:  # 只考虑四边形
            area = cv2.contourArea(approx)
            if 200 < area < 5000:  # 过滤掉太小或太大的轮廓
                squares.append(approx)
    return squares




def get_center(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    return cX, cY

def get_square_center(square):
    M = cv2.moments(square)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    return cX, cY

def create_roi_mask(frame_shape, x_ranges):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    for x_range in x_ranges:
        mask[:, x_range[0]:x_range[1]] = 255
    return mask

def sort_and_store_coordinates(centers):
    return sorted(centers, key=lambda coord: coord[1])

def remove_duplicates_and_limit_centers(centers, threshold=5, limit=5):
    unique_centers = []
    for center in centers:
        if all(np.linalg.norm(np.array(center) - np.array(unique_center)) > threshold for unique_center in unique_centers):
            unique_centers.append(center)
        if len(unique_centers) == limit:
            break
    return unique_centers
def img_process(frame,grid_coordinates):
    global cX_dingwei, cY_dingwei


    # grid_coordinates = biaoding(edges)
    # print(grid_coordinates)
    # squares = find_squares(edges)
    frame_qizi = frame.copy()
    frame_qizi = cv2.GaussianBlur(frame_qizi, (5, 5), 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 对于黑白棋子的阈值提取
    mask_black, mask_white, mask_dingwei = threshold_frame(frame_qizi)
    # 自适应阈值处理
    gray_black = cv2.bitwise_and(gray, gray, mask=mask_black)
    gray_white = cv2.bitwise_and(gray, gray, mask=mask_white)
    adaptive_thresh_black = cv2.adaptiveThreshold(gray_black, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY_INV, 11, 2)
    adaptive_thresh_white = cv2.adaptiveThreshold(gray_white, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY_INV, 11, 2)
    # print('1',adaptive_thresh_white)
    # 形态学操作
    kernel = np.ones((3, 3), np.uint8)
    adaptive_thresh_black = cv2.morphologyEx(adaptive_thresh_black, cv2.MORPH_CLOSE, kernel)
    adaptive_thresh_white = cv2.morphologyEx(adaptive_thresh_white, cv2.MORPH_CLOSE, kernel)

    # 查找黑色和白色棋子的中心
    # 找到黑色和白色棋子的轮廓
    black_contours, _ = cv2.findContours(adaptive_thresh_black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    white_contours, _ = cv2.findContours(adaptive_thresh_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    dingwei_contours, _ = cv2.findContours(mask_dingwei, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    black_centers = []
    white_centers = []
    #对两边的棋子进行标定---------------------------------------------------------
    for contour in black_contours:
        if cv2.contourArea(contour) > 130 and cv2.contourArea(contour) < 300:  # 过滤掉小噪声
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if 0.7 < circularity < 1.3:  # 判断是否为圆形
                cX, cY = get_center(contour)
                black_centers.append((cX, cY))
                cv2.circle(frame, (cX, cY), 5, (0, 0, 0), -1)
                cv2.putText(frame, f'B({cX},{cY})', (cX - 20, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                            1)

    for contour in white_contours:
        if cv2.contourArea(contour) > 180 and cv2.contourArea(contour) < 300:  # 过滤掉小噪声
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if 0.7 < circularity < 1.3:  # 判断是否为圆形
                cX, cY = get_center(contour)
                white_centers.append((cX, cY))
                cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
                cv2.putText(frame, f'W({cX},{cY})', (cX - 20, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    #--------------------------------------------------------------------------------

    # 取最大轮廓
    if len(dingwei_contours) > 0:
        max_contour = max(dingwei_contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 150:  # 过滤掉小噪声
            cX_dingwei, cY_dingwei = get_center(max_contour)
            # print('dingwei_x', cX_dingwei)
            cv2.circle(frame, (cX_dingwei, cY_dingwei), 5, (255, 0, 125), -1)
            cv2.putText(frame, f'D({cX_dingwei},{cY_dingwei})', (cX_dingwei - 20, cY_dingwei - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 对放置在两边的棋子进行检测，并将坐标储存！ 一次标定即可
    sorted_black_centers = sort_and_store_coordinates(black_centers)
    sorted_white_centers = sort_and_store_coordinates(white_centers)
    # 对于black_centers和white_centers进行筛选
    sorted_black_centers = remove_duplicates_and_limit_centers(sorted_black_centers)
    sorted_white_centers = remove_duplicates_and_limit_centers(sorted_white_centers)

    print('black',sorted_black_centers)
    print('white',sorted_white_centers)
    cv2.imshow("Squares with Centers", frame)
    #
    cv2.imshow("Black Mask", mask_black)
    cv2.imshow("mask_white Mask", mask_white)

    # cv2.imshow("White Mask", mask_dingwei)
    cv2.imshow("mask_dingwei", mask_dingwei)

    return sorted_black_centers, sorted_white_centers, cX_dingwei, cY_dingwei,white_centers_mid

def nothing(x):
    pass

def target_x_err(select_num, bool_white, sorted_white_centers, sorted_black_centers, cx_dingwei_current):
    if 1 <= select_num <= 5:
        if bool_white == 1 and len(sorted_white_centers) >= select_num:
            err_x = sorted_white_centers[select_num - 1][0] - cx_dingwei_current
            print(f"White piece coordinate at index {select_num - 1}: {sorted_white_centers[select_num - 1]}")
        elif bool_white == 0 and len(sorted_black_centers) >= select_num:
            err_x = sorted_black_centers[select_num - 1][0] - cx_dingwei_current
            print(f"Black piece coordinate at index {select_num - 1}: {sorted_black_centers[select_num - 1]}")
        else:
            err_x = 0
        print("Error x:", err_x)
    else:
        err_x = 0
        print("Invalid selection number or no pieces available.")
    return err_x

def target_y_err(select_num, bool_white, sorted_white_centers, sorted_black_centers, cy_dingwei_current):
    if 1 <= select_num <= 5:
        if bool_white == 1 and len(sorted_white_centers) >= select_num:
            err_y = sorted_white_centers[select_num - 1][0] - cy_dingwei_current
            print(f"White piece coordinate at index {select_num - 1}: {sorted_white_centers[select_num - 1]}")
        elif bool_white == 0 and len(sorted_black_centers) >= select_num:
            err_y = sorted_black_centers[select_num - 1][0] - cy_dingwei_current
            print(f"Black piece coordinate at index {select_num - 1}: {sorted_black_centers[select_num - 1]}")
        else:
            err_y = 0
        print("Error y:", err_y)
    else:
        err_y = 0
        print("Invalid selection number or no pieces available.")
    return err_y

def target_x_err_place_qizi(select_num, cx_dingwei_current, grid_coordinates):
    if 1 <= select_num <= 9:
        # 获取目标x坐标
        target_x = grid_coordinates[0, select_num - 1][0]
        err_x = target_x - cx_dingwei_current
        print(f"Target x coordinate for select_num {select_num}: {target_x}")
        print("Error x:", err_x)
        return err_x
    else:
        print("Invalid selection number.")
        return None

def target_y_err_place_qizi(select_num, cy_dingwei_current, grid_coordinates):
    if 1 <= select_num <= 9:
        # 获取目标y坐标
        target_y = grid_coordinates[0, select_num - 1][1]
        err_y = target_y - cy_dingwei_current
        print(f"Target y coordinate for select_num {select_num}: {target_y}")
        print("Error y:", err_y)
        return err_y
    else:
        print("Invalid selection number.")
        return None

# 判断人把黑棋放在哪里了---------------------------------------------
# 假设这是接收到的按钮按下的标志位
button_pressed = False
# 类似的逻辑可以应用于 detect_white_piece
last_white_update_time = time.time()
white_update_interval = 5
white_piece_last_positions = []

def detect_white_piece(white_centers, grid_coordinates, board):
    global last_white_update_time, white_piece_last_positions
    detected = False

    current_time = time.time()
    if current_time - last_white_update_time < white_update_interval:
        return detected  # 如果在时间间隔内，不进行更新

    for (x, y) in white_centers:
        min_distance = 10
        closest_row, closest_col = -1, -1
        for i in range(3):
            for j in range(3):
                grid_x, grid_y = grid_coordinates[0, i * 3 + j]
                distance = np.sqrt((x - grid_x) ** 2 + (y - grid_y) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_row, closest_col = i, j
        if closest_row != -1 and closest_col != -1:
            if board[closest_row, closest_col] != 2 and (closest_row, closest_col) not in white_piece_last_positions:
                board[closest_row, closest_col] = 2  # 更新棋盘状态，2表示白棋
                print(f"Detected white piece at grid ({closest_row}, {closest_col})")
                white_piece_last_positions.append((closest_row, closest_col))
                if len(white_piece_last_positions) > 30:
                    white_piece_last_positions.pop(0)
                detected = True
                last_white_update_time = current_time
    return detected
# def detect_white_piece(white_centers, grid_coordinates, board):
#     # 检测白棋位置
#     detected = False
#     for (x, y) in white_centers:
#         min_distance = 5
#         closest_row, closest_col = -1, -1
#         for i in range(3):
#             for j in range(3):
#                 grid_x, grid_y = grid_coordinates[0, i * 3 + j]
#                 distance = np.sqrt((x - grid_x) ** 2 + (y - grid_y) ** 2)
#                 if distance < min_distance:
#                     min_distance = distance
#                     closest_row, closest_col = i, j
#         if closest_row != -1 and closest_col != -1:
#             if board[closest_row, closest_col] != 2:  # 检查是否已更新
#                 board[closest_row, closest_col] = 2  # 更新棋盘状态，1表示黑棋
#                 print(f"Detected white piece at grid ({closest_row}, {closest_col})")
#                 detected = True
#     return detected

# 引入一个计数器和时间戳来记录棋子的位置变化
last_black_update_time = time.time()
black_update_interval = 5  # 设置时间间隔为1秒
black_piece_last_positions = []
# def detect_black_piece(black_centers, grid_coordinates, board):
#     # 检测黑棋位置
#     detected = False
#     for (x, y) in black_centers:
#         min_distance = 10
#         closest_row, closest_col = -1, -1
#         for i in range(3):
#             for j in range(3):
#                 grid_x, grid_y = grid_coordinates[0, i * 3 + j]
#                 distance = np.sqrt((x - grid_x) ** 2 + (y - grid_y) ** 2)
#                 if distance < min_distance:
#                     min_distance = distance
#                     closest_row, closest_col = i, j
#         if closest_row != -1 and closest_col != -1:
#             if board[closest_row, closest_col] != 1:  # 检查是否已更新
#                 board[closest_row, closest_col] = 1  # 更新棋盘状态，1表示黑棋
#                 print(f"Detected black piece at grid ({closest_row}, {closest_col})")
#                 detected = True
#     print('detected',detected)
#     return detected
def detect_black_piece(black_centers, grid_coordinates, board):
    global last_black_update_time, black_piece_last_positions
    detected = False

    current_time = time.time()
    if current_time - last_black_update_time < black_update_interval:
        return detected  # 如果在时间间隔内，不进行更新

    for (x, y) in black_centers:
        min_distance = 10
        closest_row, closest_col = -1, -1
        for i in range(3):
            for j in range(3):
                grid_x, grid_y = grid_coordinates[0, i * 3 + j]
                distance = np.sqrt((x - grid_x) ** 2 + (y - grid_y) ** 2)
                if distance < min_distance:

                    min_distance = distance
                    closest_row, closest_col = i, j
        if closest_row != -1 and closest_col != -1:
            # 确保位置没有被多次更新
            if board[closest_row, closest_col] != 1 and (closest_row, closest_col) not in black_piece_last_positions:
                board[closest_row, closest_col] = 1  # 更新棋盘状态，1表示黑棋
                print(f"Detected black piece at grid ({closest_row}, {closest_col})")
                black_piece_last_positions.append((closest_row, closest_col))
                if len(black_piece_last_positions) > 30:  # 保持最近10个位置
                    black_piece_last_positions.pop(0)
                detected = True
                last_black_update_time = current_time
    return detected




original_board = board.copy()
import numpy as np
import cv2

def find_winning_move(board, piece):
    # 检查是否有即将连成三子的情况，返回可以获胜的坐标
    # 横 竖 直
    for row in range(3):
        if board[row, 0] == board[row, 1] == piece and board[row, 2] == 0:
            return (row, 2)
        if board[row, 0] == board[row, 2] == piece and board[row, 1] == 0:
            return (row, 1)
        if board[row, 1] == board[row, 2] == piece and board[row, 0] == 0:
            return (row, 0)
    for col in range(3):
        if board[0, col] == board[1, col] == piece and board[2, col] == 0:
            return (2, col)
        if board[0, col] == board[2, col] == piece and board[1, col] == 0:
            return (1, col)
        if board[1, col] == board[2, col] == piece and board[0, col] == 0:
            return (0, col)
    if board[0, 0] == board[1, 1] == piece and board[2, 2] == 0:
        return (2, 2)
    if board[0, 0] == board[2, 2] == piece and board[1, 1] == 0:
        return (1, 1)
    if board[1, 1] == board[2, 2] == piece and board[0, 0] == 0:
        return (0, 0)
    if board[0, 2] == board[1, 1] == piece and board[2, 0] == 0:
        return (2, 0)
    if board[0, 2] == board[2, 0] == piece and board[1, 1] == 0:
        return (1, 1)
    if board[1, 1] == board[2, 0] == piece and board[0, 2] == 0:
        return (0, 2)

    return None

def calculate_white_piece_move(board):
    # 先检查是否可以三连
    move = find_winning_move(board, 2)  # 1表示黑棋 2是白棋
    if move:
        board[move[0], move[1]] = 2  # 更新棋盘状态
        return move

    # 再检查是否有需要防守的情况
    move = find_winning_move(board, 1)  # 1表示黑棋
    if move:
        board[move[0], move[1]] = 1  # 更新棋盘状态
        return move

    # 检查黑棋是否下在中间，且棋盘其他位置为空
    if board[1, 1] == 1 and np.all(
            board[np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1], [2, 2]])] == 0):
        for (i, j) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
            if board[i, j] == 0:
                return (i, j)

    # 如果没有需要防守的情况，随机选择一个空位
    for row in range(3):
        for col in range(3):
            if board[row, col] == 0:
                return (row, col)
    return None

def place_white_piece(row, col,grid_coordinates):
    board[row, col] = 2  # 更新棋盘状态，2表示白棋
    target_point = grid_coordinates[0, row * 3 + col]  # 获取对应的坐标点
    print(f"Device places white piece at ({row}, {col})")
    return target_point

# 检查胜利条件
def check_winner(board):
    # 检查行
    for row in board:
        if row[0] == row[1] == row[2] and row[0] != '':
            return row[0]

    # 检查列
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] != '':
            return board[0][col]

    # 检查对角线
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != '':
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != '':
        return board[0][2]
    return None

def detect_and_correct_moved_piece(current_white_centers, previous_white_centers, threshold=20):
    for i, current_center in enumerate(current_white_centers):
        if i < len(previous_white_centers):
            previous_center = previous_white_centers[i]
            distance = np.linalg.norm(np.array(current_center) - np.array(previous_center))
            if distance > threshold:
                print(f"Piece moved from {previous_center} to {current_center}. Moving it back.")
                # 将棋子放回之前的位置
                move_piece_back(previous_center)
                previous_white_centers_mid = white_centers_mid

        else:
            break  # 防止 current_white_centers 的长度大于 previous_white_centers

def move_piece_back(position):
    # 实现将棋子移动到指定位置的逻辑
    print(f"Moving piece back to {position}")
    # 这里需要加入实际的控制代码来移动棋子

# 第一个任务
def first_task(board,target_num,sorted_black_centers_current):
    # 第五个方框的坐标点
    target_point_first_task = grid_coordinates[0, 4]

    # Get the coordinates of the black piece corresponding to target_num
    if target_num <= len(sorted_black_centers_current):
        black_piece_coords = sorted_black_centers_current[target_num - 1]

    board[1,1] = 1
    print(f"目标棋子的坐标点: {target_point_first_task}")


    # 将 x 和 y 转换为字符串，并添加 \r\n 作为分隔符
    # message = f"{x_rr}\r\n{y_rr}\r\n"
    # ser.write(message.encode('utf-8'))
    # print(f"Sent: {x_rr}, {y_rr}")
    #传输x y 坐标值 形成闭环控制

# 第二个任务
black_or_white_temp_flag = 0
def second_task(board,target_num,sorted_black_centers_current,sorted_white_centers_current):
    # B + 11 12 13 14 15 白发
    # A + 21 22 23 24 25 黑边
    global black_or_white_temp_flag
    if black_or_white_temp_flag == 0:
        sorted_black_centers_current_temp = sorted_black_centers_current
        sorted_white_centers_current_temp = sorted_white_centers_current
        black_or_white_temp_flag = 1

    # if target_num == 'B11':
    #     target_point_sec_task = grid_coordinates[0, 0]
    #
    # if target_num <= len(sorted_black_centers_current):
    #     black_piece_coords = sorted_black_centers_current[target_num - 1]

    #传输x y 坐标值 形成闭环控制

def get_diagonal_positions():
    """ 返回所有对角线位置的坐标 """
    return [(0, 0), (2, 2), (0, 2), (2, 0)]

def get_empty_diagonal_position(board):
    """ 返回一个空的对角线位置 """
    diagonal_positions = get_diagonal_positions()
    for pos in diagonal_positions:
        if board[pos[0], pos[1]] == 0:
            return pos
    return None  # 如果没有空的对角线位置

count_cishu = 0


#第四个任务 ---------------------------------------------------------------------------
def get_diagonal_positions():
    """ 返回所有对角线位置的坐标 """
    return [(0, 0), (2, 2), (0, 2), (2, 0)]

def get_empty_diagonal_position(board):
    """ 返回一个空的对角线位置 """
    diagonal_positions = get_diagonal_positions()
    for pos in diagonal_positions:
        if board[pos[0], pos[1]] == 0:
            return pos
    return None  # 如果没有空的对角线位置

def fourth_task(board, bisheng_flag_1, judge_mid_or_bian_1):
    global bisheng_flag
    global judge_mid_or_bian
    win_flag = 0
    back_normal = 0
    jiance_again = 0
    print('judge_mid_or_bian_1', judge_mid_or_bian_1)
    # if xiaqi_flag == 1:
    #     original_board = board
    #     bisheng_flag = detect_white_piece(sorted_white_centers_current, grid_coordinates, bisheng_flag)

    # 先检查是否可以三连

    #第一步下棋 fourth_task_plack 1~9
    fourth_task_plack = 3
    if np.all(board == 0) and fourth_task_plack is not None:
        row = (fourth_task_plack - 1) // 3
        col = (fourth_task_plack - 1) % 3
        board[row, col] = 1

        position_index = 3 * row + col + 1
        # 将 x 和 y 转换为字符串，并添加 \r\n 作为分隔符
        message = f"{position_index}\r\n{1}\r\n"  # 发送给下位机下第几个棋子
        ser.write(message.encode('utf-8'))
        print(f"发送给下位机: {row + 3 * col}, {1}")

        print('row', row)
        print('col', col)
        return (row, col), win_flag

    print(board)

    if judge_mid_or_bian_1 == 0:
        if board[1, 1] == 1 and \
                (board[0, 0] == 0 and board[0, 1] == 0 and board[0, 2] == 0 and
                 board[1, 0] == 0 and board[1, 2] == 0 and
                 board[2, 0] == 0 and board[2, 1] == 0 and board[2, 2] == 0):
            judge_mid_or_bian_1 = 1  # 中间的情况
            # bisheng_flag_1 = detect_white_piece(sorted_white_centers_current, grid_coordinates,bisheng_flag_1)
            # print("mamba out:",bisheng_flag_1)
        else:
            judge_mid_or_bian_1 = 2

    if (judge_mid_or_bian_1 == 1):
        if detect_white_piece(sorted_white_centers_current, grid_coordinates, board):
            if board[1, 1] == 1 and \
                    (board[1, 0] == 2 or board[1, 2] == 2 or board[0, 1] == 2 or
                     board[2, 1] == 2):  # 白棋GG
                judge_mid_or_bian_1 = 3
                jiance_again = 1

            elif board[1, 1] == 1 and \
                    (board[0, 0] == 2 or board[0, 2] == 2 or board[2, 0] == 2 or
                     board[2, 2] == 2):  # 白棋继续
                judge_mid_or_bian_1 = 2
                back_normal = 1
                print('back to normal')
        else:
            print("No new white pieces detected. Waiting... 同时：judge_mid_or_bian_1", judge_mid_or_bian_1)

    # 正常下棋
    if judge_mid_or_bian_1 == 2:
        print('11111')
        if detect_white_piece(sorted_white_centers_current, grid_coordinates, board) or back_normal:
            move = find_winning_move(board, 1)  # 1表示黑棋
            if move:
                board[move[0], move[1]] = 1  # 更新棋盘状态
                print('win!!!!')
                win_flag = 1
                return move, win_flag
            # 再检查是否有需要防守的情况
            move = find_winning_move(board, 2)  # 2表示白棋
            if move:
                board[move[0], move[1]] = 1  # 更新棋盘状态
                return move, win_flag

            # 确保白棋数量不超过黑棋数量
            black_pieces = np.sum(board == 1)
            white_pieces = np.sum(board == 2)

            if white_pieces <= black_pieces:
                empty_positions = [(row, col) for row in range(3) for col in range(3) if board[row, col] == 0]

                if empty_positions:
                    row, col = random.choice(empty_positions)
                    print("Move white piece to diagonal,", (row, col))
                    board[row, col] = 1  # 继续下黑棋

                    position_index = 3 * row + col + 1
                    # 将 x 和 y 转换为字符串，并添加 \r\n 作为分隔符
                    message = f"{position_index}\r\n{1}\r\n" #发送给下位机下第几个棋子
                    ser.write(message.encode('utf-8'))
                    print(f"Sent: {row + 3 * col}, {1}")

                    return (row, col), win_flag
        else:
            print("No new White pieces detected. Waiting...")

    if judge_mid_or_bian_1 == 3:
        print('111111111111111111111:', judge_mid_or_bian_1)
        if detect_white_piece(sorted_white_centers_current, grid_coordinates, board) or jiance_again == 1:
            jiance_again = 0
            print('222222222222222')
            move = get_empty_diagonal_position(board)
            print(move)
            if move:
                print(f"Move black piece to diagonal position: {move}")
                board[move[0], move[1]] = 1
            else:
                print("No empty diagonal position available")
            return move, win_flag
        else:
            print("No new White pieces detected. Waiting...")

    bisheng_flag = bisheng_flag_1
    judge_mid_or_bian = judge_mid_or_bian_1

    return None, win_flag
# ----------------------------------------------------------------------------------

#第五个任务 -----------------------------------------------------------------------------
def fifth_task(board,bisheng_flag_1,judge_mid_or_bian_1,board_qizi):
    global bisheng_flag
    global judge_mid_or_bian
    global count_cishu
    print('judge_mid_or_bian_1',judge_mid_or_bian_1)

    back_normal = 0
    jiance_again = 0

    print(board)
    # if xiaqi_flag == 1:
    #     original_board = board
    #     bisheng_flag = detect_white_piece(sorted_white_centers_current, grid_coordinates, bisheng_flag)

    if judge_mid_or_bian_1 == 0:
        if detect_black_piece(sorted_black_centers_current, grid_coordinates, board):

            if board[1, 1] == 1 and \
                    (board[0, 0] == 0 and board[0, 1] == 0 and board[0, 2] == 0 and
                     board[1, 0] == 0 and board[1, 2] == 0 and
                     board[2, 0] == 0 and board[2, 1] == 0 and board[2, 2] == 0):
                judge_mid_or_bian_1 = 1  # 中间的情况
                jiance_again = 1
                # bisheng_flag_1 = detect_white_piece(sorted_white_centers_current, grid_coordinates,bisheng_flag_1)
                print("mamba out:", judge_mid_or_bian_1)
            else:
                judge_mid_or_bian_1 = 2
                back_normal = 1
        else:
            print("等人下第一步棋子")


    if (judge_mid_or_bian_1 == 1):
        if detect_black_piece(sorted_black_centers_current, grid_coordinates, board)or jiance_again:
            jiance_again = 0
            print(111)
            for pos in [(0, 0), (0, 2), (2, 0), (2, 2)]:
                # 为空 并且棋盘白色有变化
                if board[pos[0], pos[1]] == 0 :
                    #计算当前棋盘内所有坐标并保存
                    board[pos[0], pos[1]] = 2
                    judge_mid_or_bian_1 = 2
                    back_normal = 1
                    break
        else:
            print("No new black pieces detected. Waiting...")

    #正常下棋

    #思路
    #人下完后 发送标志位xiaqi_flag，然后装置开始下棋，下完后亮灯发标志位给下位机
    if judge_mid_or_bian_1 == 2:
        #if xiaqi_flag == 1: 只需要接收一个下位机发过来的下棋标志位就行了
        if detect_black_piece(sorted_black_centers_current, grid_coordinates, board) or back_normal:
            back_normal = 0
            move = find_winning_move(board, 1)  # 1表示黑棋
            if move:
                board[move[0], move[1]] = 2  # 更新棋盘状态
                print('win!!!!')
                return move
            # 再检查是否有需要防守的情况
            move = find_winning_move(board, 1)  # 2表示白棋
            if move:
                board[move[0], move[1]] = 2  # 更新棋盘状态
                return move

            # 确保白棋数量不超过黑棋数量
            black_pieces = np.sum(board == 1)
            white_pieces = np.sum(board == 2)

            if white_pieces < black_pieces:
                empty_positions = [(row, col) for row in range(3) for col in range(3) if board[row, col] == 0]

                if empty_positions:
                    # 随机选择一个空白位置
                    row, col = random.choice(empty_positions)
                    print("Move white piece to diagonal,", (row, col))
                    board[row, col] = 2  # 2表示白棋
                    return (row, col)

        else:
            print("No new black pieces detected. Waiting...")


    #装置把黑棋放在中间 第一

    #
    # # 如果bisheng_flag为3，黑棋必须下在对角线的一个位置
    # if judge_mid_or_bian_1 == 3:
    #     print('111111111111111111111:   ', judge_mid_or_bian_1)
    #     # if detect_black_piece(sorted_black_centers_current, grid_coordinates, board):
    #     if detect_white_piece(sorted_white_centers_current, grid_coordinates, board):
    #         print('222222222222222')
    #         move = get_empty_diagonal_position(board)
    #         print(move)
    #         if move:
    #             print(f"Move black piece to diagonal position: {move}")
    #             # 这里可以加入实际下棋的逻辑，例如更新棋盘状态
    #             board[move[0], move[1]] = 1
    #             # 发送亮灯
    #             # light_up_flag = 1
    #         else:
    #             print("No empty diagonal position available")
    #     else:
    #         print("No new White pieces detected. Waiting...")

    bisheng_flag = bisheng_flag_1
    judge_mid_or_bian = judge_mid_or_bian_1




# 定义黄色的HSV范围
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])
# 创建窗口和滑动条
cv2.namedWindow("Trackbars")
cv2.createTrackbar("Canny Thresh1", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("Canny Thresh2", "Trackbars", 150, 255, nothing)

yunxing_once = 0
yunxing_once_1 = 0
def get_perspective_transform(src_points, dst_points):
    return cv2.getPerspectiveTransform(src_points, dst_points)

def sort_points(points):
    points = sorted(points, key=lambda x: x[1])  # 按y值排序
    top_points = sorted(points[:2], key=lambda x: x[0])  # 按x值排序获取顶部的两个点
    bottom_points = sorted(points[2:], key=lambda x: x[0])  # 按x值排序获取底部的两个点
    return np.array([top_points[0], top_points[1], bottom_points[0], bottom_points[1]], dtype="float32")

def perspective_transform_points(points, M):
    points = np.array(points, dtype="float32")
    points = np.array([points])
    transformed_points = cv2.perspectiveTransform(points, M)
    return transformed_points[0]


# 定义帧率计算相关变量
prev_time = time.time()
frame_count = 0
while True:
    ret, frame = cap.read()

    frame_count += 1
    current_time = time.time()
    elapsed_time = current_time - prev_time
    if elapsed_time > 1.0:
        fps = frame_count / elapsed_time
        frame_count = 0
        prev_time = current_time
        print(f"FPS: {fps:.2f}")


    if yunxing_once <=10:
        yunxing_once +=1
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        src_points = find_background_contour(mask_yellow)
        src_points = sort_points(src_points)
    else:
        if len(src_points) == 4:
            # 提取感兴趣区域 (ROI)
            x_min = int(min(src_points, key=lambda x: x[0])[0])
            x_max = int(max(src_points, key=lambda x: x[0])[0])
            y_min = int(min(src_points, key=lambda x: x[1])[1])
            y_max = int(max(src_points, key=lambda x: x[1])[1])

            roi = frame[y_min:y_max, x_min:x_max]

            # 显示提取的区域
        cv2.imshow("ROI", roi)

        canny_thresh1 = cv2.getTrackbarPos("Canny Thresh1", "Trackbars")
        canny_thresh2 = cv2.getTrackbarPos("Canny Thresh2", "Trackbars")
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blurred, 50, 250)
        # # 膨胀操作
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        squares = find_squares(edges)
        cv2.imshow("edges", edges)

        centers = []
        for square in squares:
            cX, cY = get_square_center(square)
            if 50 < cX < 200:
                centers.append((cX, cY))
                # cv2.circle(roi, (cX, cY), 5, (0, 0, 255), -1)
                # cv2.drawContours(roi, [square], 0, (0, 255, 0), 2)

        if yunxing_once_1 <= 10:
            yunxing_once_1 += 1
            if len(centers) == 9:
                # 进行透视变换
                M = get_perspective_transform(src_points, dst_points)
                transformed_centers = perspective_transform_points(centers, M)


                # 按 y 值分组
                centers_sorted_by_y = sorted(transformed_centers, key=lambda p: p[1])
                y_groups = []

                # 创建三组
                group_size = max(len(centers) // 3, 1)
                for i in range(0, len(centers_sorted_by_y), group_size):
                    y_groups.append(centers_sorted_by_y[i:i + group_size])

                # 对每组内的坐标按 x 值排序
                sorted_groups = [sorted(group, key=lambda p: p[0]) for group in y_groups]

                # 合并所有排序后的组
                final_sorted_centers = [item for group in sorted_groups for item in group]

                # 更新 grid_coordinates
                for i, sorted_center in enumerate(final_sorted_centers):
                    for orig_center in centers:
                        if np.allclose(perspective_transform_points([orig_center], M)[0], sorted_center, atol=1):
                            grid_coordinates[0, i] = orig_center
            else:
                # print(f"Error: Expected 9 centers, but got {len(final_sorted_centers)}. Cannot reshape array.")
                continue
        print('grid_coordinates')
        print(grid_coordinates)
        sorted_black_centers_current, sorted_white_centers_current, cx_dingwei_current, cy_dingwei_current , white_centers_mid= img_process(roi,grid_coordinates)

        # 黑白棋子的坐标

        print('sorted_white_centers_current',sorted_white_centers_current)
        print('sorted_black_centers_current',sorted_black_centers_current)


        #第一个任务
        # target_num = 3
        # first_task(board,target_num,sorted_black_centers_current)

        #第二个任务
        # second_task(board,target_num,sorted_black_centers_current,sorted_white_centers_current)


        #第四个任务
        move,win_flag = fourth_task(board, bisheng_flag, judge_mid_or_bian)
        #第五个任务
        # move= fifth_task(board,bisheng_flag,judge_mid_or_bian,board_qizi)
        # print('Move to',move)


        # if win_flag == 1:
        #     win_flag = 0
        #     print("Win!!!!")
        #     print("Win!!!!")
        #     print("Win!!!!")
        #     print("Win!!!!")
        #     break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()