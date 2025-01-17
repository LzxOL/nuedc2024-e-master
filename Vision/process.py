import cv2
import numpy as np

cX_dingwei, cY_dingwei = 0, 0
# 打开摄像头
cap = cv2.VideoCapture(0)

chosed_black = 0
count_flag = 0
y_count_flag = 0#抓棋子的x标志位
catch_qizi_flag = 0 #抓棋子的标志位

place_where_num = 0 #下位机发来的选择哪个格子的标志位
biaoding_flag = 0 #
# 第四个任务
fourth_task_plack = 0#第四个任务接收下位机的放置位置的标志位
bisheng_flag = 0
# 各种列表 ——————————————————————————————————————————————————————
# 将棋盘中的九宫格各个宫格存在 1*9 的列表中
grid_coordinates = np.zeros((1, 9, 2), dtype=int)

# 棋盘部分霍夫圆的坐标列表
black_centers_mid = []
white_centers_mid = []
previous_white_centers_mid = []

# 棋盘两边棋子的坐标列表
black_centers = []
white_centers = []

# 棋盘中已经放好了的棋子的坐标

# 初始化棋盘状态，0表示空位，1表示黑棋，2表示白棋
board = np.zeros((3, 3), dtype=int)
previous_board = np.zeros((3, 3), dtype=int)  # 用于保存之前的棋盘状态



# ——————————————————————————————————————————————————————————————————



# 四个点定义的坐标
src_points = np.array([[193, 98], [495, 103], [190, 290], [494, 293]], dtype="float32")
# 目标点，定义变换后图像的大小
dst_points = np.array([[0, 0], [300, 0], [0, 200], [300, 200]], dtype="float32")


# 黑白棋子的阈值
def threshold_frame(frame, mask):
    # 设置黑色和白色的阈值
    black_lower = np.array([0, 0, 0])
    black_upper = np.array([179, 146, 147])

    white_lower = np.array([0, 0, 173])
    white_upper = np.array([173, 63, 255])

    DINGWEI_lower = np.array([0, 144, 128])
    DINGWEI_upper = np.array([22, 255, 255])

    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 对图像进行阈值处理
    mask_black = cv2.inRange(hsv, black_lower, black_upper)
    mask_white = cv2.inRange(hsv, white_lower, white_upper)
    mask_dingwei = cv2.inRange(hsv, DINGWEI_lower, DINGWEI_upper)

    # 仅保留感兴趣区域
    mask_black = cv2.bitwise_and(mask_black, mask_black, mask=mask)
    mask_white = cv2.bitwise_and(mask_white, mask_white, mask=mask)

    return mask_black, mask_white, mask_dingwei

# def find_squares(edges):
#     contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     squares = []
#     for contour in contours:
#         epsilon = 0.02 * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)
#         if len(approx) == 4:  # 只考虑四边形
#             area = cv2.contourArea(approx)
#             if 500 < area < 5000:  # 过滤掉太小或太大的轮廓
#                 squares.append(approx)
#     return squares
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
def img_process(frame):
    global cX_dingwei, cY_dingwei
    canny_thresh1 = cv2.getTrackbarPos("Canny Thresh1", "Trackbars")
    canny_thresh2 = cv2.getTrackbarPos("Canny Thresh2", "Trackbars")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)

    # 膨胀操作
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    cv2.imshow("edges", edges)


    squares = find_squares(edges)



    # biaoding
    centers = []
    for square in squares:
        cX, cY = get_square_center(square)
        if cX>100 and cX <200:
            centers.append((cX, cY))

            # cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
            # cv2.drawContours(frame, [square], 0, (0, 255, 0), 2)
        if len(centers) == 9:
            # 按照 cY 坐标倒序排序
            centers_sorted = sorted(centers, key=lambda x: x[1], reverse=True)
            for i in range(9):
                grid_coordinates[0, i] = centers_sorted[i]
        # print('centers', grid_coordinates)


    # # 显示中心坐标
    # for center in centers:
    #     cv2.putText(frame, f'({center[0]},{center[1]})', (center[0] - 20, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.5, (255, 0, 0), 1)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    frame_qizi = frame.copy()
    frame_qizi = cv2.GaussianBlur(frame_qizi, (5, 5), 0)

    # frame_qizi = cv2.erode(frame_qizi,(7, 7), 0)

    roi_mask = create_roi_mask(frame_qizi.shape, [(0, 400), (650, frame.shape[1])])

    # 对于黑白棋子的阈值提取
    mask_black, mask_white, mask_dingwei = threshold_frame(frame_qizi, roi_mask)
    # 自适应阈值处理
    gray_black = cv2.bitwise_and(gray, gray, mask=mask_black)
    gray_white = cv2.bitwise_and(gray, gray, mask=mask_white)
    adaptive_thresh_black = cv2.adaptiveThreshold(gray_black, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY_INV, 7, 2)
    adaptive_thresh_white = cv2.adaptiveThreshold(gray_white, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY_INV, 7, 2)

    # 形态学操作
    kernel = np.ones((3, 3), np.uint8)
    adaptive_thresh_black = cv2.morphologyEx(adaptive_thresh_black, cv2.MORPH_CLOSE, kernel)
    adaptive_thresh_white = cv2.morphologyEx(adaptive_thresh_white, cv2.MORPH_CLOSE, kernel)

    # 裁剪感兴趣区域
    region_black = mask_black[150:350, 150:350]
    region_white = mask_white[150:350, 150:350]

    # # 霍夫圆检测
    # circles_black = cv2.HoughCircles(region_black, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30,
    #                                  minRadius=5, maxRadius=30)
    # circles_white = cv2.HoughCircles(region_white, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30,
    #                                  minRadius=5, maxRadius=30)


    # if circles_black is not None:
    #     circles_black = np.round(circles_black[0, :]).astype("int")
    #     for (x, y, r) in circles_black:
    #         black_centers_mid.append((x + 150, y + 150))  # 将区域内的坐标转换为原始图像的坐标
    #         cv2.circle(frame, (x + 150, y + 150), r, (0, 0, 0), 2)
    #         cv2.circle(frame, (x + 150, y + 150), 5, (0, 0, 255), -1)
    #         cv2.putText(frame, f'B({x + 150},{y + 150})', (x + 130, y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #                     (255, 255, 255), 1)
    #
    #
    # if circles_white is not None:
    #     circles_white = np.round(circles_white[0, :]).astype("int")
    #     for (x, y, r) in circles_white:
    #         white_centers_mid.append((x + 150, y + 150))  # 将区域内的坐标转换为原始图像的坐标
    #         cv2.circle(frame, (x + 150, y + 150), r, (255, 255, 255), 2)
    #         cv2.circle(frame, (x + 150, y + 150), 5, (0, 0, 255), -1)
    #         cv2.putText(frame, f'W({x + 150},{y + 150})', (x + 130, y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
    #                     1)


    # 查找黑色和白色棋子的中心
    # 找到黑色和白色棋子的轮廓
    black_contours, _ = cv2.findContours(adaptive_thresh_black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    white_contours, _ = cv2.findContours(adaptive_thresh_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # black_contours, _ = cv2.findContours(mask_black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # white_contours, _ = cv2.findContours(mask_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    dingwei_contours, _ = cv2.findContours(mask_dingwei, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #对两边的棋子进行标定---------------------------------------------------------
    for contour in black_contours:
        if cv2.contourArea(contour) > 200:  # 过滤掉小噪声
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
        if cv2.contourArea(contour) > 200:  # 过滤掉小噪声
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if 0.7 < circularity < 1.2:  # 判断是否为圆形
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

    return sorted_black_centers, sorted_white_centers, cX_dingwei, cY_dingwei,grid_coordinates,white_centers_mid

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

def detect_black_piece(black_centers, grid_coordinates):
    # 检测黑棋位置
    for (x, y) in black_centers:
        min_distance = float('inf')
        closest_row, closest_col = -1, -1
        for i in range(3):
            for j in range(3):
                grid_x, grid_y = grid_coordinates[0, i * 3 + j]
                distance = np.sqrt((x - grid_x) ** 2 + (y - grid_y) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_row, closest_col = i, j
        if closest_row != -1 and closest_col != -1:
            board[closest_row, closest_col] = 1  # 更新棋盘状态，1表示黑棋
            print(f"Detected black piece at grid ({closest_row}, {closest_col})")

def detect_white_piece(white_centers, grid_coordinates):
    # 检测白棋位置
    for (x, y) in white_centers:
        min_distance = float('inf')
        closest_row, closest_col = -1, -1
        for i in range(3):
            for j in range(3):
                grid_x, grid_y = grid_coordinates[0, i * 3 + j]
                distance = np.sqrt((x - grid_x) ** 2 + (y - grid_y) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_row, closest_col = i, j
        if closest_row != -1 and closest_col != -1:
            board[closest_row, closest_col] = 2  # 更新棋盘状态，2表示白棋
            print(f"Detected white piece at grid ({closest_row}, {closest_col})")

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
        return move

    # 再检查是否有需要防守的情况
    move = find_winning_move(board, 1)  # 1表示黑棋
    if move:
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

# 第四个任务
def fourth_task(board):

    # 先检查是否可以三连
    move = find_winning_move(board, 1)   # 1表示黑棋
    if move:
        return move

    # 再检查是否有需要防守的情况
    move = find_winning_move(board, 2)  # 2表示白棋
    if move:
        return move


    #第一步下棋 fourth_task_plack 1~9
    if np.all(board == 0) and fourth_task_plack is not None:
        row = fourth_task_plack // 3
        col = fourth_task_plack % 3
        return board[row, col]

    #装置把黑棋放在中间
    if board[1, 1] == 1 and np.all(board[np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1], [2, 2]])] == 0):
        bisheng_flag = 1

    if bisheng_flag == 1 and board[1, 1] == 1 and np.all(board[np.array(([1,0],[1,2],[0,1],[2,1]))] == 2):
        bisheng_flag = 2

    # 如果bisheng_flag为2，黑棋必须下在对角线的一个位置
    if bisheng_flag == 2:
        # 对角线位置的映射
        diag_moves = {
            0: (2, 2),
            2: (0, 0),
            6: (0, 2),
            8: (2, 0)
        }

        # 确定当前白棋的位置并找到一个空的对角位置
        for (i, j) in [0, 2, 6, 8]:
            row, col = divmod(i, 3)
            if board[row, col] == 0:
                print(f"Move black piece to diagonal position ({row}, {col})")
                return (row, col)

    if bisheng_flag == 0:
        # 如果没有需要防守的情况，随机选择一个空位
        for row in range(3):
            for col in range(3):
                if board[row, col] == 0:
                    return (row, col)



# 定义黄色的HSV范围
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])
# 创建窗口和滑动条
cv2.namedWindow("Trackbars")
cv2.createTrackbar("Canny Thresh1", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("Canny Thresh2", "Trackbars", 150, 255, nothing)

while True:
    ret, frame = cap.read()

    if not ret:
        break
    # 获取透视变换矩阵
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    # 进行透视变换
    warped = cv2.warpPerspective(frame, M, (300, 200))
    # # 转换为HSV颜色空间
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #
    # # 创建遮罩，过滤出黄色区域
    # mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    #
    # # 对遮罩应用一些形态学操作以消除噪点
    # mask = cv2.erode(mask, None, iterations=2)
    # mask = cv2.dilate(mask, None, iterations=2)
    #
    # # 寻找轮廓
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # for contour in contours:
    #     # 忽略太小的轮廓
    #     if cv2.contourArea(contour) < 300:
    #         continue
    #
    #     # 计算边界框
    #     x, y, w, h = cv2.boundingRect(contour)
    #
    #     # 提取黄色边框内的图像
    #     frame = frame[y+10:y + h-10, x+5:x + w-10]
    # cv2.imshow('Original', frame)
    sorted_black_centers_current, sorted_white_centers_current, cx_dingwei_current, cy_dingwei_current ,grid_coordinates, white_centers_mid= img_process(warped)
    # y_count_flag = 1

    #抓棋子-----------------------------------------------
    # if y_count_flag == 0:
    #     err_x = target_x_err(1, 1, sorted_white_centers_current, sorted_black_centers_current, cx_dingwei_current)
    #     if err_x<=8 and err_x>= -8 :
    #         count_flag +=1
    #         if count_flag >= 10:
    #             count_flag = 0
    #             y_count_flag = 1
    # else :
    #     count_flag = 0
    #     err_y = target_y_err(1, 1, sorted_white_centers_current, sorted_black_centers_current, cy_dingwei_current)
    #     if err_y<=8 and err_y>= -8 :
    #         count_flag +=1
    #         if count_flag >= 10:
    #             count_flag = 0
    #             catch_qizi_flag = 1
                  #y_count_flag = 0
    # ---------------------------------------------------
    #放棋子-----------------------------------------------

    # catch_qizi_flag = 1
    # if catch_qizi_flag == 1:
    #     #上位机发消息给下位机 抓到棋子
    #     #开始放棋子
    #     # err_x = target_x_err(1, cx_dingwei_current , grid_coordinates)#需要修改为 place——where变量，放在哪个指定的框内
    #     err_x = target_x_err_place_qizi(3,cx_dingwei_current,grid_coordinates)
    #     if err_x<=8 and err_x>= -8 :
    #         count_flag +=1
    #         if count_flag >= 10:
    #             count_flag = 0
    #             y_count_flag = 1

    # 任务四 装置执黑棋先行与人对弈（第 1 步方格可设置），若人应对的第 1 步白棋有错误，装置能获胜。-------------------------------------------------------------------------
    detect_white_piece(sorted_white_centers_current, grid_coordinates)
    move = fourth_task(board)
    # 任务五 人执黑棋先行，装置能正确放置白棋子以保持不输棋。-------------------------------------------------------------------------


    # if button_pressed:#如果收到指令
    #     # # 检测并纠正移动的棋子
    #     # detect_and_correct_moved_piece(white_centers_mid, previous_white_centers_mid)
    #
    #     # 检测黑棋位置
    #     detect_black_piece(sorted_black_centers_current, grid_coordinates)
    #     # 计算装置的下一步棋
    #     move = calculate_white_piece_move(board)
    #     print(move)

        # if move:
        #     row, col = move
        #     next_piece_point = place_white_piece(row, col ,grid_coordinates)
        # winner = check_winner(board)
        # if winner:
        #     print(f"{winner} wins!")
        #     break
        # button_pressed = False

    # 任务六------------------------------------------------------------------------
            #输入目标坐标点，然后抓棋子，放棋子
         # # 重置按钮标志位
        # button_pressed = False




    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
