import cv2

# 定义鼠标回调函数
def show_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        img_copy = param.copy()
        cv2.putText(img_copy, f'({x}, {y})', (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow('Image', img_copy)

def main():
    # 打开摄像头
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 设置鼠标回调函数
        cv2.imshow('Image', frame)
        cv2.setMouseCallback('Image', show_coordinates, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
