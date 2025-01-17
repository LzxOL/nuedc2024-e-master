import serial
import time

def main():
    ser = serial.Serial(
        port='COM9',          # 更改为你的串口端口
        baudrate=115200,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1
    )

    if ser.isOpen():
        print(f"Serial port {ser.port} opened successfully")
    else:
        print(f"Failed to open serial port {ser.port}")
        return

    try:
        while True:
            x_rr = 100
            y_rr = 200
            position_index = 11
            message = f"A{position_index}B"  # 发送给下位机下第几个棋子
            ser.write(message.encode('utf-8'))
            print(f"下位机接收下棋: {position_index}")

            # 等待一段时间
            time.sleep(1)

            # 接收字符
            if ser.in_waiting > 0:
                received = ser.read(ser.in_waiting)  # 读取所有可用字节
                print(f"Received: {received.decode('utf-8')}")
            else:
                print("No data received")

            # 等待一段时间
            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        ser.close()
        print(f"Serial port {ser.port} closed")

if __name__ == "__main__":
    main()
