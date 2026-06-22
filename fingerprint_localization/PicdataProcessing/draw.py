import cv2
import numpy as np

def extract_and_draw_red_coordinates(image_path):
    # 读取图片并转换颜色空间
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 红色的 HSV 范围（可能需根据实际图片微调）
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    mask = cv2.bitwise_or(mask1, mask2)  # 合并红色区域掩码
    
    # 提取红色物体坐标
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coordinates = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            coordinates.append((cX, cY))
    
    # 在原图绘制坐标点（绿色圆点）,并且把坐标画图上
    for x, y in coordinates:
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(img, f"({x},{y})", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1, cv2.LINE_AA)
    
    return img, coordinates

if __name__ == "__main__":
    image_path = r"PicdataProcessing\20250402110730.png"  # 替换为实际图片路径
    drawn_img, red_coords = extract_and_draw_red_coordinates(image_path)
    
    # 显示绘制结果,保存图片
    # cv2.imwrite("dataProcessing\\red_coordinates.png", drawn_img)
    cv2.imshow("Red Coordinates Visualization", drawn_img)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()  # Close all OpenCV windows
    # cv2.destroyAllWindows()
    #把这个二维坐标安装纵坐标递增的顺序排列,如果纵坐标相同 则按照横坐标递增的顺序排列
    red_coords.sort(key=lambda x: (x[1], x[0]))
    print("红色物体二维坐标：", red_coords)