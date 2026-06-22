import cv2

# 读取图像
image = cv2.imread(r'PicdataProcessing\image.png')

# 显示图像并获取鼠标点击坐标
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked coordinates: ({x}, {y})")

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)

while True:
    cv2.imshow('Image', image)
    if cv2.waitKey(1) & 0xFF == 27:  # 按ESC退出
        break

cv2.destroyAllWindows()