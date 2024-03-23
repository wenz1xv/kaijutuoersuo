# 定位测试
import cv2
import numpy as np  
from collections import Counter
import pickle
 
template = pickle.load(open('template.pkl', 'rb'))

def crop_region(image, square):
    (x1, y1, x2, y2) = square
    # 通过切片提取矩形区域
    cropped_region = image[y1:y2, x1:x2]
    return cropped_region

def recognize_digit(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_ = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    height, width = image_.shape
    scores = np.zeros(10)
    for number, template_img in template.items():
        template_img = cv2.resize(template_img, (width, height))
        score = cv2.matchTemplate(image_[:-3,3:], template_img[:-3,3:], cv2.TM_CCOEFF)
        scores[int(number)] = score[0]
    if np.max(scores) < 200000:
        print('识别出错！')
    return np.argmax(scores)


img = cv2.imread("shot.png")
original_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# canny(): 边缘检测
img1 = cv2.GaussianBlur(original_img,(3,3),0)
edges = cv2.Canny(img1, 50, 150)
# 使用霍夫线变换检测直线
lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
horizontal_lines = []
vertical_lines = []
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        if rho < 0 :
            continue
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # 根据角度进行分类，阈值可以根据实际情况调整
        if 0 <= int(theta*180/np.pi) <= 1 or 179 <= abs(theta*180/np.pi):
            horizontal_lines.append(int(x0))
        elif 89 <= int(theta*180/np.pi) <= 91:
            vertical_lines.append(int(y0))
# 对横线按照从上到下的顺序排序
horizontal_lines.sort()
vertical_lines.sort()

gaps = []
for i in range(len(horizontal_lines)-1):
    gaps.append(horizontal_lines[i+1] - horizontal_lines[i])
cnt = Counter(gaps)
gaps = [cnt.most_common(2)[0][0], cnt.most_common(2)[1][0]]
hwidth = max(gaps)
hgap = min(gaps)

gaps = []
for i in range(len(vertical_lines)-1):
    gaps.append(vertical_lines[i+1] - vertical_lines[i])
cnt = Counter(gaps)
gaps = [cnt.most_common(2)[0][0], cnt.most_common(2)[1][0]]
vwidth = max(gaps)
vgap = min(gaps)

for i in range(len(horizontal_lines)-1):
    if horizontal_lines[i+1] - horizontal_lines[i] == hwidth:
        print(horizontal_lines[i+1])
        anchor_x = horizontal_lines[i]
        break

print(horizontal_lines)
for i in range(len(vertical_lines)-1):
    if vertical_lines[i+1] - vertical_lines[i] == vwidth:
        anchor_y = vertical_lines[i]
        break
print(f'左上角坐标{anchor_x},{anchor_y}, 方块宽度{vwidth}, 方块间隔{vgap}')

squares = []
for i in range(16):
    for j in range(10):
        squares.append((anchor_x+j*(hwidth+hgap), anchor_y+i*(vwidth+vgap), anchor_x+hwidth+j*(hwidth+hgap), anchor_y+vwidth+i*(vwidth+vgap)))

for i in squares:
    crop_images = crop_region(img, i)
    recognized_digits = recognize_digit(crop_images)
    print(recognized_digits)