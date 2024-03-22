# 匹配方式识别数字
import pyautogui
import pygetwindow as gw
import numpy as np
import cv2
import pickle
import time
from multiprocessing import Pool

template = pickle.load(open('template.pkl', 'rb'))

def recognize_digit(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_ = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    height, width = image_.shape
    scores = np.zeros(10)
    for number, template_img in template.items():
        template_img = cv2.resize(template_img, (width, height))
        score = cv2.matchTemplate(image_[:-3,3:], template_img[:-3,3:], cv2.TM_CCOEFF)
        scores[int(number)] = score[0]
    return np.argmax(scores)

def get_intersection(h_line, v_line):
    rho_h, theta_h = h_line
    rho_v, theta_v = v_line
    # 计算交点坐标
    x, y = np.linalg.solve(np.array([[np.cos(theta_h), np.sin(theta_h)],
                                    [np.cos(theta_v), np.sin(theta_v)]]).astype(float),
                        np.array([rho_h, rho_v]).astype(float))
    # 将交点坐标转为整数
    x, y = int(x), int(y)
    return x, y

class Recognizer:
    def __init__(self, thread=1):
        self.thread=thread

    def find_all_squares(self):
        height, width, _ = self.image.shape
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        sharpened = cv2.filter2D(blurred, -1, np.array([[0, -2, 0], [-2, 9, -2], [0, -2, 0]]))  # 强化锐化处理
        edges = cv2.Canny(sharpened, 200, 500)
        # 使用霍夫线变换检测直线
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=175)
        horizontal_lines = []
        vertical_lines = []
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                # 转换为图像上的坐标
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                # 计算直线的角度
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # 根据角度进行分类，阈值可以根据实际情况调整
                if 0 <= abs(angle) <= 2 or 178 <= abs(angle) <= 175:
                    horizontal_lines.append((rho, theta))
                elif 88 <= abs(angle) <= 92:
                    vertical_lines.append((rho, theta))
        # 对横线按照从上到下的顺序排序
        horizontal_lines.sort(key=lambda line: line[0])
        merged_horizontal_lines = []
        merged_vertical_lines = []
        merge_threshold = 3
        previous_line = None
        for current_line in horizontal_lines:
            if previous_line is None or current_line[0] - previous_line[0] > merge_threshold:
                merged_horizontal_lines.append((current_line[0], current_line[1]))
            previous_line = current_line
        # 对竖线按照从左到右的顺序排序
        vertical_lines.sort(key=lambda line: line[0])
        previous_line = None
        for current_line in vertical_lines:
            if previous_line is not None and current_line[0] - previous_line[0] <= merge_threshold:
                # 合并相邻的水平线
                merged_vertical_lines[-1] = (current_line[0], current_line[1])
            else:
                merged_vertical_lines.append((current_line[0], current_line[1]))
            previous_line = current_line
        found_squares = []
        threshold = 3
        # 寻找正方形
        for i, h_line in enumerate(merged_horizontal_lines):
            if i >= len(merged_horizontal_lines)-1:
                break
            next_h_line = merged_horizontal_lines[i+1]
            for j, v_line in enumerate(merged_vertical_lines):
                if j >= len(merged_vertical_lines) - 1:
                    break
                next_v_line = merged_vertical_lines[j+1]
                p_x1, p_y1 = get_intersection(h_line, v_line)
                p_x2, p_y2 = get_intersection(next_h_line, next_v_line)
                is_square = abs(abs(p_x2-p_x1) - abs(p_y2-p_y1)) <= threshold and abs(p_x2-p_x1) > 15 and (p_x2 -p_x1) < width//10
                if is_square:
                    found_squares.append((p_x1, p_y1, p_x2, p_y2))
        return found_squares

    def crop_region(self, square):
        (x1, y1, x2, y2) = square
        # 通过切片提取矩形区域
        cropped_region = self.image[y1:y2, x1:x2]
        return cropped_region

    def get_matrix(self, image):
        self.image = image
        self.squares = self.find_all_squares() # 寻找所有方块的四角坐标 (x1, y1, x2, y2) 
        if len(self.squares)!= 160:
            print('find squares error!')
        self.crop_images = list(map(self.crop_region, self.squares)) # 根据坐标提取每个方块图片
        worker = Pool(self.thread)
        recognized_digits = worker.map(recognize_digit, self.crop_images)  # 多线程识别图片
        worker.close()
        worker.join()
        self.digits_matrix = []
        for i in range(16):
            self.digits_matrix.append((recognized_digits[i * 10:i * 10 + 10]))
        return self.digits_matrix, self.squares

class eliminater:
    def __init__(self, window_title="开局托儿所"):
        # self.window = gw.getWindowsWithTitle(window_title)[0]
        self.recoginer = Recognizer()
        self.width = self.window.width
        self.height = self.window.height
        self.s1list = []
        self.runtime = 0
        self.thread = 3
        self.thd = 80

    def capture_window(self, record=False):
        """
        窗口截图，record=True时仅保存截图，用于在游戏结束后截图记录
        """
        screen = cv2.imread('shot.jpg')
        self.screenshot = screen
        return self.screenshot
                
    def init_game(self):
        """
        初始化矩阵
        """
        time.sleep(1)
        print('\t截图中……')
        screenshot = self.capture_window()
        if screenshot is not None:
            print('\t匹配模式识别图像中，请耐心等待……')
            matrix, self.digit_squares = self.recoginer.get_matrix(screenshot)
            try:
                self.matrix = np.array(matrix).astype(int) 
                assert self.matrix.shape == (16,10)
                return True
            except Exception as e:
                print(e)
                print(matrix)
                print('\t识别错误，尝试重启')
                self.trys += 1
                return False
            time.sleep(3)
        else:
            print("截图失败！")
            return False
        
                
if __name__ == '__main__':
    runner = eliminater()
    runner.init_game()
