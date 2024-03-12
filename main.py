# new
import pyautogui
import pygetwindow as gw
import time
import numpy as np
import pytesseract
import os
import cv2
from multiprocessing import Pool

os.environ['TESSDATA_PREFIX'] = r'Tesseract-OCR\tessdata'
pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR\tesseract.exe'

def recognize_digit(image):
    # 预处理图像，例如二值化处理
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    # 使用 pytesseract 进行数字识别
    digit = pytesseract.image_to_string(thresholded, lang='eng', config='--psm 6 digits -c tessedit_char_whitelist=123456789')  # --psm 6 表示按行识别
    return digit.strip()

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
    def __init__(self, tessert_path='Tesseract-OCR', thread=1):
        os.environ['TESSDATA_PREFIX'] =  tessert_path+'\tessdata'
        pytesseract.pytesseract.tesseract_cmd = tessert_path+'\tesseract.exe'
        self.thread=thread

    def find_all_squares(self):
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
                is_square = abs(abs(p_x2-p_x1) - abs(p_y2-p_y1)) <= threshold and abs(p_x2-p_x1) > 15
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
        self.window = gw.getWindowsWithTitle(window_title)[0]
        self.recoginer = Recognizer()
        self.width = self.window.width
        self.height = self.window.height
        self.s1list = []
        self.runtime = 0
        self.thread = 3
        self.thd = 80

    @property
    def score(self):
        if hasattr(self, 'cal_matrix'):
            return 160 - np.sum(self.cal_matrix.astype(bool))
        else:
            print('未初始化')
            return 0
        
    def watchAD(self):
        """
        看广告
        """
        times = int(input('看的次数：'))
        for i in range(times):
            print('开始看广告')
            x = self.window.left + self.window.width//2
            y = self.window.top + 510
            pyautogui.click(x, y)
            time.sleep(3)
            print('静音')
            x = self.window.left + 370
            y = self.window.top + 80
            pyautogui.click(x, y)
            time.sleep(35)
            print('关闭广告')
            x = self.window.left + 420
            y = self.window.top + 80
            pyautogui.click(x, y)
            time.sleep(3)
        
    def action(self, begin_x, end_x, begin_y, end_y,duration=0.1):
        """
        消除方块
        """
        x1, y1, x2, y2 = self.digit_squares[begin_x * 10 + begin_y]
        x1, y1 = ((x1 + x2) / 2, (y1 + y2) / 2)
        x3, y3, x4, y4 = self.digit_squares[(end_x - 1) * 10 + end_y - 1]
        x2, y2 = ((x3 + x4) / 2, (y3 + y4) / 2)
        pyautogui.moveTo(self.window.left + x1, self.window.top+self.height//7 + y1)
        pyautogui.mouseDown()
        pyautogui.moveTo(self.window.left + x2, self.window.top+self.height//7 + y2, duration=duration)
        pyautogui.mouseUp()
        
    def restart(self):
        """
        重启游戏
        """
        # 设置
        x = self.window.left + 40
        y = self.window.top + 70
        pyautogui.click(x, y)
        time.sleep(1)
        # 放弃
        x = self.window.left+ (self.width // 2)
        y = self.window.top + 500
        pyautogui.click(x, y)
        time.sleep(1)
        # 确定
        y = self.window.top + 520
        pyautogui.click(x, y)
        time.sleep(1)
        # 开始游戏
        y = self.window.top + 780
        pyautogui.click(x, y)
        time.sleep(2)

    def capture_window(self, record=False):
        """
        窗口截图，record=True时仅保存截图，用于在游戏结束后截图记录
        """
        try:
            try:
                self.window.activate()
            except:
                pass
            time.sleep(1)
            screenshot = pyautogui.screenshot(region=(self.window.left, self.window.top,
                                                      self.window.width, self.window.height))
            screen = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            if record:
                cv2.imwrite(f'result_{int(time.time())}.png', screen)
            else:
                self.screenshot = screen[self.height//7:-self.height//25,:,:]
                cv2.imwrite('shot.png', self.screenshot)
                return self.screenshot
        except IndexError:
            print("窗口未找到")
            return None
        
    def record(self, x):
        """
        记录分数
        """
        with open('历史分数.txt', 'a') as file:
            if x[1]==0:
                file.write(f'\n')
            else:
                file.write(f'\t策略{x[0]}{x[1]}: {self.score},')
                
    def init_game(self):
        """
        初始化矩阵
        """
        time.sleep(1)
        print('\t截图中……')
        screenshot = self.capture_window()
        if screenshot is not None:
            print('\t识别图像中，请耐心等待……')
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
        
    def run_strategy(self, strategy, action=False):
        self.cal_matrix = self.matrix.copy()
        if strategy[0] == 1:
            self.cal_two_x(action=action)
        elif strategy[0] == 2:
            self.cal_two_y(action=action)
        if strategy[1] == 1:
            self.cal_all_x(action=action)
        elif strategy[1] == 2:
            self.cal_all_y(action=action)
#         return self.score
        
    def run(self):
        """
        运行
        """
        self.thread = int(input('OCR线程数（推荐3）：'))
        self.thd = int(input('请输入分数阈值（低于该值将自动放弃重开）：'))
        print(f"开始运行...")
        self.trys = 0
        while True:
            if self.trys > 5:
                print('错误次数过多，终止运行')
                break
            if not self.init_game():
                self.restart()
                continue
            self.runtime += 1
            print('\t识别完毕，执行策略……')
            go = [0,1]
            self.run_strategy([0,1])
            self.record([0,1])
            maxscore = self.score
            print(f'\t策略1分数:{self.score}')
#                     print('\t识别完毕，执行策略2……')
            self.run_strategy([0,2])
            self.record([0,2])
            if self.score> maxscore:
                maxscore = self.score
                go = [0,2]
            print(f'\t策略2分数:{self.score}')
#                     print('\t识别完毕，执行策略3……')
            self.run_strategy([1,1])
            self.record([1,1])
            if self.score> maxscore:
                maxscore = self.score
                go = [1,1]
            print(f'\t策略3分数:{self.score}')
#                     print('\t识别完毕，执行策略4……')
            self.run_strategy([1,2])
            self.record([1,2])
            if self.score> maxscore:
                maxscore = self.score
                go = [1,2]
            print(f'\t策略4分数:{self.score}')
#                     print('\t识别完毕，执行策略5……')
            self.run_strategy([2,1])
            self.record([2,1])
            if self.score> maxscore:
                maxscore = self.score
                go = [2,1]
            print(f'\t策略5分数:{self.score}')
#                     print('\t识别完毕，执行策略6……')
            self.run_strategy([2,2])
            self.record([2,2])
            if self.score> maxscore:
                maxscore = self.score
                go = [2,2]
            print(f'\t策略6分数:{self.score}')

            self.record([0,0])
            self.trys = 0
            if maxscore < self.thd:
                print(f'\t均小于目标{self.thd}，放弃本次')
                self.restart()
            else:
                print('\t执行最优策略', go)
                self.run_strategy(go, action=True)
                self.capture_window(record=True)
                time.sleep(100)
                # 点击再次挑战
                x = self.window.left + (self.width // 2)
                y = self.window.top + 620
                pyautogui.click(x, y)
            print(f"游戏{self.runtime}结束, 开始下一次...")
            time.sleep(1)
        
    def cal_all_x(self, End=False, action=False):
        if End:
#             if not action:
#                 print(f'\t\t求解任意和后分数：{self.score}')
            return
        else:
            End=True
            for x_len in range(1, 16):
                for y_len in range(1, 10):
                    for begin_x in range(0, 16-x_len+1):
                        for begin_y in range(0, 10-y_len+1):
                            _sum = np.sum(self.cal_matrix[begin_x:begin_x+x_len,begin_y: begin_y + y_len])
                            if _sum == 10:
                                self.cal_matrix[begin_x:begin_x+x_len,begin_y: begin_y + y_len] = 0
                                if action:
                                    self.action(begin_x, begin_x+x_len, begin_y, begin_y + y_len)
                                End = False
            self.cal_all_x(End=End, action=action)
            
    def cal_all_y(self, End=False, action=False):
        if End:
#             if not action:
#                 print(f'\t\t求解任意和后分数：{self.score}')
            return
        else:
            End=True
            for y_len in range(1, 10):
                for x_len in range(1, 16):
                    for begin_x in range(0, 16-x_len+1):
                        for begin_y in range(0, 10-y_len+1):
                            _sum = np.sum(self.cal_matrix[begin_x:begin_x+x_len,begin_y: begin_y + y_len])
                            if _sum == 10:
                                self.cal_matrix[begin_x:begin_x+x_len,begin_y: begin_y + y_len] = 0
                                if action:
                                    self.action(begin_x, begin_x+x_len, begin_y, begin_y + y_len)
                                End = False
            self.cal_all_y(End=End, action=action)
    
    def cal_two_x(self, End=False, action=False):
        if End:
#             if not action:
#                 print(f'\t\t求解两数和后分数：{self.score}')
            return
        else:
            End=True
            for begin_x in range(0, 16):
                for begin_y in range(0, 10):
                    # 搜索右边
                    if self.cal_matrix[begin_x, begin_y] ==0:
                        continue
                    if len(self.s1list) >0 and self.cal_matrix[begin_x, begin_y] not in self.s1list:
                        continue
                    for x in range(begin_x+1, 16):
                        if self.cal_matrix[x, begin_y] ==0:
                            continue
                        elif self.cal_matrix[begin_x, begin_y]+self.cal_matrix[x, begin_y] ==10:
                            self.cal_matrix[x, begin_y] = 0
                            self.cal_matrix[begin_x, begin_y] = 0
                            if action:
                                self.action(begin_x, x+1, begin_y, begin_y+1)
                            End = False
                            break
                        else:
                            break
                    # 搜索左边
                    for x in range(begin_x-1, -1, -1):
                        if x < 0:
                            break
                        if self.cal_matrix[x, begin_y] ==0:
                            continue
                        elif self.cal_matrix[begin_x, begin_y]+self.cal_matrix[x, begin_y] ==10:
                            self.cal_matrix[x, begin_y] = 0
                            self.cal_matrix[begin_x, begin_y] = 0
                            if action:
                                self.action(x, begin_x+1, begin_y, begin_y+1)
                            End = False
                            break
                        else:
                            break
                    # 搜索下面
                    for y in range(begin_y+1, 10):
                        if self.cal_matrix[begin_x, y] ==0:
                            continue
                        elif self.cal_matrix[begin_x, begin_y]+self.cal_matrix[begin_x, y] ==10:
                            self.cal_matrix[begin_x, begin_y] = 0
                            self.cal_matrix[begin_x, y] = 0
                            if action:
                                self.action(begin_x, begin_x+1, begin_y, y+1)
                            End = False
                            break
                        else:
                            break
                    # 搜索上面
                    for y in range(begin_y-1, -1, -1):
                        if y < 0:
                            break
                        if self.cal_matrix[begin_x, y] ==0:
                            continue
                        elif self.cal_matrix[begin_x, begin_y]+self.cal_matrix[begin_x, y] ==10:
                            self.cal_matrix[begin_x, begin_y] = 0
                            self.cal_matrix[begin_x, y] = 0
                            if action:
                                self.action(begin_x, begin_x+1, y, begin_y+1)
                            End = False
                            break
                        else:
                            break
            self.cal_two_x(End=End, action=action)
            
    def cal_two_y(self, End=False, action=False):
        if End:
#             if not action:
#                 print(f'\t\t求解两数和后分数：{self.score}')
            return
        else:
            End=True
            for begin_y in range(0, 10):
                for begin_x in range(0, 16):
                    # 搜索右边
                    if self.cal_matrix[begin_x, begin_y] ==0:
                        continue
                    if len(self.s1list) >0 and self.cal_matrix[begin_x, begin_y] not in self.s1list:
                        continue
                    for x in range(begin_x+1, 16):
                        if self.cal_matrix[x, begin_y] ==0:
                            continue
                        elif self.cal_matrix[begin_x, begin_y]+self.cal_matrix[x, begin_y] ==10:
                            self.cal_matrix[x, begin_y] = 0
                            self.cal_matrix[begin_x, begin_y] = 0
                            if action:
                                self.action(begin_x, x+1, begin_y, begin_y+1)
                            End = False
                            break
                        else:
                            break
                    # 搜索左边
                    for x in range(begin_x-1, -1, -1):
                        if x < 0:
                            break
                        if self.cal_matrix[x, begin_y] ==0:
                            continue
                        elif self.cal_matrix[begin_x, begin_y]+self.cal_matrix[x, begin_y] ==10:
                            self.cal_matrix[x, begin_y] = 0
                            self.cal_matrix[begin_x, begin_y] = 0
                            if action:
                                self.action(x, begin_x+1, begin_y, begin_y+1)
                            End = False
                            break
                        else:
                            break
                    # 搜索下面
                    for y in range(begin_y+1, 10):
                        if self.cal_matrix[begin_x, y] ==0:
                            continue
                        elif self.cal_matrix[begin_x, begin_y]+self.cal_matrix[begin_x, y] ==10:
                            self.cal_matrix[begin_x, begin_y] = 0
                            self.cal_matrix[begin_x, y] = 0
                            if action:
                                self.action(begin_x, begin_x+1, begin_y, y+1)
                            End = False
                            break
                        else:
                            break
                    # 搜索上面
                    for y in range(begin_y-1, -1, -1):
                        if y < 0:
                            break
                        if self.cal_matrix[begin_x, y] ==0:
                            continue
                        elif self.cal_matrix[begin_x, begin_y]+self.cal_matrix[begin_x, y] ==10:
                            self.cal_matrix[begin_x, begin_y] = 0
                            self.cal_matrix[begin_x, y] = 0
                            if action:
                                self.action(begin_x, begin_x+1, y, begin_y+1)
                            End = False
                            break
                        else:
                            break
            self.cal_two_y(End=End, action=action)
                
if __name__ == '__main__':
    runner = eliminater()
    runner.run()
