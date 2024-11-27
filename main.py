# 匹配方式识别数字
import pyautogui
import numpy as np
import json
import cv2
import pickle
import sys
import time
from collections import Counter

template = pickle.load(open('template.pkl', 'rb'))

def recognize_digit(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_ = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    scores = np.zeros(10)
    for number, template_img in template.items():
        score = cv2.matchTemplate(image_, template_img, cv2.TM_CCOEFF)
        scores[int(number)] = np.max(score)
    if np.max(scores) < 200000:
        print('识别出错！')
    return np.argmax(scores)

class Recognizer:
    def __init__(self):
        try:
            self.sqinfo = json.load(open('sqinfo.json','r'))
            print()
            print('从sqinfo.json加载识别模块')
            print(f"左上角方块锚点坐标({self.sqinfo['anchor_x']},{self.sqinfo['anchor_y']})")
            print(f"方块高度{self.sqinfo['hwidth']}, 方块高度间隔{self.sqinfo['hgap']}")
            print(f"方块宽度{self.sqinfo['vwidth']}, 方块宽度间隔{self.sqinfo['vgap']}")
            print()
            return
        except:
            pass
    
    def get_sqinfo(self, image):
        try:
            return self.sqinfo
        except:
            print()
            print('初始化识别模块，请判断定位是否准确')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img1 = cv2.GaussianBlur(gray,(3,3),0)
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
                if 0 <= int(theta*180/np.pi) <= 2 or 178 <= int(theta*180/np.pi) <= 175:
                    horizontal_lines.append(int(x0))
                elif 88 <= int(theta*180/np.pi) <= 92:
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
                anchor_x = horizontal_lines[i]
                break
        for i in range(len(vertical_lines)-1):
            if vertical_lines[i+1] - vertical_lines[i] == vwidth:
                anchor_y = vertical_lines[i]
                break
        self.sqinfo = {
            'anchor_x':anchor_x,
            'anchor_y':anchor_y,
            'hwidth':hwidth,
            'vwidth':vwidth,
            'hgap':hgap,
            'vgap':vgap,
            'h':hgap+hwidth,
            'v':vgap+vwidth
        }
        print(f'左上角方块锚点坐标({anchor_x},{anchor_y})，参考值（20,137）')
        print(f'方块高度{hwidth}, 方块高度间隔{hgap}')
        print(f'方块宽度{vwidth}, 方块宽度间隔{vgap}')
        print('识别信息保存到sqinfo.json')
        print()
        json.dump(self.sqinfo, open('sqinfo.json','w'), indent=2)
        return self.sqinfo

    def crop_region(self, square):
        (x1, y1, x2, y2) = square
        # 通过切片提取矩形区域
        cropped_region = self.image[y1:y2, x1:x2]
        return cropped_region

    def get_matrix(self, image):
        self.image = image
        sqinfo = self.get_sqinfo(image)
        # self.squares = self.find_all_squares() # 寻找所有方块的四角坐标 (x1, y1, x2, y2) 
        squares = []
        for i in range(16):
            for j in range(10):
                squares.append((sqinfo['anchor_x']+j*sqinfo['h'],
                                sqinfo['anchor_y']+i*sqinfo['v'],
                                sqinfo['anchor_x']+sqinfo['hwidth']+j*sqinfo['h'],
                                sqinfo['anchor_y']+sqinfo['vwidth']+i*sqinfo['v']))
        if len(squares)!= 160:
            print(squares)
            print('find squares error!')
            return None, squares
        self.crop_images = list(map(self.crop_region, squares)) # 根据坐标提取每个方块图片
        recognized_digits = list(map(recognize_digit, self.crop_images))  # 多线程识别图片
        self.digits_matrix = []
        for i in range(16):
            self.digits_matrix.append((recognized_digits[i * 10:i * 10 + 10]))
        return self.digits_matrix, squares

class eliminater:
    def __init__(self):
        self.width = 450
        self.height = 844
        if sys.platform[:3] == 'win':
            self.anchor, self.owidth, self.oheight = self.find_win_window()
        elif sys.platform[:6] == 'darwin':
            self.anchor, self.owidth, self.oheight = self.find_mac_window()
        else:
            raise NameError('Unknow system')
        print(f'初始化成功，窗口坐标为：{self.anchor}, 窗口高度为： {self.oheight}, 窗口宽度为：{self.owidth}')
        self.scale = self.owidth / self.width * 1.
        if int(self.scale*self.height) != int(self.oheight):
            delta = int(self.oheight - self.scale*self.height)
            self.anchor = (self.anchor[0], self.anchor[1]+delta)
            self.oheight = int(self.oheight - delta)
            print(f'重新调整窗口坐标为：{self.anchor}, 窗口高度为： {self.oheight}, 窗口宽度为：{self.owidth}')
        self.recoginer = Recognizer()
        self.runtime = 0
        self.thd = 80
        self.terminate = False
    
    def find_win_window(self):
        import pygetwindow as gw
        window = gw.getWindowsWithTitle("开局托儿所")[0]
        anchor = (window.left, window.top)
        return anchor, window.width, window.height

    def find_mac_window(self):
        import Quartz
        all_apps = Quartz.NSWorkspace.sharedWorkspace().runningApplications()
        target_app = None
        for app in all_apps:
            if app.localizedName() == '小程序':
                target_app = app
                break
        if not target_app:
            print("ERROR: NO APP FOUND")
            exit()
        # options = Quartz.kCGWindowListOptionOnScreenOnly
        # options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
        # options = Quartz.kCGWindowListOptionAll
        options = Quartz.kCGWindowListExcludeDesktopElements
        # window_list = Quartz.CGWindowListCopyWindowInfo(options, target_app.processIdentifier())
        window_list = Quartz.CGWindowListCopyWindowInfo(options, target_app.processIdentifier())

        # 通过PID查找窗口 由于微信有多个进程，需要找到PID仅重复一次的进程 即为小程序
        all_pid = []
        for window in window_list:
            if window.get('kCGWindowOwnerName', '') != '微信':
                continue
            window_frame = window.get('kCGWindowBounds', None)
            top = int(window_frame['Y'])
            left = int(window_frame['X'])
            anchor = (left, top)
            if anchor[0] < 10 or anchor[1] <= 100:
                continue
            window_pid = window.get('kCGWindowOwnerPID', '')
            all_pid.append(window_pid)
        target_pid = None
        # 查找PID仅重复一次的进程
        for pid in all_pid:
            if all_pid.count(pid) == 1:
                target_pid = pid
                breakpoint
        if not target_pid:
            print("找不到微信窗口，请确保微信已打开小程序，或将窗口往下拖一点")
            exit()
        for window in window_list:
            window_pid = window.get('kCGWindowOwnerPID', '')
            if window_pid == target_pid:  # Modify the window PID accordingly
                window_frame = window.get('kCGWindowBounds', None)
                if window_frame:
                    left = int(window_frame['X'])
                    top = int(window_frame['Y'])
                    anchor = (left, top)
                    if anchor[0] < 10 or anchor[1] <= 100:
                        continue
                    width = int(window_frame['Width'])
                    height = int(window_frame['Height'])
                    return anchor, width, height
        print("找不到微信窗口，请确保微信已打开小程序")
        exit()

    @property
    def score(self):
        if hasattr(self, 'cal_matrix'):
            return 160 - np.sum(self.cal_matrix.astype(bool))
        else:
            print('未初始化')
            return 0
        
    def shift2pos(self, deltax,deltay):
        x = self.anchor[0] + int(deltax * self.scale)
        y = self.anchor[1] + int(deltay * self.scale)
        return (x,y)
        
    def watchAD(self):
        """
        看广告
        """
        times = int(input('看的次数：'))
        for i in range(times):
            print('开始看广告')
            pos = self.shift2pos(225,510)
            self.activate()
            pyautogui.click(pos[0], pos[1])
            time.sleep(3)
            print('静音')
            pos = self.shift2pos(365,80)
            self.activate()
            pyautogui.click(pos[0], pos[1])
            time.sleep(3)
            pyautogui.click(pos[0], pos[1])
            time.sleep(32)
            print('关闭广告')
            pos = self.shift2pos(410,80)
            self.activate()
            pyautogui.click(pos[0], pos[1])
            time.sleep(3)
        
    def action(self, begin_x, end_x, begin_y, end_y,duration=0.1):
        """
        消除方块
        """
        x1, y1, x2, y2 = self.digit_squares[begin_x * 10 + begin_y]
        x1, y1 = ((x1 + x2) / 2, (y1 + y2) / 2)
        x3, y3, x4, y4 = self.digit_squares[(end_x - 1) * 10 + end_y - 1]
        x2, y2 = ((x3 + x4) / 2, (y3 + y4) / 2)
        # x1 *= self.scale
        # x2 *= self.scale
        # y1 *= self.scale
        # y2 *= self.scale
        mouse = pyautogui.position()
        if self.terminate:
            return
        if mouse[0] < self.anchor[0] or mouse[1] < self.anchor[1]:
            print('移出游戏范围，终止运行')
            self.terminate = True
            return
        pos = self.shift2pos(x1,y1)
        pyautogui.moveTo(pos[0], pos[1])
        pyautogui.mouseDown()
        pos = self.shift2pos(x2,y2)
        pyautogui.moveTo(pos[0], pos[1], duration=duration)
        pyautogui.mouseUp()
        
    def restart(self):
        """
        重启游戏
        """
        # 设置
        self.activate()
        pos = self.shift2pos(40,70)
        pyautogui.click(pos[0], pos[1])
        time.sleep(1)
        # 放弃
        pos = self.shift2pos(225,500)
        pyautogui.click(pos[0], pos[1])
        time.sleep(1)
        # 确定
        pos = self.shift2pos(225,520)
        pyautogui.click(pos[0], pos[1])
        time.sleep(1)
        # 开始游戏
        pos = self.shift2pos(225,780)
        pyautogui.click(pos[0], pos[1])
        time.sleep(2)

    def capture_window(self, record=False):
        """
        窗口截图，record=True时仅保存截图，用于在游戏结束后截图记录
        """
        try:
            time.sleep(1)
            screenshot = pyautogui.screenshot(region=(self.anchor[0], self.anchor[1],
                                                      self.owidth, self.oheight))
            screen = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            self.screenshot = cv2.resize(screen, (self.width,self.height))
            if record:
                cv2.imwrite(f'result_{int(time.time())}.png', screen)
            else:
                cv2.imwrite('shot.png', screen)
                cv2.imwrite('shot_resize.png', self.screenshot)
                return self.screenshot
        except IndexError:
            print("窗口未找到, 请确保窗口未被遮挡")
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
            print('\t匹配模式识别图像中，请耐心等待……')
            matrix, self.digit_squares = self.recoginer.get_matrix(screenshot)
            try:
                assert len(self.digit_squares) == 160
                self.matrix = np.array(matrix).astype(int) 
                assert self.matrix.shape == (16,10)
                with open('shot.txt', 'w') as file:
                    file.write(str(self.matrix))
                return True
            except Exception as e:
                print(e)
                print(matrix)
                print('\t识别错误，尝试重启')
                self.trys += 1
                return False
        else:
            print("截图失败！")
            return False
        
    def activate(self):
        pos = self.shift2pos(225,25)
        pyautogui.click(pos[0], pos[1])
        pyautogui.click(pos[0], pos[1])
        
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
        
    def run(self, once=False):
        """
        运行
        """
        self.thd = int(input('请输入分数阈值（低于该值将自动放弃重开）：'))
        print(f"开始运行...")
        self.trys = 0
        stop = False
        while not stop:
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
                self.activate()
                self.run_strategy(go, action=True)
                self.capture_window(record=True)
                if once:
                    stop = True
                    exit()
                else:
                    time.sleep(100)
                    # 点击再次挑战
                    self.activate()
                    pos = self.shift2pos(225,620)
                    pyautogui.click(pos[0], pos[1])
            print(f"游戏{self.runtime}结束, 开始下一次...")
            time.sleep(1)
        
    def cal_all_x(self, End=False, action=False):
        if End or self.terminate:
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
        if End or self.terminate:
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
        if End or self.terminate:
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
        if End or self.terminate:
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
    func = input('选择功能：1、运行(请确保已经正在游戏)；2、看广告：')
    func = int(func)
    if func == 1:
        runner.run(once=True)
    elif func == 2:
        runner.watchAD()
    else:
        print('unknow choice')
