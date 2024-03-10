# new
import pyautogui
import pygetwindow as gw
import time
import copy
import numpy as np
from recognizeNum import *

class eliminater:
    def __init__(self, window_title="开局托儿所"):
        self.window = gw.getWindowsWithTitle(window_title)[0]
        self.width = self.window.width
        self.height = self.window.height
        self.s1list = []
        self.runtime = 0
        self.thread = 3
        self.thd = 80
        
    def action(self, begin_x, end_x, begin_y, end_y,duration=0.1):
        x1, y1, x2, y2 = self.digit_squares[begin_x * 10 + begin_y]
        x1, y1 = ((x1 + x2) / 2, (y1 + y2) / 2)
        x3, y3, x4, y4 = self.digit_squares[(end_x - 1) * 10 + end_y - 1]
        x2, y2 = ((x3 + x4) / 2, (y3 + y4) / 2)
        pyautogui.moveTo(self.window.left + x1, self.window.top+self.height//7 + y1)
        pyautogui.mouseDown()
        pyautogui.moveTo(self.window.left + x2, self.window.top+self.height//7 + y2, duration=duration)
        pyautogui.mouseUp()
        
    def restart(self):
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
        
    @property
    def score(self):
        if hasattr(self, 'cal_matrix'):
            return 160 - np.sum(self.cal_matrix.astype(bool))
        else:
            print('未初始化')
            return 0
        
    def record(self, x):
        with open('历史分数.txt', 'a') as file:
            if x[1]==0:
                file.write(f'\n')
            else:
                file.write(f'\t策略{x[0]}{x[1]}: {self.score},')
                
    def init_game(self):
        time.sleep(1)
        print('\t截图中……')
        self.capture_window()
        if self.frame is not None:
            print('\t识别图像中，请耐心等待……')
            matrix, self.digit_squares = recognize_matrix(self.frame, self.thread)
            try:
                self.matrix = np.array(matrix).astype(int) 
                assert self.matrix.shape == (16,10)
                return True
            except Exception as e:
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
        
        
    def run(self, continues=True):
        self.thread = int(input('OCR线程数（推荐3）：'))
        self.thd = int(input('请输入分数阈值（低于该值将自动放弃重开）：'))
        print(f"开始运行...")
        self.trys = 0
        while continues:
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
            print(f"游戏{self.runtime}结束, 开始下一次...")
            # 点击再次挑战
            x = self.window.left + (self.width // 2)
            y = self.window.top + 620
            pyautogui.click(x, y)
            time.sleep(1)
            
    def capture_window(self, record=False):
        try:
            try:
                self.window.activate()
            except:
                pass
            time.sleep(1)
            screenshot = pyautogui.screenshot(region=(self.window.left, self.window.top,
                                                      self.window.width, self.window.height))
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            if record:
                cv2.imwrite(f'result_{int(time.time())}.png', frame)
            else:
                self.frame = frame[self.height//7:-self.height//25,:,:]
                cv2.imwrite('shot.png', self.frame)
        except IndexError:
            print("窗口未找到")
            return None
        
    def cal_recur(self, x_len=1, y_len=1, action=False):
        if x_len>15 or y_len>9:
            return
        else:
            for begin_x in range(0, 16-x_len+1):
                for begin_y in range(0, 10-y_len+1):
                    _sum = np.sum(self.cal_matrix[begin_x:begin_x+x_len,begin_y: begin_y + y_len])
                    if _sum == 10:
                        self.cal_matrix[begin_x:begin_x+x_len,begin_y: begin_y + y_len] = 0
                        if action:
                            self.action(begin_x, begin_x+x_len, begin_y, begin_y + y_len)
            self.cal_recur(x_len=x_len+1, y_len=y_len, action=action)
            self.cal_recur(x_len=x_len, y_len=y_len+1, action=action)
        
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
