# 匹配方式识别数字
import pyautogui
import numpy as np
import json
import cv2
import pickle
import sys
import time
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

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
        self.sqinfo = None 
        self.sqinfo_path = 'sqinfo.json'
        self._try_load_sqinfo()

    def _try_load_sqinfo(self):
        try:
            with open(self.sqinfo_path, 'r') as f:
                self.sqinfo = json.load(f)
            print(f"成功从 {self.sqinfo_path} 加载定位参数。")
        except FileNotFoundError:
            print(f"{self.sqinfo_path} 未找到。首次运行时将进行定位参数校验。")
            self.sqinfo = None
        except json.JSONDecodeError:
            print(f"错误: {self.sqinfo_path} 文件格式无效。将进行定位参数校验。")
            self.sqinfo = None
        except Exception as e:
            print(f"加载 {self.sqinfo_path} 时发生未知错误: {e}。将进行定位参数校验。")
            self.sqinfo = None

    def _save_sqinfo(self):
        if self.sqinfo:
            try:
                with open(self.sqinfo_path, 'w') as f:
                    json.dump(self.sqinfo, f, indent=4)
                print(f"定位参数已保存到 {self.sqinfo_path}。")
            except IOError:
                print(f"错误：无法保存定位参数到 {self.sqinfo_path}。")
            except Exception as e:
                print(f"保存 {self.sqinfo_path} 时发生未知错误: {e}。")
    
    def get_sqinfo(self, image):
        if self.sqinfo is not None: # Check if loaded from file
            print("使用已加载/已校验的定位参数。")
            required_keys = ['anchor_x', 'anchor_y', 'hwidth', 'vwidth', 'hgap', 'vgap', 'settings_x', 'settings_y']
            valid_load = True
            if not all(key in self.sqinfo for key in required_keys):
                print(f"警告: 从 {self.sqinfo_path} 加载的定位参数不完整 (缺少一个或多个必要键)。将重新进行定位参数校验。")
                self.sqinfo = None 
                valid_load = False
            elif 'h' not in self.sqinfo or 'v' not in self.sqinfo:
                if all(key in self.sqinfo for key in ['hwidth', 'hgap', 'vwidth', 'vgap']):
                    self.sqinfo['h'] = float(self.sqinfo['hwidth']) + float(self.sqinfo['hgap'])
                    self.sqinfo['v'] = float(self.sqinfo['vwidth']) + float(self.sqinfo['vgap'])
                    print(f"已为从 {self.sqinfo_path} 加载的定位参数补充 'h' 和 'v'。")
                else:
                    print(f"警告: 从 {self.sqinfo_path} 加载的定位参数严重不完整。将重新进行定位参数校验。")
                    self.sqinfo = None
                    valid_load = False
            
            if valid_load and self.sqinfo is not None:
                return self.sqinfo

        print('\n初始化识别模块（图形化交互），请在弹出的窗口中拖动线条对齐方块边界。')

        # 图形化交互校准类
        class InteractiveGridCalibration:
            def __init__(self, img):
                self.image = img
                self.fig, self.ax = plt.subplots(figsize=(8, 12))
                self.fig.canvas.manager.set_window_title('方块定位辅助窗口')
                
                # 兼容中文显示
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
                plt.rcParams['axes.unicode_minus'] = False
                
                self.ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                self.ax.set_title("1. 请拖动实线以对齐首尾方块的边界\n2. 请拖动【蓝色★】对齐游戏左上角的【设置】按钮\n(红色最外边界，黄色内外边界，紫色网格实时更新)", pad=20)
                
                img_h, img_w = img.shape[:2]
                
                # 设置初始预估位置
                self.x1_pos, self.x2_pos = img_w * 0.05, img_w * 0.14
                self.x3_pos, self.x4_pos = img_w * 0.86, img_w * 0.95
                self.y1_pos, self.y2_pos = img_h * 0.20, img_h * 0.24
                self.y3_pos, self.y4_pos = img_h * 0.80, img_h * 0.84
                
                # 设置按钮预估位置 (左上角)
                self.settings_marker, = self.ax.plot([img_w * 0.08], [img_h * 0.08], marker='*', color='blue', markersize=20, linestyle='None', picker=5)
                
                # 初始化8根可拖动的实线
                self.lines = {
                    'x1': self.ax.axvline(self.x1_pos, color='r', linewidth=2.5, picker=5),
                    'x2': self.ax.axvline(self.x2_pos, color='y', linewidth=2.5, picker=5),
                    'x3': self.ax.axvline(self.x3_pos, color='y', linewidth=2.5, picker=5),
                    'x4': self.ax.axvline(self.x4_pos, color='r', linewidth=2.5, picker=5),
                    'y1': self.ax.axhline(self.y1_pos, color='r', linewidth=2.5, picker=5),
                    'y2': self.ax.axhline(self.y2_pos, color='y', linewidth=2.5, picker=5),
                    'y3': self.ax.axhline(self.y3_pos, color='y', linewidth=2.5, picker=5),
                    'y4': self.ax.axhline(self.y4_pos, color='r', linewidth=2.5, picker=5),
                }
                
                # 初始化网格虚线(预留够用的数量：水平32条，垂直20条)
                self.v_grid_lines = [self.ax.axvline(0, color='purple', linestyle='--', linewidth=1, alpha=0.7) for _ in range(20)]
                self.h_grid_lines = [self.ax.axhline(0, color='purple', linestyle='--', linewidth=1, alpha=0.7) for _ in range(32)]
                
                self.active_line = None
                self.results = None
                
                self.update_grid()
                
                # 绑定事件
                self.fig.canvas.mpl_connect('button_press_event', self.on_press)
                self.fig.canvas.mpl_connect('button_release_event', self.on_release)
                self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
                
                # 添加确认按钮
                ax_btn = plt.axes([0.8, 0.01, 0.15, 0.05])
                self.btn = Button(ax_btn, '确认 (Confirm)')
                self.btn.on_clicked(self.on_confirm)

            def update_grid(self):
                """根据拖动的边界，实时计算并刷新所有黄色网格的位置"""
                x1, x2, x3, x4 = sorted([self.lines[k].get_xdata()[0] for k in ['x1', 'x2', 'x3', 'x4']])
                y1, y2, y3, y4 = sorted([self.lines[k].get_ydata()[0] for k in ['y1', 'y2', 'y3', 'y4']])
                
                hwidth = ((x2 - x1) + (x4 - x3)) / 2.0
                vwidth = ((y2 - y1) + (y4 - y3)) / 2.0
                
                # 游戏有 10 列 -> 产生 9 个 gap
                hgap = (x4 - x1 - 10 * hwidth) / 9.0 if x4 > x1 else 0
                # 游戏有 16 行 -> 产生 15 个 gap
                vgap = (y4 - y1 - 16 * vwidth) / 15.0 if y4 > y1 else 0
                
                for j in range(10):
                    left = x1 + j * (hwidth + hgap)
                    right = left + hwidth
                    self.v_grid_lines[j*2].set_xdata([left, left])
                    self.v_grid_lines[j*2+1].set_xdata([right, right])
                    
                for i in range(16):
                    top = y1 + i * (vwidth + vgap)
                    bottom = top + vwidth
                    self.h_grid_lines[i*2].set_ydata([top, top])
                    self.h_grid_lines[i*2+1].set_ydata([bottom, bottom])

            def on_press(self, event):
                if event.inaxes != self.ax: return
                
                # 优先检测是否点击了设置图标的蓝色星星
                contains, _ = self.settings_marker.contains(event)
                if contains:
                    self.active_line = 'settings'
                    return
                
                min_dist = float('inf')
                # 找到鼠标按下时离得最近的线条
                for name, line in self.lines.items():
                    contains, _ = line.contains(event)
                    if contains:
                        if name.startswith('x'):
                            dist = abs(line.get_xdata()[0] - event.xdata)
                        else:
                            dist = abs(line.get_ydata()[0] - event.ydata)
                        if dist < min_dist:
                            min_dist = dist
                            self.active_line = name

            def on_motion(self, event):
                if self.active_line is None: return
                if event.inaxes != self.ax: return
                
                if self.active_line == 'settings':
                    self.settings_marker.set_data([event.xdata], [event.ydata])
                    self.fig.canvas.draw_idle()
                    return
                
                line = self.lines[self.active_line]
                if self.active_line.startswith('x'):
                    line.set_xdata([event.xdata, event.xdata])
                else:
                    line.set_ydata([event.ydata, event.ydata])
                    
                self.update_grid()
                self.fig.canvas.draw_idle()

            def on_release(self, event):
                self.active_line = None

            def on_confirm(self, event):
                x1, x2, x3, x4 = sorted([self.lines[k].get_xdata()[0] for k in ['x1', 'x2', 'x3', 'x4']])
                y1, y2, y3, y4 = sorted([self.lines[k].get_ydata()[0] for k in ['y1', 'y2', 'y3', 'y4']])
                
                hwidth = ((x2 - x1) + (x4 - x3)) / 2.0
                vwidth = ((y2 - y1) + (y4 - y3)) / 2.0
                hgap = (x4 - x1 - 10 * hwidth) / 9.0
                vgap = (y4 - y1 - 16 * vwidth) / 15.0
                
                # 改为使用 float 浮点数保存，避免因为四舍五入产生向后递增的累计坐标偏移误差
                self.results = {
                    'anchor_x': float(x1),
                    'anchor_y': float(y1),
                    'hwidth': float(hwidth),
                    'vwidth': float(vwidth),
                    'hgap': float(hgap),
                    'vgap': float(vgap),
                    'h': float(hwidth + hgap),
                    'v': float(vwidth + vgap),
                    'settings_x': float(self.settings_marker.get_xdata()[0]),
                    'settings_y': float(self.settings_marker.get_ydata()[0])
                }
                plt.close(self.fig)

        calibrator = InteractiveGridCalibration(image)
        plt.show() # 此处会阻塞代码，直到点击按钮窗口关闭

        if calibrator.results is None:
            raise ValueError("定位参数校验失败或被强制关闭。")
            
        self.sqinfo = calibrator.results
        self._save_sqinfo()
        print(f"坐标已确认计算结果: {self.sqinfo}")
        return self.sqinfo

    def crop_region(self, square):
        (x1, y1, x2, y2) = square
        # 通过切片提取矩形区域，使用 int 转换浮点数坐标防报错
        cropped_region = self.image[int(y1):int(y2), int(x1):int(x2)]
        return cropped_region

    def get_matrix(self, image):
        self.image = image
        try:
            sqinfo = self.get_sqinfo(image)
        except ValueError as e:
            print(f"Error in get_sqinfo while processing image for get_matrix: {e}")
            return None, None # Indicate failure to determine grid parameters

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
        self.action_mode_active = False
        self.last_mouse_position_set_by_script = None 
    
    def _check_user_mouse_interrupt(self):
        if self.terminate or not self.action_mode_active:
            return False 

        if self.last_mouse_position_set_by_script is None:
            self.last_mouse_position_set_by_script = pyautogui.position()
            return False

        current_pos = pyautogui.position()
        dx = abs(current_pos[0] - self.last_mouse_position_set_by_script[0])
        dy = abs(current_pos[1] - self.last_mouse_position_set_by_script[1])

        if dx > 5 or dy > 5:
            print(f"用户移动了鼠标 (当前: {current_pos}, 脚本上次设置后/初始: {self.last_mouse_position_set_by_script})，操作中止。")
            self.terminate = True
            self.action_mode_active = False 
            return True 
        return False 

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
        options = Quartz.kCGWindowListExcludeDesktopElements
        window_list = Quartz.CGWindowListCopyWindowInfo(options, target_app.processIdentifier())

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
        for pid in all_pid:
            if all_pid.count(pid) == 1:
                target_pid = pid
                break
        if not target_pid:
            print("找不到微信窗口，请确保微信已打开小程序，或将窗口往下拖一点")
            exit()
        for window in window_list:
            window_pid = window.get('kCGWindowOwnerPID', '')
            if window_pid == target_pid:  
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
        
    def shift2pos(self, deltax, deltay):
        # 修复1：采用独立的 x 和 y 轴缩放比例，防止长宽比微小偏差导致的纵轴累计偏移
        scale_x = self.owidth / float(self.width)
        scale_y = self.oheight / float(self.height)
        # 修复2：使用 round 四舍五入，消除原本 int() 暴力截断小数带来的坐标偏移
        x = self.anchor[0] + deltax * scale_x
        y = self.anchor[1] + deltay * scale_y
        return (int(round(x)), int(round(y)))
        
    def watchAD(self):
        times = int(input('看的次数：'))
        original_action_mode = self.action_mode_active 
        self.action_mode_active = False 

        for i in range(times):
            if self.terminate: break
            print('开始看广告')
            pos = self.shift2pos(225,510)
            self.activate()
            if self.terminate: break
            pyautogui.click(pos[0], pos[1])
            time.sleep(3)
            
            print('静音')
            pos = self.shift2pos(365,80)
            self.activate()
            if self.terminate: break
            pyautogui.click(pos[0], pos[1])
            time.sleep(3)
            if self.terminate: break
            pyautogui.click(pos[0], pos[1]) 
            time.sleep(32)

            print('关闭广告')
            pos = self.shift2pos(410,80)
            self.activate()
            if self.terminate: break
            pyautogui.click(pos[0], pos[1])
            time.sleep(3)
        
        self.action_mode_active = original_action_mode

    def action(self, begin_x, end_x, begin_y, end_y, duration=0.15):
        if self.terminate: return
        if self._check_user_mouse_interrupt(): return

        sq1_data = self.digit_squares[begin_x * 10 + begin_y]
        # 准确计算左上方块的几何中心
        center_x1 = (sq1_data[0] + sq1_data[2]) / 2.0
        center_y1 = (sq1_data[1] + sq1_data[3]) / 2.0
        
        sq2_data = self.digit_squares[(end_x - 1) * 10 + end_y - 1]
        # 准确计算右下方块的几何中心
        center_x2 = (sq2_data[0] + sq2_data[2]) / 2.0
        center_y2 = (sq2_data[1] + sq2_data[3]) / 2.0

        screen_pos1 = self.shift2pos(center_x1, center_y1)
        screen_pos2 = self.shift2pos(center_x2, center_y2)

        if self._check_user_mouse_interrupt(): return
        
        # 修复3：先花费0.1秒“滑”到起点，防止小游戏引擎没捕捉到鼠标瞬间转移后的正确坐标
        pyautogui.moveTo(screen_pos1[0], screen_pos1[1], duration=0.1)
        if self.action_mode_active: self.last_mouse_position_set_by_script = pyautogui.position()
        
        if self.terminate: return 
        if self._check_user_mouse_interrupt(): return
        pyautogui.mouseDown()
        time.sleep(0.05) # 短暂延迟，确保按下事件被捕获生效

        if self.terminate: return
        if self._check_user_mouse_interrupt(): return
        # 拖动到目标点
        pyautogui.moveTo(screen_pos2[0], screen_pos2[1], duration=duration)
        if self.action_mode_active: self.last_mouse_position_set_by_script = pyautogui.position()

        if self.terminate: return
        if self._check_user_mouse_interrupt(): return
        time.sleep(0.05) # 拖动后略停顿，防止因运动惯性导致错位
        pyautogui.mouseUp()
        
    def restart(self):
        if self.terminate: return
        self.activate() 
        if self.terminate: return

        # 1. 点击设置按钮（使用保存的动态坐标）
        sx = self.recoginer.sqinfo.get('settings_x', 40)
        sy = self.recoginer.sqinfo.get('settings_y', 75)
        pos = self.shift2pos(sx, sy)
        if self._check_user_mouse_interrupt(): return
        pyautogui.click(pos[0], pos[1])
        if self.action_mode_active: self.last_mouse_position_set_by_script = pyautogui.position()
        time.sleep(1)
        if self.terminate: return

        # 2. 点击放弃 (通过纵轴比例计算)
        pos = self.shift2pos(self.width * 0.5, self.height * (500 / 844.0))
        if self._check_user_mouse_interrupt(): return
        pyautogui.click(pos[0], pos[1])
        if self.action_mode_active: self.last_mouse_position_set_by_script = pyautogui.position()
        time.sleep(1)
        if self.terminate: return

        # 3. 点击确定 (通过纵轴比例计算)
        pos = self.shift2pos(self.width * 0.5, self.height * (520 / 844.0))
        if self._check_user_mouse_interrupt(): return
        pyautogui.click(pos[0], pos[1])
        if self.action_mode_active: self.last_mouse_position_set_by_script = pyautogui.position()
        time.sleep(1)
        if self.terminate: return

        # 4. 点击开始游戏 (通过纵轴比例计算)
        pos = self.shift2pos(self.width * 0.5, self.height * (780 / 844.0))
        if self._check_user_mouse_interrupt(): return
        pyautogui.click(pos[0], pos[1])
        if self.action_mode_active: self.last_mouse_position_set_by_script = pyautogui.position()
        time.sleep(2)

    def capture_window(self, record=False):
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
        with open('历史分数.txt', 'a') as file:
            if x[1]==0:
                file.write(f'\n')
            else:
                file.write(f'\t策略{x[0]}{x[1]}: {self.score},')
                
    def init_game(self):
        time.sleep(1)
        print('\t截图中……')
        screenshot = self.capture_window()
        if screenshot is not None:
            print('\t匹配模式识别图像中，请耐心等待……')
            try:
                matrix_data, digit_squares_data = self.recoginer.get_matrix(screenshot)

                if matrix_data is None or digit_squares_data is None:
                    print('\t无法从图像中获取矩阵信息 (get_matrix failed).')
                    self.trys += 1
                    return False

                self.digit_squares = digit_squares_data 
                assert len(self.digit_squares) == 160
                
                self.matrix = np.array(matrix_data).astype(int)
                assert self.matrix.shape == (16,10)

                with open('shot.txt', 'w') as file:
                    file.write(str(self.matrix))
                return True

            except AssertionError as ae: 
                print(f"Assertion Error during game initialization: {ae}")
                if 'matrix_data' in locals(): 
                    print("Matrix data at assertion error:", matrix_data)
                if 'digit_squares_data' in locals():
                    print("Digit squares data at assertion error:", digit_squares_data)
                print('\t识别断言错误，尝试重启')
                self.trys += 1
                return False
            except Exception as e: 
                print(f"Unexpected error during game initialization: {e}")
                print('\t未知识别错误，尝试重启')
                self.trys += 1
                return False
        else:
            print("截图失败！")
            return False
        
    def activate(self):
        if self.terminate: return
        pos = self.shift2pos(225,25)

        if self._check_user_mouse_interrupt(): return
        pyautogui.click(pos[0], pos[1])
        if self.action_mode_active: self.last_mouse_position_set_by_script = pyautogui.position()
        if self.terminate: return 

        if self._check_user_mouse_interrupt(): return
        pyautogui.click(pos[0], pos[1]) 
        if self.action_mode_active: self.last_mouse_position_set_by_script = pyautogui.position()
        
    def run_strategy(self, strategy, action=False):
        original_action_mode = self.action_mode_active
        original_last_mouse_pos = self.last_mouse_position_set_by_script

        if action:
            self.action_mode_active = True
            self.last_mouse_position_set_by_script = pyautogui.position() 
            self.terminate = False 
        else:
            self.action_mode_active = False


        self.cal_matrix = self.matrix.copy()
        if self.terminate: 
            if action: 
                self.action_mode_active = original_action_mode
                self.last_mouse_position_set_by_script = original_last_mouse_pos
            return

        if strategy[0] == 1: 
            self.cal_two_x(action=action)
            if strategy[1] == 1:
                self.cal_all_x(action=action)
            elif strategy[1] == 2:
                self.cal_all_y(action=action)
        elif strategy[0] == 2: 
            self.cal_two_y(action=action)
            if strategy[1] == 1:
                self.cal_all_x(action=action)
            elif strategy[1] == 2:
                self.cal_all_y(action=action)
        elif strategy[0] == 0: 
            if strategy[1] == 1:
                self.cal_all_x(action=action)
            elif strategy[1] == 2:
                self.cal_all_y(action=action)
        elif strategy[0] == 3: 
            if strategy[1] == 1: 
                self.cal_all_x(action=action)
                self.cal_two_x(action=action)
        elif strategy[0] == 4: # 高效策略：贪心面积消除
            if strategy[1] == 1: 
                self.cal_smart(action=action)
        
    def run(self, once=False):
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
            if self.terminate: return
            self.record([0,1])
            maxscore = self.score
            print(f'\t策略1分数:{self.score}')

            self.run_strategy([0,2])
            if self.terminate: return
            self.record([0,2])
            if self.score > maxscore: maxscore = self.score; go = [0,2]
            print(f'\t策略2分数:{self.score}')

            self.run_strategy([1,1])
            if self.terminate: return
            self.record([1,1])
            if self.score > maxscore: maxscore = self.score; go = [1,1]
            print(f'\t策略3分数:{self.score}')

            self.run_strategy([1,2])
            if self.terminate: return
            self.record([1,2])
            if self.score > maxscore: maxscore = self.score; go = [1,2]
            print(f'\t策略4分数:{self.score}')

            self.run_strategy([2,1])
            if self.terminate: return
            self.record([2,1])
            if self.score > maxscore: maxscore = self.score; go = [2,1]
            print(f'\t策略5分数:{self.score}')

            self.run_strategy([2,2])
            if self.terminate: return
            self.record([2,2])
            if self.score > maxscore: maxscore = self.score; go = [2,2]
            print(f'\t策略6分数:{self.score}')

            self.run_strategy([3,1]) 
            if self.terminate: return
            self.record([3,1])
            if self.score > maxscore: maxscore = self.score; go = [3,1]
            print(f'\t策略7分数:{self.score}')
            
            self.run_strategy([4,1]) 
            if self.terminate: return
            self.record([4,1])
            if self.score > maxscore: maxscore = self.score; go = [4,1]
            print(f'\t策略8(高效综合)分数:{self.score}')
            
            self.record([0,0])
            self.trys = 0 

            if self.terminate: 
                print("用户操作导致评分中止")
                return


            if maxscore < self.thd:
                print(f'\t均小于目标{self.thd}，放弃本次')
                was_action_active_for_run = self.action_mode_active
                self.action_mode_active = True 
                if self.last_mouse_position_set_by_script is None: self.last_mouse_position_set_by_script = pyautogui.position()
                self.restart()
                self.action_mode_active = was_action_active_for_run 
                if self.terminate: 
                    print("重启操作被用户中止。")
            else:
                print('\t执行最优策略', go)
                self.activate() 
                if self.terminate:
                    print("激活操作被用户中止。")
                else:
                    self.run_strategy(go, action=True) 
                    if self.terminate:
                        print("最优策略执行被用户中止。")
                    else:
                        self.capture_window(record=True)
                
                if once:
                    stop = True
                    exit()
                else:
                    time.sleep(100) 
                    if self.terminate: 
                        print("操作已中止，不进行再次挑战。")
                    else:
                        self.activate() 
                        if not self.terminate:
                            # 点击再次挑战 (通过纵轴比例计算)
                            pos = self.shift2pos(self.width * 0.5, self.height * (620 / 844.0))
                            if self._check_user_mouse_interrupt(): return 
                            pyautogui.click(pos[0], pos[1])
                            if self.action_mode_active: self.last_mouse_position_set_by_script = pyautogui.position()
            
            if self.terminate:
                 print(f"游戏{self.runtime}因用户操作中止。")
                 stop = True 
            else:
                print(f"游戏{self.runtime}结束, 开始下一次...")
            time.sleep(1)

    def _cal_all_base(self, End=False, action=False, order='x'):
        if End or self.terminate:
            return
        End = True
        dims = [(x, y) for x in range(1, 17) for y in range(1, 11)] if order == 'x' else [(x, y) for y in range(1, 11) for x in range(1, 17)]
        for x_len, y_len in dims:
            for bx in range(16 - x_len + 1):
                for by in range(10 - y_len + 1):
                    if np.sum(self.cal_matrix[bx:bx+x_len, by:by+y_len]) == 10:
                        self.cal_matrix[bx:bx+x_len, by:by+y_len] = 0
                        if action:
                            self.action(bx, bx+x_len, by, by+y_len)
                        End = False
        self._cal_all_base(End=End, action=action, order=order)

    def cal_all_x(self, End=False, action=False):
        self._cal_all_base(End, action, 'x')
            
    def cal_all_y(self, End=False, action=False):
        self._cal_all_base(End, action, 'y')
    
    def _cal_two_base(self, End=False, action=False, order='x'):
        if End or self.terminate:
            return
        End = True
        coords = [(x, y) for x in range(16) for y in range(10)] if order == 'x' else [(x, y) for y in range(10) for x in range(16)]
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        for bx, by in coords:
            val1 = self.cal_matrix[bx, by]
            if val1 == 0: continue
            
            for dx, dy in directions:
                # 修复核心：如果当前方块已经在前面的方向匹配中被消除了，立刻停止其他方向的搜索！
                if self.cal_matrix[bx, by] == 0:
                    break
                    
                nx, ny = bx + dx, by + dy
                while 0 <= nx < 16 and 0 <= ny < 10:
                    val2 = self.cal_matrix[nx, ny]
                    if val2 == 0:
                        nx += dx
                        ny += dy
                        continue
                    elif val1 + val2 == 10:
                        self.cal_matrix[bx, by] = 0
                        self.cal_matrix[nx, ny] = 0
                        if action:
                            self.action(min(bx, nx), max(bx, nx) + 1, min(by, ny), max(by, ny) + 1)
                        End = False
                    break # 找到数字不管是否匹配都停止向该方向继续搜索

        self._cal_two_base(End=End, action=action, order=order)

    def cal_two_x(self, End=False, action=False):
        self._cal_two_base(End, action, 'x')
            
    def cal_two_y(self, End=False, action=False):
        self._cal_two_base(End, action, 'y')

    def cal_smart(self, End=False, action=False):
        """高效策略：贪心算法优先消除占地面积最小的对子和方块组合，避免阻塞"""
        if End or self.terminate:
            return
        moves = []
        
        # 1. 扫描所有和为10的矩形块
        for x_len in range(1, 17):
            for y_len in range(1, 11):
                area = x_len * y_len
                for bx in range(16 - x_len + 1):
                    for by in range(10 - y_len + 1):
                        if np.sum(self.cal_matrix[bx:bx+x_len, by:by+y_len]) == 10:
                            moves.append((area, 'block', (bx, bx+x_len, by, by+y_len)))
                            
        # 2. 扫描所有能直连消除的对子 (只向右、向下扫描即可避免重复找)
        for bx in range(16):
            for by in range(10):
                val1 = self.cal_matrix[bx, by]
                if val1 == 0: continue
                for dx, dy in [(1, 0), (0, 1)]:
                    nx, ny = bx + dx, by + dy
                    while 0 <= nx < 16 and 0 <= ny < 10:
                        val2 = self.cal_matrix[nx, ny]
                        if val2 == 0:
                            nx += dx
                            ny += dy
                            continue
                        elif val1 + val2 == 10:
                            area = (abs(nx - bx) + 1) * (abs(ny - by) + 1)
                            moves.append((area, 'pair', (bx, nx, by, ny)))
                        break
                        
        moves.sort(key=lambda x: x[0]) # 按覆盖面积从小到大排序
        
        executed_any = False
        for area, m_type, coords in moves:
            if m_type == 'block':
                bx, ex, by, ey = coords
                if np.sum(self.cal_matrix[bx:ex, by:ey]) == 10: # 二次检查是否仍有效
                    self.cal_matrix[bx:ex, by:ey] = 0
                    if action:
                        self.action(bx, ex, by, ey)
                    executed_any = True
            elif m_type == 'pair':
                bx, nx, by, ny = coords
                if self.cal_matrix[bx, by] != 0 and self.cal_matrix[nx, ny] != 0 and self.cal_matrix[bx, by] + self.cal_matrix[nx, ny] == 10:
                    self.cal_matrix[bx, by] = 0
                    self.cal_matrix[nx, ny] = 0
                    if action:
                        self.action(min(bx, nx), max(bx, nx) + 1, min(by, ny), max(by, ny) + 1)
                    executed_any = True

        if executed_any:
            self.cal_smart(End=False, action=action)
                
if __name__ == '__main__':
    runner = eliminater()
    runner.run(once=True)
