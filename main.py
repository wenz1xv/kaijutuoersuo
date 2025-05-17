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
        self.sqinfo = None 
        self.sqinfo_path = 'sqinfo.json'
        self._try_load_sqinfo()
        # print("Recognizer initialized. Attempted to load sqinfo.")

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
            required_keys = ['anchor_x', 'anchor_y', 'hwidth', 'vwidth', 'hgap', 'vgap']
            valid_load = True
            if not all(key in self.sqinfo for key in required_keys):
                print(f"警告: 从 {self.sqinfo_path} 加载的定位参数不完整 (缺少一个或多个必要键)。将重新进行定位参数校验。")
                self.sqinfo = None 
                valid_load = False
            elif 'h' not in self.sqinfo or 'v' not in self.sqinfo:
                # Ensure hwidth, etc. exist before trying to calculate h, v
                if all(key in self.sqinfo for key in ['hwidth', 'hgap', 'vwidth', 'vgap']):
                    self.sqinfo['h'] = self.sqinfo['hwidth'] + self.sqinfo['hgap']
                    self.sqinfo['v'] = self.sqinfo['vwidth'] + self.sqinfo['vgap']
                    print(f"已为从 {self.sqinfo_path} 加载的定位参数补充 'h' 和 'v'。")
                else: # If hwidth etc are missing, then it's definitely incomplete
                    print(f"警告: 从 {self.sqinfo_path} 加载的定位参数严重不完整 (缺少hwidth/vwidth等)。将重新进行定位参数校验。")
                    self.sqinfo = None
                    valid_load = False
            
            if valid_load and self.sqinfo is not None:
                return self.sqinfo

        # If self.sqinfo is None (initial run, file not found, invalid file, or incomplete file)
        print()
        print('初始化识别模块（从图像分析），请判断定位是否准确')

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img1 = cv2.GaussianBlur(gray,(3,3),0)
        edges = cv2.Canny(img1, 50, 150) # TODO: Consider making Canny thresholds adaptive

        # 使用霍夫线变换检测直线
        # TODO: Consider making HoughLines threshold adaptive or test a range
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)

        detected_vertical_x_coords = []   # Store x-coordinates of vertical lines
        detected_horizontal_y_coords = [] # Store y-coordinates of horizontal lines

        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                # if rho < 0: continue # rho can be negative depending on line orientation reference

                a = np.cos(theta)
                b = np.sin(theta)
                # x0 = a * rho
                # y0 = b * rho
                angle_deg = theta * 180 / np.pi

                # Vertical lines (theta near 0 or 180 degrees)
                if (0 <= angle_deg <= 5) or (175 <= angle_deg <= 180):
                    # For vertical line, rho is its x-coordinate if theta is 0 or pi.
                    # Adjust rho to be positive if it's from theta near 180.
                    current_x = abs(rho)
                    if current_x < image.shape[1]: # Check if x-coordinate is within image width
                         detected_vertical_x_coords.append(int(current_x))
                # Horizontal lines (theta near 90 degrees)
                elif (85 <= angle_deg <= 95):
                    # For horizontal line, rho is its y-coordinate if theta is pi/2.
                    current_y = abs(rho)
                    if current_y < image.shape[0]: # Check if y-coordinate is within image height
                        detected_horizontal_y_coords.append(int(current_y))
        
        # 去重和排序
        detected_vertical_x_coords = sorted(list(set(detected_vertical_x_coords)))
        detected_horizontal_y_coords = sorted(list(set(detected_horizontal_y_coords)))

        # Calculate horizontal gaps from detected_vertical_x_coords
        h_gaps_raw = []
        if len(detected_vertical_x_coords) > 1:
            for i in range(len(detected_vertical_x_coords) - 1):
                gap = detected_vertical_x_coords[i+1] - detected_vertical_x_coords[i]
                if gap > 1: # Ignore very small gaps
                    h_gaps_raw.append(gap)

        # Calculate vertical gaps from detected_horizontal_y_coords
        v_gaps_raw = []
        if len(detected_horizontal_y_coords) > 1:
            for i in range(len(detected_horizontal_y_coords) - 1):
                gap = detected_horizontal_y_coords[i+1] - detected_horizontal_y_coords[i]
                if gap > 1: # Ignore very small gaps
                    v_gaps_raw.append(gap)

        if not h_gaps_raw or not v_gaps_raw:
            print("Error: Not enough horizontal or vertical gaps detected after filtering.")
            raise ValueError("Failed to determine grid parameters: Insufficient gap data after filtering.")

        def get_dims_from_gaps(gaps_raw, dim_type="horizontal"):
            if not gaps_raw:
                raise ValueError(f"Failed to determine grid parameters: No positive {dim_type} gaps.")

            cnt = Counter(gaps_raw)
            most_common_gaps = cnt.most_common(2)

            if len(most_common_gaps) < 2:
                if len(most_common_gaps) == 1 and most_common_gaps[0][0] > 0:
                    print(f"Warning: Only one dominant {dim_type} gap size found: {most_common_gaps[0][0]}. Inferring.")
                    width_val = most_common_gaps[0][0]
                    gap_candidates = sorted([g for g in cnt if g > 0 and g < width_val])
                    gap_val = gap_candidates[0] if gap_candidates else max(1, int(width_val * 0.15))
                    return gap_val, width_val
                else:
                    raise ValueError(f"Not enough distinct positive {dim_type} gap sizes. Detected: {most_common_gaps}")
            
            dims = sorted([most_common_gaps[0][0], most_common_gaps[1][0]])
            if dims[0] <= 0:
                 raise ValueError(f"Non-positive or zero gap found in {dim_type} gaps: {dims}")
            
            gap_val = dims[0]
            width_val = dims[1]
            
            # Sanity check: width_val should ideally be larger than gap_val.
            # If they are too close, it might indicate an issue.
            if width_val < gap_val * 1.5: # Heuristic: width should be at least 1.5x gap
                 print(f"Warning: {dim_type} width_val ({width_val}) is not significantly larger than gap_val ({gap_val}). Results may be inaccurate.")

            return gap_val, width_val

        try:
            hgap, hwidth = get_dims_from_gaps(h_gaps_raw, "horizontal")
            vgap, vwidth = get_dims_from_gaps(v_gaps_raw, "vertical")
        except ValueError as e:
            print(f"Error in get_dims_from_gaps: {e}")
            raise

        anchor_x = -1
        potential_anchor_x = []
        if len(detected_vertical_x_coords) >= 1: # Need at least one line
            # Try to find a line that, when considered an anchor, forms a grid consistent with hwidth and hgap
            for i in range(len(detected_vertical_x_coords) -1) : # Iterate up to second to last
                x_coord = detected_vertical_x_coords[i]
                # Check if the next line is at x_coord + hwidth (content block)
                if abs(detected_vertical_x_coords[i+1] - (x_coord + hwidth)) < max(2, hwidth * 0.1): # Allow 10% tolerance or 2px
                    if x_coord < image.shape[1] * 0.3: # Must be in the left 30%
                        potential_anchor_x.append(x_coord)
            
            if potential_anchor_x:
                anchor_x = sorted(potential_anchor_x)[0] # Smallest valid one
            else: # Fallback
                print("Warning: Could not find a clear anchor_x based on hwidth. Using fallback: first plausible line.")
                for x_coord in detected_vertical_x_coords:
                    if x_coord < image.shape[1] * 0.3: # Must be in the left 30%
                        anchor_x = x_coord
                        break
                if anchor_x == -1 and detected_vertical_x_coords: # If still not found, take the very first one
                    anchor_x = detected_vertical_x_coords[0]

        if anchor_x == -1:
            print("Error: Failed to determine anchor_x.")
            raise ValueError("Failed to determine grid parameters: anchor_x not found.")

        anchor_y = -1
        potential_anchor_y = []
        if len(detected_horizontal_y_coords) >= 1:
            for i in range(len(detected_horizontal_y_coords) -1):
                y_coord = detected_horizontal_y_coords[i]
                if abs(detected_horizontal_y_coords[i+1] - (y_coord + vwidth)) < max(2, vwidth * 0.1): # Allow 10% tolerance
                    if y_coord < image.shape[0] * 0.3: # Must be in the top 30%
                        potential_anchor_y.append(y_coord)

            if potential_anchor_y:
                anchor_y = sorted(potential_anchor_y)[0]
            else: # Fallback
                print("Warning: Could not find a clear anchor_y based on vwidth. Using fallback: first plausible line.")
                for y_coord in detected_horizontal_y_coords:
                    if y_coord < image.shape[0] * 0.3: # Must be in the top 30%
                        anchor_y = y_coord
                        break
                if anchor_y == -1 and detected_horizontal_y_coords:
                    anchor_y = detected_horizontal_y_coords[0]
        
        if anchor_y == -1:
            print("Error: Failed to determine anchor_y.")
            raise ValueError("Failed to determine grid parameters: anchor_y not found.")

        if hgap < 0 or vgap < 0: # Should be caught by get_dims_from_gaps
             print(f"Error: Negative gap values hgap={hgap}, vgap={vgap}")
             raise ValueError("Failed to determine valid grid parameters: Negative gaps.")

        initial_detected_sqinfo = {
            'anchor_x': anchor_x,
            'anchor_y': anchor_y,
            'hwidth': hwidth,
            'vwidth': vwidth,
            'hgap': hgap,
            'vgap': vgap
            # 'h' and 'v' will be added by validate_and_adjust_sqinfo
        }
        print(f'自动检测到的初始参数: 左上角方块锚点坐标({initial_detected_sqinfo["anchor_x"]},{initial_detected_sqinfo["anchor_y"]})，参考值（20,137）')
        print(f'方块内容宽度{initial_detected_sqinfo["hwidth"]}, 方块水平间隔{initial_detected_sqinfo["hgap"]}')
        print(f'方块内容高度{initial_detected_sqinfo["vwidth"]}, 方块垂直间隔{initial_detected_sqinfo["vgap"]}')
        print()
        
        validated_sqinfo = self.validate_and_adjust_sqinfo(image, initial_detected_sqinfo)
        
        if validated_sqinfo:
            self.sqinfo = validated_sqinfo
            self._save_sqinfo()
        else:
            # This case should ideally not happen if validate_and_adjust_sqinfo always returns a dict or raises error
            print("错误：定位参数校验未能成功返回有效参数。")
            raise ValueError("定位参数校验失败或被取消。")

        return self.sqinfo

    def _draw_grid_for_validation(self, image_to_draw_on, sqinfo_params):
        """Draws the grid on the image based on sqinfo_params for validation."""
        
        color_content_start = (0, 255, 0)  # Green for start of content block lines
        color_content_end = (0, 165, 255) # Orange for end of content block lines (start of gap)
        thickness = 1

        anchor_x = sqinfo_params['anchor_x']
        anchor_y = sqinfo_params['anchor_y']
        hwidth = sqinfo_params['hwidth']
        hgap = sqinfo_params['hgap']
        vwidth = sqinfo_params['vwidth']
        vgap = sqinfo_params['vgap']

        num_rows = 16
        num_cols = 10
        
        img_height, img_width = image_to_draw_on.shape[:2]

        # Draw vertical lines
        for j in range(num_cols + 1):
            x_start_line = anchor_x + j * (hwidth + hgap)
            if 0 <= x_start_line < img_width:
                 cv2.line(image_to_draw_on, (x_start_line, anchor_y),
                         (x_start_line, min(img_height -1, anchor_y + num_rows * (vwidth + vgap))),
                         color_content_start, thickness)

            if j < num_cols:
                x_content_end_line = anchor_x + j * (hwidth + hgap) + hwidth
                if 0 <= x_content_end_line < img_width:
                    cv2.line(image_to_draw_on, (x_content_end_line, anchor_y),
                             (x_content_end_line, min(img_height -1, anchor_y + num_rows * (vwidth + vgap))),
                             color_content_end, thickness)

        # Draw horizontal lines
        for i in range(num_rows + 1):
            y_start_line = anchor_y + i * (vwidth + vgap)
            if 0 <= y_start_line < img_height:
                cv2.line(image_to_draw_on, (anchor_x, y_start_line),
                         (min(img_width -1, anchor_x + num_cols * (hwidth + hgap)), y_start_line),
                         color_content_start, thickness)

            if i < num_rows:
                y_content_end_line = anchor_y + i * (vwidth + vgap) + vwidth
                if 0 <= y_content_end_line < img_height:
                    cv2.line(image_to_draw_on, (anchor_x, y_content_end_line),
                             (min(img_width -1, anchor_x + num_cols * (hwidth + hgap)), y_content_end_line),
                             color_content_end, thickness)
        
        if 0 <= anchor_x < img_width and 0 <= anchor_y < img_height:
            cv2.circle(image_to_draw_on, (anchor_x, anchor_y), 5, (255, 0, 0), -1) # Blue circle for anchor


    def validate_and_adjust_sqinfo(self, image, initial_sqinfo):
        """Interactively validates and allows adjustment of sqinfo parameters."""
        current_sqinfo = initial_sqinfo.copy()
        
        # Prepare base image for potential saving with grid
        if len(image.shape) == 2 or image.shape[2] == 1:
            display_image_base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            display_image_base = image.copy()

        while True:
            print("\n--- 定位参数校验 ---")
            print(f"当前锚点 (X, Y): ({current_sqinfo['anchor_x']}, {current_sqinfo['anchor_y']})")
            print(f"方块内容宽度 (hwidth): {current_sqinfo['hwidth']}, 水平间隔 (hgap): {current_sqinfo['hgap']}")
            print(f"方块内容高度 (vwidth): {current_sqinfo['vwidth']}, 垂直间隔 (vgap): {current_sqinfo['vgap']}")
            
            correct = input("以上参数是否正确? (y/n): ").lower().strip()
            if correct == 'y':
                current_sqinfo['h'] = current_sqinfo['hwidth'] + current_sqinfo['hgap']
                current_sqinfo['v'] = current_sqinfo['vwidth'] + current_sqinfo['vgap']
                # self.sqinfo = current_sqinfo # Removed: will be set by the caller
                print("参数已确认。")
                return current_sqinfo # Return the validated dict

            if correct != 'n':
                print("无效输入，请输入 'y' 或 'n'。")
                # Optionally, save the image with the current grid for inspection
                save_img_choice = input("是否需要保存当前网格定位图到文件 (validation_grid.png) 以便查看? (y/n): ").lower().strip()
                if save_img_choice == 'y':
                    validation_image_to_save = display_image_base.copy()
                    self._draw_grid_for_validation(validation_image_to_save, current_sqinfo)
                    save_path = "validation_grid.png"
                    cv2.imwrite(save_path, validation_image_to_save)
                    print(f"校验图像已保存到: {save_path}")
                continue # Re-ask if parameters are correct or proceed to adjustment

            # If correct is 'n', proceed to adjustment
            print("\n--- 参数微调 ---")
            print("请输入新的参数值，或直接按 Enter 保留当前值。")
            
            try:
                ax_str = input(f"新锚点 X [{current_sqinfo['anchor_x']}]: ").strip()
                if ax_str: current_sqinfo['anchor_x'] = int(ax_str)
                
                ay_str = input(f"新锚点 Y [{current_sqinfo['anchor_y']}]: ").strip()
                if ay_str: current_sqinfo['anchor_y'] = int(ay_str)

                hw_str = input(f"新内容宽度 hwidth [{current_sqinfo['hwidth']}]: ").strip()
                if hw_str: current_sqinfo['hwidth'] = int(hw_str)

                hg_str = input(f"新水平间隔 hgap [{current_sqinfo['hgap']}]: ").strip()
                if hg_str: current_sqinfo['hgap'] = int(hg_str)

                vw_str = input(f"新内容高度 vwidth [{current_sqinfo['vwidth']}]: ").strip()
                if vw_str: current_sqinfo['vwidth'] = int(vw_str)

                vg_str = input(f"新垂直间隔 vgap [{current_sqinfo['vgap']}]: ").strip()
                if vg_str: current_sqinfo['vgap'] = int(vg_str)
            except ValueError:
                print("输入无效，请输入数字。请重试。")
                continue
    
    def crop_region(self, square):
        (x1, y1, x2, y2) = square
        # 通过切片提取矩形区域
        cropped_region = self.image[y1:y2, x1:x2]
        return cropped_region

    def get_matrix(self, image):
        self.image = image
        try:
            sqinfo = self.get_sqinfo(image)
        except ValueError as e:
            print(f"Error in get_sqinfo while processing image for get_matrix: {e}")
            return None, None # Indicate failure to determine grid parameters

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
        self.action_mode_active = False  # True when script is auto-controlling mouse
        self.last_mouse_position_set_by_script = None  # Expected mouse position
    
    def _check_user_mouse_interrupt(self):
        """Checks for user mouse movement and terminates if detected during action mode."""
        if self.terminate or not self.action_mode_active:
            return False # Already terminated or not in action mode

        if self.last_mouse_position_set_by_script is None:
            # Initialize if called unexpectedly, though it should be set by calling context
            self.last_mouse_position_set_by_script = pyautogui.position()
            return False

        current_pos = pyautogui.position()
        dx = abs(current_pos[0] - self.last_mouse_position_set_by_script[0])
        dy = abs(current_pos[1] - self.last_mouse_position_set_by_script[1])

        # Threshold for detecting user movement (e.g., 5 pixels)
        if dx > 5 or dy > 5:
            print(f"用户移动了鼠标 (当前: {current_pos}, 脚本上次设置后/初始: {self.last_mouse_position_set_by_script})，操作中止。")
            self.terminate = True
            self.action_mode_active = False # Stop further checks for this run
            return True # Interruption detected
        return False # No interruption

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
        # watchAD is manually triggered, so mouse checks might not be desired here
        # unless it's considered part of a broader "auto" context by the user.
        # For now, assuming it's a distinct manual operation.
        original_action_mode = self.action_mode_active # Preserve state
        self.action_mode_active = False # Temporarily disable for this manual op if needed

        for i in range(times):
            if self.terminate: break
            print('开始看广告')
            pos = self.shift2pos(225,510)
            self.activate() # activate will handle its own checks if action_mode_active is true
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
            pyautogui.click(pos[0], pos[1]) # Click again
            time.sleep(32)

            print('关闭广告')
            pos = self.shift2pos(410,80)
            self.activate()
            if self.terminate: break
            pyautogui.click(pos[0], pos[1])
            time.sleep(3)
        
        self.action_mode_active = original_action_mode # Restore state

    def action(self, begin_x, end_x, begin_y, end_y,duration=0.1):
        """
        消除方块
        """
        if self.terminate: return
        if self._check_user_mouse_interrupt(): return

        sq1_data = self.digit_squares[begin_x * 10 + begin_y]
        center_x1, center_y1 = ((sq1_data[0] + sq1_data[2]) / 2, (sq1_data[1] + sq1_data[3]) / 2)
        
        sq2_data = self.digit_squares[(end_x - 1) * 10 + end_y - 1]
        center_x2, center_y2 = ((sq2_data[0] + sq2_data[2]) / 2, (sq2_data[1] + sq2_data[3]) / 2)

        screen_pos1 = self.shift2pos(center_x1, center_y1)
        screen_pos2 = self.shift2pos(center_x2, center_y2)

        # Check mouse out of bounds (original check)
        # current_mouse_before_action = pyautogui.position()
        # if self.action_mode_active and \
        #    (current_mouse_before_action[0] < self.anchor[0] or \
        #     current_mouse_before_action[1] < self.anchor[1]):
        #     print('鼠标移出游戏范围，终止运行 (action pre-check)')
        #     self.terminate = True
        #     return
        # This check is now largely covered by _check_user_mouse_interrupt if mouse moves.
        # If mouse is ALREADY out of bounds before script moves, _check_user_mouse_interrupt
        # might not trigger if last_mouse_position_set_by_script was also out of bounds.
        # Keeping it might be redundant or could be refined. For now, relying on new check.

        if self._check_user_mouse_interrupt(): return
        pyautogui.moveTo(screen_pos1[0], screen_pos1[1])
        if self.action_mode_active: self.last_mouse_position_set_by_script = pyautogui.position()
        
        if self.terminate: return 
        if self._check_user_mouse_interrupt(): return
        pyautogui.mouseDown()
        # mouseDown does not change position, so no update to last_mouse_position_set_by_script

        if self.terminate: return
        if self._check_user_mouse_interrupt(): return
        pyautogui.moveTo(screen_pos2[0], screen_pos2[1], duration=duration)
        if self.action_mode_active: self.last_mouse_position_set_by_script = pyautogui.position()

        if self.terminate: return
        if self._check_user_mouse_interrupt(): return
        pyautogui.mouseUp()
        # mouseUp does not change position
        
    def restart(self):
        """
        重启游戏
        """
        if self.terminate: return
        # Assuming restart can be part of an automated sequence, checks are active if action_mode_active is true.
        
        # 设置
        self.activate() # activate will perform its own checks
        if self.terminate: return

        pos = self.shift2pos(40,75)
        if self._check_user_mouse_interrupt(): return
        pyautogui.click(pos[0], pos[1])
        if self.action_mode_active: self.last_mouse_position_set_by_script = pyautogui.position()
        time.sleep(1)
        if self.terminate: return

        # 放弃
        pos = self.shift2pos(225,500)
        if self._check_user_mouse_interrupt(): return
        pyautogui.click(pos[0], pos[1])
        if self.action_mode_active: self.last_mouse_position_set_by_script = pyautogui.position()
        time.sleep(1)
        if self.terminate: return

        # 确定
        pos = self.shift2pos(225,520)
        if self._check_user_mouse_interrupt(): return
        pyautogui.click(pos[0], pos[1])
        if self.action_mode_active: self.last_mouse_position_set_by_script = pyautogui.position()
        time.sleep(1)
        if self.terminate: return

        # 开始游戏
        pos = self.shift2pos(225,780)
        if self._check_user_mouse_interrupt(): return
        pyautogui.click(pos[0], pos[1])
        if self.action_mode_active: self.last_mouse_position_set_by_script = pyautogui.position()
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
            try:
                matrix_data, digit_squares_data = self.recoginer.get_matrix(screenshot)

                if matrix_data is None or digit_squares_data is None:
                    # This case handles failure from get_matrix (e.g., due to get_sqinfo error)
                    print('\t无法从图像中获取矩阵信息 (get_matrix failed due to get_sqinfo error).')
                    self.trys += 1
                    return False

                # Proceed with assertions and matrix creation
                self.digit_squares = digit_squares_data # Assign to instance variable
                assert len(self.digit_squares) == 160
                
                self.matrix = np.array(matrix_data).astype(int)
                assert self.matrix.shape == (16,10)

                with open('shot.txt', 'w') as file:
                    file.write(str(self.matrix))
                return True

            except AssertionError as ae: # Catch assertion errors specifically
                print(f"Assertion Error during game initialization: {ae}")
                if 'matrix_data' in locals(): # Check if matrix_data was assigned
                    print("Matrix data at assertion error:", matrix_data)
                if 'digit_squares_data' in locals():
                    print("Digit squares data at assertion error:", digit_squares_data)
                print('\t识别断言错误，尝试重启')
                self.trys += 1
                return False
            except Exception as e: # Catch other unexpected errors
                print(f"Unexpected error during game initialization: {e}")
                print('\t未知识别错误，尝试重启')
                self.trys += 1
                return False
        else:
            print("截图失败！")
            return False
        
    def activate(self):
        if self.terminate: return
        # activate is called by other methods; checks depend on self.action_mode_active state.
        pos = self.shift2pos(225,25)

        if self._check_user_mouse_interrupt(): return
        pyautogui.click(pos[0], pos[1])
        if self.action_mode_active: self.last_mouse_position_set_by_script = pyautogui.position()
        if self.terminate: return # Check after first click

        if self._check_user_mouse_interrupt(): return
        pyautogui.click(pos[0], pos[1]) # Second click
        if self.action_mode_active: self.last_mouse_position_set_by_script = pyautogui.position()
        
    def run_strategy(self, strategy, action=False):
        original_action_mode = self.action_mode_active
        original_last_mouse_pos = self.last_mouse_position_set_by_script

        if action:
            self.action_mode_active = True
            # Initialize with current mouse position at the start of an action sequence
            self.last_mouse_position_set_by_script = pyautogui.position() 
            self.terminate = False # Reset terminate flag for this new strategy attempt
        else:
            # If not in action mode for this strategy (e.g. just scoring), disable checks
            self.action_mode_active = False


        self.cal_matrix = self.matrix.copy()
        if self.terminate: # Check if terminate was set by user interaction before strategy starts
            if action: # Only restore if we changed it for action mode
                self.action_mode_active = original_action_mode
                self.last_mouse_position_set_by_script = original_last_mouse_pos
            return

        if strategy[0] == 1: # cal_two_x first
            self.cal_two_x(action=action)
            if strategy[1] == 1:
                self.cal_all_x(action=action)
            elif strategy[1] == 2:
                self.cal_all_y(action=action)
        elif strategy[0] == 2: # cal_two_y first
            self.cal_two_y(action=action)
            if strategy[1] == 1:
                self.cal_all_x(action=action)
            elif strategy[1] == 2:
                self.cal_all_y(action=action)
        elif strategy[0] == 0: # Only cal_all
            if strategy[1] == 1:
                self.cal_all_x(action=action)
            elif strategy[1] == 2:
                self.cal_all_y(action=action)
        elif strategy[0] == 3: # New strategy 7: cal_all_x then cal_two_x
            if strategy[1] == 1: # Represents [3,1]
                self.cal_all_x(action=action)
                self.cal_two_x(action=action)
            # Potentially other [3,X] combinations if needed later
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
            self.run_strategy([0,1]) # This is for scoring, action=False implicitly
            if self.terminate and action: self.action_mode_active = original_action_mode; self.last_mouse_position_set_by_script = original_last_mouse_pos; return
            self.record([0,1])
            maxscore = self.score
            print(f'\t策略1分数:{self.score}')

            self.run_strategy([0,2])
            if self.terminate and action: self.action_mode_active = original_action_mode; self.last_mouse_position_set_by_script = original_last_mouse_pos; return
            self.record([0,2])
            if self.score > maxscore: maxscore = self.score; go = [0,2]
            print(f'\t策略2分数:{self.score}')

            self.run_strategy([1,1])
            if self.terminate and action: self.action_mode_active = original_action_mode; self.last_mouse_position_set_by_script = original_last_mouse_pos; return
            self.record([1,1])
            if self.score > maxscore: maxscore = self.score; go = [1,1]
            print(f'\t策略3分数:{self.score}')

            self.run_strategy([1,2])
            if self.terminate and action: self.action_mode_active = original_action_mode; self.last_mouse_position_set_by_script = original_last_mouse_pos; return
            self.record([1,2])
            if self.score > maxscore: maxscore = self.score; go = [1,2]
            print(f'\t策略4分数:{self.score}')

            self.run_strategy([2,1])
            if self.terminate and action: self.action_mode_active = original_action_mode; self.last_mouse_position_set_by_script = original_last_mouse_pos; return
            self.record([2,1])
            if self.score > maxscore: maxscore = self.score; go = [2,1]
            print(f'\t策略5分数:{self.score}')

            self.run_strategy([2,2])
            if self.terminate and action: self.action_mode_active = original_action_mode; self.last_mouse_position_set_by_script = original_last_mouse_pos; return
            self.record([2,2])
            if self.score > maxscore: maxscore = self.score; go = [2,2]
            print(f'\t策略6分数:{self.score}')

            self.run_strategy([3,1]) # New Strategy 7
            if self.terminate and action: self.action_mode_active = original_action_mode; self.last_mouse_position_set_by_script = original_last_mouse_pos; return
            self.record([3,1])
            if self.score > maxscore: maxscore = self.score; go = [3,1]
            print(f'\t策略7分数:{self.score}')
            
            self.record([0,0])
            self.trys = 0 # Reset trys if all strategies were evaluated

            if self.terminate: # If terminated during scoring strategies
                print("用户操作导致评分中止")
                if action: # Restore if we were in action mode context
                    self.action_mode_active = original_action_mode
                    self.last_mouse_position_set_by_script = original_last_mouse_pos
                # Decide if restart is appropriate or just stop
                # self.restart() # Or maybe not, if user interrupted.
                return


            if maxscore < self.thd:
                print(f'\t均小于目标{self.thd}，放弃本次')
                # For restart, ensure action_mode is handled if it's an auto-sequence
                was_action_active_for_run = self.action_mode_active
                self.action_mode_active = True # Restart is an automated sequence
                if self.last_mouse_position_set_by_script is None: self.last_mouse_position_set_by_script = pyautogui.position()
                self.restart()
                self.action_mode_active = was_action_active_for_run # Restore after restart
                if self.terminate: # If restart was interrupted
                    print("重启操作被用户中止。")
                    # Further handling might be needed here, e.g., exiting the loop
            else:
                print('\t执行最优策略', go)
                # This is the main action execution
                self.activate() # activate will use self.action_mode_active
                if self.terminate:
                    print("激活操作被用户中止。")
                else:
                    self.run_strategy(go, action=True) # This sets action_mode_active internally
                    if self.terminate:
                        print("最优策略执行被用户中止。")
                    else:
                        self.capture_window(record=True)
                
                if once:
                    stop = True
                    exit()
                else:
                    time.sleep(100) # Long pause, user might move mouse here.
                                    # If they do, the next action (e.g. "再次挑战") should be interrupted.
                    if self.terminate: # Check if terminated during the long sleep or previous actions
                        print("操作已中止，不进行再次挑战。")
                    else:
                        # 点击再次挑战
                        self.activate() # activate will handle checks
                        if not self.terminate:
                            pos = self.shift2pos(225,620)
                            if self._check_user_mouse_interrupt(): return # Check before click
                            pyautogui.click(pos[0], pos[1])
                            if self.action_mode_active: self.last_mouse_position_set_by_script = pyautogui.position()
            
            if self.terminate:
                 print(f"游戏{self.runtime}因用户操作中止。")
                 stop = True # Exit the while loop
            else:
                print(f"游戏{self.runtime}结束, 开始下一次...")
            time.sleep(1)
        
        # Restore original action_mode_active state if it was changed by this run method's logic
        # This is tricky because run_strategy also manages it.
        # The run_strategy should handle its own restoration.

    def cal_all_x(self, End=False, action=False):
        if End or self.terminate:
#             if not action:
#                 print(f'\t\t求解任意和后分数：{self.score}')
            return
        else:
            End=True
            # 修正循环范围以匹配16行x10列的矩阵
            # x_len 是块中的行数 (1到16)
            # y_len 是块中的列数 (1到10)
            for x_len in range(1, 17): # 16行, so range(1, 16 + 1)
                for y_len in range(1, 11): # 10列, so range(1, 10 + 1)
                    for begin_x in range(0, 16-x_len+1): # begin_x 是起始行
                        for begin_y in range(0, 10-y_len+1): # begin_y 是起始列
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
            # 修正循环范围以匹配16行x10列的矩阵
            # y_len 是块中的列数 (1到10)
            # x_len 是块中的行数 (1到16)
            for y_len in range(1, 11): # 10列, so range(1, 10 + 1)
                for x_len in range(1, 17): # 16行, so range(1, 16 + 1)
                    for begin_x in range(0, 16-x_len+1): # begin_x 是起始行
                        for begin_y in range(0, 10-y_len+1): # begin_y 是起始列
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
