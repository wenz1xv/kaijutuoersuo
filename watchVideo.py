import pyautogui
import pygetwindow as gw
import time
import copy
import numpy as np
                 
if __name__ == "__main__":
    window_title = "开局托儿所"
    window = gw.getWindowsWithTitle(window_title)[0]
    screen_width, screen_height = pyautogui.size()
    try:
        window.activate()
    except:
        pass
    times = int(input('看的次数：'))
    for i in range(times):
        print('开始看广告')
        x = window.left + window.width//2
        y = window.top + 510
        pyautogui.click(x, y)
        time.sleep(3)
        print('静音')
        x = window.left + 370
        y = window.top + 80
        pyautogui.click(x, y)
        time.sleep(35)
        print('关闭广告')
        x = window.left + 420
        y = window.top + 80
        pyautogui.click(x, y)
        time.sleep(3)
    
