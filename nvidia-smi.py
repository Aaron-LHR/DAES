
import os
import time

while 1:
    os.system('nvidia-smi')
    time.sleep(5)  # 1秒刷新一次
    os.system('cls')  # 这个是清屏操作

