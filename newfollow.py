import cv2 as cv
import os
import numpy as np

import time


# 遍历文件夹函数
def getFileList(dir, Filelist, ext=None):
    """
    获取文件夹及其子文件夹中文件列表
    输入 dir：文件夹根目录
    输入 ext: 扩展名
    返回： 文件路径列表
    """
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)

    return Filelist


def mid(follow, mask, img):
    height = follow.shape[0]  # 输入图像高度
    width = follow.shape[1]  # 输入图像宽度

    half = int(width / 2)  # 输入图像中线

    # 从下往上扫描赛道,最下端取图片中线为分割线
    for y in range(height - 1, -1, -1):

        if y == height - 1:  # 刚开始从底部扫描时
            left = 0
            right = width - 1
            left_scale = 0.5  # 初始赛道追踪范围
            right_scale = 0.5  # 初始赛道追踪范围
        elif left == 0 and right == width - 1:  # 下层没有扫描到赛道时
            left_scale = 0.25  # 赛道追踪范围
            right_scale = 0.25  # 赛道追踪范围
        elif left == 0:  # 仅左下层没有扫描到赛道时
            left_scale = 0.25  # 赛道追踪范围
            right_scale = 0.2  # 赛道追踪范围
        elif right == width - 1:  # 仅右下层没有扫描到赛道时
            left_scale = 0.2  # 赛道追踪范围
            right_scale = 0.25  # 赛道追踪范围
        else:
            left_scale = 0.2  # 赛道追踪范围
            right_scale = 0.2  # 赛道追踪范围

        # 根据下层左线位置和scale，设置左线扫描范围
        left_range = mask[y][max(0, left - int(left_scale * width)):min(left + int(left_scale * width), width - 1)]
        # 根据下层右线位置和scale，设置右线扫描范围
        right_range = mask[y][max(0, right - int(right_scale * width)):min(right + int(right_scale * width), width - 1)]

        # 左侧规定范围内未找到赛道
        if (left_range == np.zeros_like(left_range)).all():
            left = left  # 取图片最左端为左线
        else:
            left = int(
                (max(0, left - int(left_scale * width)) + np.average(
                    np.where(left_range == 255))) * 0.4 + left * 0.6)  # 取左侧规定范围内检测到赛道像素平均位置为左线

        # 右侧规定范围内未找到赛道
        if (right_range == np.zeros_like(right_range)).all():
            right = right  # 取图片最右端为右线
        else:
            right = int(
                (max(0, right - int(right_scale * width)) + np.average(
                    np.where(right_range == 255))) * 0.4 + right * 0.6)  # 取右侧规定范围内检测到赛道像素平均位置为右线

        mid = int((left + right) / 2)  # 计算中点

        # follow[y, mid] = 255  # 画出拟合中线,实际使用时为提高性能可省略
        # img[y, max(0, left - int(left_scale * width)):min(left + int(left_scale * width), width - 1)] = [0, 0, 255]
        # img[y, max(0, right - int(right_scale * width)):min(right + int(right_scale * width), width - 1)] = [0, 0, 255]

        if y == int((360 / 480) * follow.shape[0]):  # 设置指定提取中点的纵轴位置
            mid_output = mid
    cv.circle(follow, (mid_output, int((360 / 480) * follow.shape[0])), 5, 255, -1)  # opencv为(x,y),画出指定提取中点

    error = (half - mid_output) / width * 640  # 计算图片中点与指定提取中点的误差

    return follow, error, img  # error为正数右转,为负数左转


n = -1
# 存放图片的文件夹路径
path = "./d1"
imglist = getFileList(path, [])
for imgpath in imglist:
    n += 1
    if n < 0:
        continue

    start_time = time.time()

    img = cv.imread(imgpath)

    img = cv.resize(img, (100, 100))

    # HSV阈值分割
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(img_hsv, np.array([43, 60, 90]), np.array([62, 255, 255]))

    follow = mask.copy()

    follow, error, img = mid(follow, mask, img)

    print(n, f"error:{error}")
    end_time = time.time()
    print("time:", end_time - start_time, "s")
    cv.imshow("img", img)
    cv.imshow("mask", mask)
    cv.imshow("follow", follow)
    cv.waitKey(0)

cv.destroyAllWindows()
