import cv2 as cv
import os
import numpy as np


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


def mid(follow, mask):
    crossroads = False
    halfWidth = follow.shape[1] // 2
    half = halfWidth  # 从下往上扫描赛道,最下端取图片中线为分割线
    for y in range(follow.shape[0] - 1, -1, -1):
        # V2改动:加入分割线左右各半张图片的宽度作为约束,减小邻近赛道的干扰
        if (mask[y][max(0, half - halfWidth):half] == np.zeros_like(
                mask[y][max(0, half - halfWidth):half])).all():  # 分割线左端无赛道
            left = max(0, half - halfWidth)  # 取图片左边界
        else:
            left = np.average(np.where(mask[y][0:half] == 255))  # 计算分割线左端平均位置
        if (mask[y][half:min(follow.shape[1], half + halfWidth)] == np.zeros_like(
                mask[y][half:min(follow.shape[1], half + halfWidth)])).all():  # 分割线右端无赛道
            right = min(follow.shape[1], half + halfWidth)  # 取图片右边界
        else:
            right = np.average(np.where(mask[y][half:follow.shape[1]] == 255)) + half  # 计算分割线右端平均位置

        mid = (left + right) // 2  # 计算拟合中点

        vibra = abs(mid - half)  # 振荡偏差
        # V3改动:检测到异常振荡则判定为十字路口,并保持直行
        # V4改动:设置了检测异常振荡的纵轴位置范围
        if vibra > 30 and y < 430 and y > 50:
            crossroads = True

        mid = int(mid)

        half = mid  # 递归,从下往上确定分割线
        follow[y, mid] = 255  # 画出拟合中线

        if y == 360:  # 设置指定提取中点的纵轴位置
            mid_output = mid
    if crossroads:
        # print("crossroads!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        mid_output = halfWidth
    cv.circle(follow, (mid_output, 360), 5, 255, -1)  # opencv为(x,y),画出指定提取中点

    error = follow.shape[1] // 2 - mid_output  # 计算图片中点与指定提取中点的误差

    return follow, error  # error为正数右转,为负数左转


n = -1
# 存放图片的文件夹路径
path = "./crossroads20230802"
imglist = getFileList(path, [])
for imgpath in imglist:
    n += 1
    if n < 0:
        continue
    img = cv.imread(imgpath)
    img = cv.resize(img, (640, 480))

    # HSV阈值分割
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(img_hsv, np.array([43, 60, 90]), np.array([62, 255, 255]))

    follow = mask.copy()
    follow, error = mid(follow, mask)
    # print(n, f"error:{error}")

    cv.imshow("img", img)
    cv.imshow("mask", mask)
    cv.imshow("follow", follow)
    cv.waitKey(0)

cv.destroyAllWindows()
