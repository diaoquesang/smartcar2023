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

#鼠标响应事件
def onmouse(event, x, y, flags, param):
    cv.imshow("img", img)
    # 双击
    if event == cv.EVENT_LBUTTONDBLCLK:
        print("(" + str(x) + "," + str(y) + ")的HSV为" + str(img_hsv[y, x]))
n=0
# 存放图片的文件夹路径
path = "./B"
imglist = getFileList(path, [])
for imgpath in imglist:
    n+=1
    if n<0:
        continue
    img = cv.imread(imgpath)
    print(imgpath)
    img=cv.resize(img,dsize=None,fx=0.5,fy=0.5)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    cv.namedWindow("img")
    cv.namedWindow("imgDivided")
    # mask默认为灰度图

    mask = cv.inRange(img_hsv, np.array([43,60,90]), np.array([62, 255, 255]))
    # 取色器(以下两行仅保留一行注释，若第一行被注释则为快速展示模式，若第二行被注释则为调整阈值模式)
    # cv.setMouseCallback("img", onmouse)
    cv.imshow("img", img)

    div = np.array(img)
    div[mask == 255] = [0, 0, 255]
    cv.imshow("imgDivided", div)
    cv.waitKey(0)
cv.destroyAllWindows()
