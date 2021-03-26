import numpy as np
import cv2
import glob
'''
  在这里，我的棋盘格是8*8的，所以角点个数为7*7,当然棋盘格的行列个数可以不一样；
  如果想方便代码改变棋盘格数，是以定义两个变量w（列角点数）和h（行角点数）
'''

# 终止标准
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#w = 7
#h = 7
#准备对象点，如（0,0,0），（1,0,0），（2,0,0）......，（6,5,0）
#objp = np.zeros((w*h,3), np.float32)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# 用于存储所有图像中的对象点和图像点的数组。
objpoints = [] # 在现实世界空间的3d点
imgpoints = [] # 图像平面中的2d点。
#glob是个文件名管理工具
images = glob.glob('F:/all/cam/test5/pic/pic/*.png')
print('...loading')
for fname in images:
    #对每张图片，识别出角点，记录世界物体坐标和图像坐标
    print(f'processing img:{fname}')
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#转灰度
    print('grayed')
    #寻找角点，存入corners，ret是找到角点的flag
    #ret, corners = cv2.findChessboardCorners(gray, (w, h),None)
    ret, corners = cv2.findChessboardCorners(gray, (9, 6),None)

    # 如果找到，添加对象点，图像点（精炼后）
    if ret == True:
        print('chessboard detected')
        objpoints.append(objp)
        #执行亚像素级角点检测
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # 绘制并显示角点
        #img = cv2.drawChessboardCorners(img, (w,h), corners2,ret)
        img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)
        cv2.namedWindow('img',0)
        cv2.resizeWindow('img', 500, 500)
        cv2.imshow('img',img)
        cv2.waitKey(500)
        cv2.destroyAllWindows()
'''
传入所有图片各自角点的三维、二维坐标，相机标定。
每张图片都有自己的旋转和平移矩阵，但是相机内参和畸变系数只有一组。
mtx，相机内参；dist，畸变系数；revcs，旋转矩阵；tvecs，平移矩阵。
'''

img2 = cv2.imread("F:/all/cam/test5/pic/pic/19.png")
print(f"type objpoints:{objpoints[0].shape}")
print(f"type imgpoints:{imgpoints[0].shape}")

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
h,w = img2.shape[0:2]


'''
优化相机内参（camera matrix），这一步可选。
参数1表示保留所有像素点，同时可能引入黑色像素，
设为0表示尽可能裁剪不想要的像素，这是个scale，0-1都可以取。
'''
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
#纠正畸变
dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)

# 裁剪图像，输出纠正畸变以后的图片
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)

#打印我们要求的两个矩阵参数
print ("newcameramtx外参:\n",newcameramtx)
print ("dist畸变值:\n",dist)
print ("newcameramtx旋转（向量）外参:\n",rvecs)
print ("dist平移（向量）外参:\n",tvecs)
#计算误差
tot_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error

print ("total error: ", tot_error/len(objpoints))

