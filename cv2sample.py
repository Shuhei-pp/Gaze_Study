# coding: utf-8 
import cv2

import matplotlib.pyplot as plt

# 顔分類器読込
cascade_path = "haarcascade_frontalface_default.xml"

img = cv2.imread('sample.jpg')

# グレースケール変換
img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# カスケード検出器の特徴量抽出
cascade = cv2.CascadeClassifier(cascade_path)

# 顔検出実行
facerect = cascade.detectMultiScale(img_gry, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))

# 顔検出の色
rectangle_color = (0, 255, 0)#緑

if len(facerect) > 0:
    for rect in facerect:
        cv2.rectangle(img, tuple(rect[0:2]),tuple(rect[0:2] + rect[2:4]), rectangle_color,thickness=2)

cv2.namedWindow("display", cv2.WINDOW_NORMAL)

# ウインドウ表示
cv2.imshow("display",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
 
