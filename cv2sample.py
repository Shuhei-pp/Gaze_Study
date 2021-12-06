# coding: utf-8 
import sys
import cv2

import matplotlib.pyplot as plt

# videocaptureのインスタンス作成
cap = cv2.VideoCapture(0)

# カメラ使用できない場合
if cap.isOpened() is False:
    print("カメラが使用できません")
    sys.exit()


# 顔分類器読込
cascade_path = "haarcascade_frontalface_default.xml"

# カスケード検出器の特徴量抽出
cascade = cv2.CascadeClassifier(cascade_path)

while True:
    # videocaptureで1フレーム読み込む
    ret, frame = cap.read()

    # グレースケール変換
    frame_gry = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 顔検出実行
    facerect = cascade.detectMultiScale(frame_gry, scaleFactor=1.1, minNeighbors=2,  minSize=(30, 30))

    # 顔検出の色
    rectangle_color = (0, 255, 0)#緑

    # 枠線描写
    if len(facerect) > 0:
        for rect in facerect:
            cv2.rectangle(frame, tuple(rect[0:2]),tuple(rect[0:2] + rect[2:4]), rectangle_color,thickness=2)

    cv2.namedWindow("display", cv2.WINDOW_NORMAL)

    # ウインドウ表示
    cv2.imshow("display", frame)

    # ESCで終了
    k = cv2.waitKey(1)
    if k == 27: #ESC
        break

cv2.destroyAllWindows()


