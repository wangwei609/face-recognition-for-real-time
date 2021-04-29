#2018年完成的
import os
import cv2
import time
import schedule
import threading
import imageio
import numpy as np
import face_recognition
from PIL import Image,ImageDraw,ImageFont
video_capture = cv2.VideoCapture(0)
#通过路径找到文件名字和里面的图片(把图片特征提取).
def read_file(path):
    name_list = []
    dir_counter = 0
    img_encoding = [[] for i in range(10)]
    for child_dir in os.listdir(path):
        name_list.append(child_dir)
        child_path = os.path.join(path, child_dir) 
        for dir_image in  os.listdir(child_path):
            if True in map(dir_image.endswith,'jpg'):
                #img = scipy.misc.imread(os.path.join(child_path, dir_image)) 
                img = imageio.imread(os.path.join(child_path, dir_image)) 
                img_encoding[dir_counter].append(face_recognition.face_encodings(img)[0])
        dir_counter += 1
    fileName = str(int(time.time()))
    os.makedirs(path + "/"+ fileName)
    fileName = fileName + "/"
    return img_encoding,name_list,fileName
#更新工作
def job():
    global count,all_encoding,name_list,fileName
    all_encoding,name_list,fileName = read_file("./dataset")
    count = 0
#拍照控速工作
def job2():
    global flag2
    flag2 = 1
#开新线程
def Rthread(job_func):
     job_thread = threading.Thread(target=job_func)
     job_thread.start()
job()
job2()
flag = 0
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
schedule.every(15).seconds.do(Rthread,job)
schedule.every(1).seconds.do(job2)

while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    if process_this_frame:
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        face_names = []
        #匹配，并赋值
        for face_encoding in face_encodings:
            i = 0
            j = 0
            for t in all_encoding:
                for k in t:
                    match = face_recognition.compare_faces([k], face_encoding,tolerance=0.5)
                    if match[0]:
                        name = name_list[i]
                        j=1
                i = i+1
            if j == 0:
                name = "陌生人"
                schedule.run_pending()
                if flag == 0:
                    flag = 1
            face_names.append(name)
    process_this_frame = not process_this_frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        if flag == 1 and count < 6 and flag2 ==1 : 
            face_image = frame[top:bottom, left:right]
            cv2.imwrite('./dataset/'+ fileName + str(int(time.time()))+'.jpg',face_image)
            flag = 0
            flag2 = 0
            count += 1
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255),  2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), 2)
        cv2img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # cv2和PIL中颜色的hex码的储存顺序不同
        pilimg = Image.fromarray(cv2img)                # PIL图片上打印汉字
        draw = ImageDraw.Draw(pilimg)                   # 图片上打印
        font = ImageFont.truetype("C:/Windows/Fonts/STXINGKA.TTF", 20, encoding="utf-8")    #字体选择
        draw.text((left+50, bottom-26), name, (255, 0, 0), font=font)   # 参数：打印坐标,文本,字体颜色,字体 
        frame = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)       # PIL 转 cv2 
        
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == 27: #按Esc键退出
        break
    
video_capture.release()
cv2.destroyAllWindows()
