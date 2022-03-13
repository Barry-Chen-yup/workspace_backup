from sort import *
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import glob
import datetime
import csv
import json
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import math
import re
import pandas as pd
import pymysql
from datetime import datetime, date, timedelta

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



memory = {}
counter = 0

# construct the argument parse and parse the arguments
#os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
parser = argparse.ArgumentParser()
parser.add_argument("input_video_dir")
args = parser.parse_args()
input_video_dir = args.input_video_dir


dt_confidence = 0.15
nms_threshold = 0.3
pixel2mm_1920 = 3.416
pixel2mm_1280 = 3.48*2
pixel2mm_640 = 6.521*6


def get_dist(box1, box2):

    ctr_1 = [(box1[0]+box1[2])/2, (box1[1]+box1[3])/2]
    ctr_2 = [(box2[0]+box2[2])/2, (box2[1]+box2[3])/2]
    dist = np.linalg.norm(np.subtract(ctr_1, ctr_2))

    # print(dist)

    return dist

def cal_movement_hist(tracked, width1, fps_vs):

    movements = []
    if width1==1920:
        pixel2mm = pixel2mm_1920
    elif width1 == 1280:
        pixel2mm = pixel2mm_1280
    elif width1 == 640:
        pixel2mm = pixel2mm_640
    else:
        pixel2mm = 0
    print('resolution: ', pixel2mm)
    for ids in tracked:

        frames = len(tracked[ids])
        total_dist = 0
        first_box = tracked[ids][0]

        for box in tracked[ids]:
            dist = get_dist(first_box, box)
            total_dist += dist
            first_box = box
        movements.append((fps_vs * pixel2mm * total_dist) / frames)
    return movements
def cal_NNI(det, width1, height1):
    #print(width1, height1)
    area = width1*height1
    num = len(det)
    nbrs = NearestNeighbors(n_neighbors=2).fit(det)
    dist, index = nbrs.kneighbors(det)
    d_obs = sum(dist)[1]/float(num)
    d_exp = 0.5/math.sqrt(num/float(area))
    NNI = d_obs/d_exp

    return NNI

print("[INFO] loading YOLO from disk...")


videos = {}


# for date
for rpi in os.listdir(input_video_dir):
    rpi_path = os.path.join(input_video_dir, rpi)
    if os.path.isdir(rpi_path):
        for video in os.listdir(rpi_path):
            video_path = os.path.join(rpi_path, video)
            if rpi in videos:
                videos[rpi].append(video_path)
            else:
                videos[rpi] = [video_path]
vid_date = os.path.basename(args.input_video_dir)
print('v', vid_date)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.set_option('colheader_justify', 'centre')
# try:
#     conn.ping()
#     print('First Connect success!')
# except:
#     print('First Connect fail!')
#     conn = pymysql.connect(host='top-view-database.barrysmall-personal', user='chicken', password='chicken_mysql', db='chicken_chart')
#     cur = conn.cursor()
#     print('First Connect success!')
# for rpi in videos:
#     rpi_num = re.search(r"\d+(\.\d+)?", rpi)
#     print(rpi)
#     try:
#         conn.ping()
#         print('2nd Connect success!')
#     except:
#         print('2nd Connect fail!')
#         conn = pymysql.connect(host='top-view-database.barrysmall-personal', user='chicken', password='chicken_mysql', db='chicken_chart')
#         cur = conn.cursor()
#         print('2nd Connect success!')
#     fake_time_str = vid_date + ' ' +'05:50:00'
#     fake_time = datetime.strptime(fake_time_str, '%Y%m%d %H:%M:%S')
#     insert_sql = 'INSERT INTO data(id,time,movement_avg,movement_std,NNI_avg,NNI_std) VALUES({}, %s, 0 , 0, 0, 0)'.format(int(rpi_num.group(0)))
#     cur.execute(insert_sql, [str(fake_time)])
#     conn.commit()
#     fake_time_str2 = vid_date + ' ' +'18:50:00'
#     fake_time2 = datetime.strptime(fake_time_str2, '%Y%m%d %H:%M:%S')
#     insert_sql2 = 'INSERT INTO data(id,time,movement_avg,movement_std,NNI_avg,NNI_std) VALUES({}, %s, 0, 0, 0, 0)'.format(int(rpi_num.group(0)))
#     cur.execute(insert_sql, [str(fake_time2)])
#     conn.commit()
#     conn.close()
for rpi in videos:
    rpi_num = re.search(r"\d+(\.\d+)?", rpi)
    print(rpi)

    # video_path = os.path.join(input_video_dir, videos)
    #if rpi != 'rpi9':
        #continue

    #if not os.path.isdir("/home/ubuntu/Yolo_chicken_tracker/"+ vid_date + "/" + rpi):
        #os.mkdir("/home/ubuntu/Yolo_chicken_tracker/"+ vid_date + "/" + rpi)
    # if not os.path.isdir("/home/nas/Research_Group/Personal/chenBA/test_video/NNI/"+ vid_date + "/" + rpi):
    #     os.mkdir("/home/nas/Research_Group/Personal/chenBA/test_video/NNI/"+ vid_date + "/" + rpi)

    for video in videos[rpi]:
        print('video', video)
        tracker = Sort(max_age = 15)
        vs = cv2.VideoCapture(video)
        #save track
        width1=int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
        height1=int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_vs=int(vs.get(cv2.CAP_PROP_FPS))
        #codec=cv2.VideoWriter_fourcc(*'XVID')
        #out1=cv2.VideoWriter('/home/ubuntu/20210219_rpi9_tracking.avi', codec, fps, (width1, height1))
        #
        writer = None
        (W, H) = (None, None)

        frameIndex = 0
        beeIn = 0
        beeOut = 0

        tracked = {}
        movement_hist = []
        avg_movements = []
        avg_NNI = []

        std_movements = []
        std_NNI = []

        NNI = []
        move = {}
        fps = []
        #print(os.path.basename(args.input_video_dir) + "/" + rpi + "/" + os.path.basename(video).split('.')[0] + ".json", 'w')
        print("[INFO] Running {}".format(video))

        # try to determine the total number of frames in the video file
        try:
            prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
                else cv2.CAP_PROP_FRAME_COUNT
            total = int(vs.get(prop))
            print("[INFO] {} total frames in video".format(total))

        # an error occurred while trying to determine the total
        # number of frames in the video file
        except:
            print("[INFO] could not determine # of frames in video")
            print("[INFO] no approx. completion time can be provided")
            total = -1

        # loop over frames from the video file stream
        while True:
            # read the next frame from the file

            (grabbed, frame) = vs.read()
            # if the frame was not grabbed, then we have reached the end
            # of the stream
            if not grabbed:
                break
            #frame = cv2.resize(frame, (640,480))
            # if the frame dimensions are empty, grab them
            if W is None or H is None:
                (H, W) = frame.shape[:2]

            # construct a blob from the input frame and then perform a forward
            # pass of the YOLO object detector, giving us our bounding boxes
            # and associated probabilities
            start = time.time()
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                        swapRB=True, crop=False)
        end = time.time()
        print('Resize time:', 1/(end-start))
        # try:
        #     conn.ping()
        #     print('3rd Connect success!')
        # except:
        #     print('3rd Connect fail!')
        #     conn = pymysql.connect(host='top-view-database.barrysmall-personal', user='chicken', password='chicken_mysql', db='chicken_chart')
        #     cur = conn.cursor()
        #     print('3rd Connect success!')
        # read_hms = os.path.basename(video)
        # today_time = vid_date + ' ' + read_hms[0:8]
        # base_time = datetime.strptime(today_time, '%Y%m%d %H-%M-%S')
        # avg_mov=1;std_mov=2;nni=3;std_nni=4
        # insert_sql = 'INSERT INTO data(id,time,movement_avg,movement_std,NNI_avg,NNI_std) VALUES({},%s,{},{},{},{})'.format(int(rpi_num.group(0)),avg_mov,std_mov,nni,std_nni)
        # cur.execute(insert_sql,[str(base_time)])
        # conn.commit()
        # conn.close()