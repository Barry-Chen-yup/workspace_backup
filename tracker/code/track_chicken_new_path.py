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

yolo_dir = "./yolo"
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
starts = time.time()
# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([yolo_dir, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3),
                           dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([yolo_dir, "yolov4-tiny-chicken_best.weights"])
configPath = os.path.sep.join([yolo_dir, "yolov4-tiny-chicken.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# print(ln)
# initialize the video stream, pointer to output video file, and
# frame dimensions
# print('a=', os.listdir(input_video_dir))
for date in os.listdir(input_video_dir):
    date_path = os.path.join(input_video_dir, date)
    #print('a', date_path)

    videos = {}
    for rpi in os.listdir(date_path):
        rpi_path = os.path.join(date_path, rpi)
        # print(rpi_path)
        if os.path.isdir(rpi_path):
            for video in os.listdir(rpi_path):
                video_path = os.path.join(rpi_path, video)
                if rpi in videos:
                    videos[rpi].append(video_path)
                else:
                    videos[rpi] = [video_path]
    vid_date = os.path.basename(date_path)
    # print("a", videos)
#if not os.path.isdir("/home/ubuntu/Yolo_chicken_tracker/"+ vid_date):
    #os.mkdir("/home/ubuntu/Yolo_chicken_tracker/"+ vid_date)
    # if not os.path.isdir("/home/nas/Research_Group/Personal/chenBA/test_video/NNI/"+ vid_date):
    #     os.mkdir("/home/nas/Research_Group/Personal/chenBA/test_video/NNI/"+ vid_date)


    for rpi in videos:
        print(rpi)
        # video_path = os.path.join(input_video_dir, videos)
        #if rpi != 'rpi9':
            #continue
    
        #if not os.path.isdir("/home/ubuntu/Yolo_chicken_tracker/"+ vid_date + "/" + rpi):
            #os.mkdir("/home/ubuntu/Yolo_chicken_tracker/"+ vid_date + "/" + rpi)
        # if not os.path.isdir("/home/nas/Research_Group/Personal/chenBA/test_video/NNI/"+ vid_date + "/" + rpi):
        #     os.mkdir("/home/nas/Research_Group/Personal/chenBA/test_video/NNI/"+ vid_date + "/" + rpi)

        for video in videos[rpi]:
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
                blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                            swapRB=True, crop=False)
                #print(blob)
                net.setInput(blob)
                start = time.time()
                layerOutputs = net.forward(ln)
            
                #print(layerOutputs[1])
                # initialize our lists of detected bounding boxes, confidences,
                # and class ID s, respectively
                boxes = []
                confidences = []
                classIDs = []
                # loop over each of the layer outputs
                for output in layerOutputs:
                    # loop over each of the detections
                    for detection in output:
                        # extract the class ID and confidence (i.e., probability)
                        # of the current object detection
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]

                        # filter out weak predictions by ensuring the detected
                        # probability is greater than the minimum probability
                        if confidence > dt_confidence:
                            # scale the bounding box coordinates back relative to
                            # the size of the image, keeping in mind that YOLO
                            # actually returns the center (x, y)-coordinates of
                            # the bounding box followed by the boxes' width and
                            # height
                            box = detection[0:4] * np.array([W, H, W, H])
                            (centerX, centerY, width, height) = box.astype("int")

                            # use the center (x, y)-coordinates to derive the top
                            # and and left corner of the bounding box
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))

                            # update our list of bounding box coordinates,
                            # confidences, and class IDs
                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            classIDs.append(classID)

                # apply non-maxima suppression to suppress weak, overlapping
                # bounding boxes
                idxs = cv2.dnn.NMSBoxes(
                    boxes, confidences, dt_confidence, nms_threshold)

                dets = []
                if len(idxs) > 0:
                    # loop over the indexes we are keeping
                    for i in idxs.flatten():
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                        dets.append([x, y, x+w, y+h, confidences[i]])

                np.set_printoptions(
                    formatter={'float': lambda x: "{0:0.3f}".format(x)})
                dets = np.asarray(dets)
                # print(len(dets))
                if len(dets) > 2:
                    NNI.append(cal_NNI(dets[:, :2], width1, height1))
                    # avg_NNI.append(NNI)
                try:
                    tracks = tracker.update(dets)
                except:
                    continue
                # print(tracks[:5])
                boxes = []
                indexIDs = []
                c = []
                previous = memory.copy()
                memory = {}

                end = time.time()

                for track in tracks:
                    boxes.append([track[0], track[1], track[2], track[3]])
                    indexIDs.append(int(track[4]))
                    memory[indexIDs[-1]] = boxes[-1]

                    ID = str(int(track[4]))
                    if ID in tracked:
                        tracked[ID].append(track[:4])
                    else:
                        tracked[ID] = [track[:4]]

                    if ID in move:
                        move[ID].append(track[:4])
                    else:
                        move[ID] = [track[:4]]

                if len(boxes) > 0:
                    i = int(0)
                    for box in boxes:
                        # extract the bounding box coordinates
                        (x, y) = (int(box[0]), int(box[1]))
                        (w, h) = (int(box[2]), int(box[3]))

                        # draw a bounding box rectangle and label on the image
                        color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
                        cv2.rectangle(frame, (x, y), (w, h), color, 2)

                        if indexIDs[i] in previous:
                            previous_box = previous[indexIDs[i]]
                            (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                            (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                            p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                            p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                            cv2.line(frame, p0, p1, color, 3)
                        # class+confidence
                        # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                        # ID
                        text = "{}".format(indexIDs[i])
                        cv2.putText(frame, text, (x, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        i += 1
                # saves image file
            

                fps.append(round(1/(end-start)))
                # print(fps)
                # cv2.putText(frame, "FPS: " + str(fps), (50,50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 10)

                imS = cv2.resize(frame, (960, 540))

                #save track
                #out1.write(frame)
            
                cv2.imshow("show", imS)
                cv2.waitKey(1)

                # print("[INFO] Fps: ", round(1/(end-start),2))
                # if frameIndex == 0:
                #     break

                # check if the video writer is None
                frameIndex += 1
                # if frameIndex == 50:
                #     break

                # if frameIndex % 1500 == 0:
                #     movement = cal_movement_hist(move, width1, height1)
                #     # print('move:', move)
                #     # print('movement:', movement, len(movement))
                #     #if movement < 0 :
                #         #print('000:',movement)
                #     avg_mov = sum(movement)/len(movement)
                #     movement = [x for x in movement if x < 200 and x != 0]
                #     std_mov = np.std(movement)

                #     # print(movement)

                #     nni = sum(NNI)/1500
                #     std_nni = np.std(NNI)
                #     move = {}
                #     avg_movements.append(avg_mov)
                #     avg_NNI.append(nni)

                #     std_movements.append(std_mov)
                #     std_NNI.append(std_nni)

                #     NNI = []
                #     print(avg_mov, nni, std_mov, std_nni)


            #test
            if int(vs.get(prop)) != 0:
                movement = cal_movement_hist(move, width1, fps_vs)
                movement = [x for x in movement if x < 300 and x != 0]
                avg_mov = sum(movement)/len(movement)
                std_mov = np.std(movement)
            # print(movement)
                nni = sum(NNI)/frameIndex
                std_nni = np.std(NNI)
                move = {}
                avg_movements.append(avg_mov)
                avg_NNI.append(nni)

                std_movements.append(std_mov)
                std_NNI.append(std_nni)

                NNI = []
                ends = time.time()
                print('Code time', ends-starts)
                print('Model predict fps: ', sum(fps)/frameIndex)
                print(avg_mov, nni, std_mov, std_nni, frameIndex)
            #with open(os.path.basename(args.input_video_dir) + "/" + rpi + "/" + os.path.basename(video).split('.')[0] + ".json", 'w') as fp:
                #json.dump(dumped, fp, indent=2)

            #with open(os.path.basename(args.input_video_dir) + "/" + rpi + "/" + os.path.basename(video).split('.')[0] + ".csv", 'w') as f:
                #csv_writer = csv.writer(f, delimiter=',') 
                #for i, x in enumerate(avg_movements):
                    #csv_writer.writerow([5*(i+1), x, std_movements[i], avg_NNI[i], std_NNI[i]])
            # with open("/home/nas/Research_Group/Personal/chenBA/test_video/NNI" + "/" + os.path.basename(date_path) + "/" + rpi + "/" + os.path.basename(video).split('.')[0] + ".csv", 'w') as f:
            #     csv_writer = csv.writer(f, delimiter=',') 
            #     for i, x in enumerate(avg_movements):
            #         csv_writer.writerow([5*(i+1), x, std_movements[i], avg_NNI[i], std_NNI[i]])
            vs.release()

