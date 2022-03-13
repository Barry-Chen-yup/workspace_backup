import cv2
import time
vs = cv2.VideoCapture('/workspace/nas-data/Animal/chicken_video/run_video/20211105/rpi5/06-05-05.avi')
# end=time.time()
framess = []
# (grabbed, frame) = vs.read()
i=0
# print(grabbed)
while vs.isOpened():
    grabbed, frame = vs.read()
    framess.append(frame)
#     if i == 100:
    start_time = time.time()
#     frame = cv2.resize(frame, (416,416))
    # blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    #     print(blob.shape)
    # net.setInput(blob)
    # layerOutputs = net.forward(ln)
    #     print(layerOutputs)
cv2.imshow('d', frame)
end_time = time.time()

#     if i == 10:
#         start = time.time()
# #         frame = cv2.resize(frame, (416,416))
#         blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
#         #     print(blob.shape)
#         net.setInput(blob)
#         layerOutputs = net.forward(ln)
#         #     print(layerOutputs)
#         end = time.time()
#     framess.append(frame)
#     i+=1
vs.release()