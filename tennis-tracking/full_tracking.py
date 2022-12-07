import argparse
import queue
import pandas as pd
import pickle
import os
from PIL import Image, ImageDraw
import cv2
import numpy as np
import sys
import time
import sktime
import numpy

from sklearn.pipeline import Pipeline
from Models.tracknet import trackNet
from utils import get_video_properties, judge_ball
from pickle import load
from detection import *
from court_detector import CourtDetector
from sktime.datatypes._panel._convert import from_2d_array_to_nested
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.transformations.panel.compose import ColumnConcatenator
from data_generation import *

#ML_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# parse parameters
parser = argparse.ArgumentParser()

parser.add_argument("--input_video_path", type=str)
parser.add_argument("--output_video_path", type=str, default="")

args = parser.parse_args()

input_video_path = args.input_video_path
output_video_path = args.output_video_path

n_classes = 256
save_weights_path = 'WeightsTracknet/model.1'

if output_video_path == "":
    # output video in same path
    output_video_path = input_video_path.split('.')[0] + "VideoOutput/video_output.mp4"

# get video fps&video size
video = cv2.VideoCapture(input_video_path)
fps = int(video.get(cv2.CAP_PROP_FPS))
print('fps : {}'.format(fps))
output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# try to determine the total number of frames in the video file
prop = cv2.CAP_PROP_FRAME_COUNT
total = int(video.get(prop))

# start from first frame
currentFrame = 0

# width and height in TrackNet
width, height = 640, 360
img, img1, img2 = None, None, None

# load TrackNet model
modelFN = trackNet
m = modelFN(n_classes, input_height=height, input_width=width)
m.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
m.load_weights(save_weights_path)

# In order to draw the trajectory of tennis, we need to save the coordinate of previous 7 frames
q = queue.deque()
for i in range(0, 8):
    q.appendleft(None)

# save prediction images as videos
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

# court
court_detector = CourtDetector()

# get videos properties
fps, length, v_width, v_height = get_video_properties(video)

coords = []
frame_i = 0
frames = []
t = []
xTopLeft = 0
yTopLeft = 0
xBottomLeft = 0
yBottomLeft = 0
xTopRight = 0
yTopRight= 0
xBottomRight = 0
yBottomRight = 0
xMiddleLeft = 0
yMiddleLeft = 0
xMiddleRight = 0
yMiddleRight = 0

while True:
  ret, frame = video.read()
  frame_i += 1

  if ret:
    if frame_i == 1:
      print('Detecting the court and the players...')
      lines = court_detector.detect(frame)
    else: # then track it
      lines = court_detector.track_court(frame)
      xTopLeft = lines[20]
      yTopLeft = lines[21]
      xBottomLeft = lines[22]
      yBottomLeft = lines[23]
      xTopRight = lines[24]
      yTopRight= lines[25]
      xBottomRight = lines[26]
      yBottomRight = lines[27]
      xMiddleLeft = lines[8]
      yMiddleLeft = lines[9]
      xMiddleRight = lines[10]
      yMiddleRight = lines[11]
    for i in range(0, len(lines), 4):
      x1, y1, x2, y2 = lines[i],lines[i+1], lines[i+2], lines[i+3]
      cv2.line(frame, (int(x1),int(y1)),(int(x2),int(y2)), (0,0,255), 5)
    new_frame = cv2.resize(frame, (v_width, v_height))
    frames.append(new_frame)
    output_video.write(new_frame)
  else:
    break
video.release()
print('FINISHED FIRST STEP')

video = cv2.VideoCapture(input_video_path)
frame_i = 0

last = time.time() # start counting
# while (True):
for img in frames:
    print('Tracking the ball: {}'.format(round( (currentFrame / total) * 100, 2)))
    frame_i += 1

    # detect the ball
    # img is the frame that TrackNet will predict the position
    # since we need to change the size and type of img, copy it to output_img
    output_img = img

    # resize it
    img = cv2.resize(img, (width, height))
    # input must be float type
    img = img.astype(np.float32)

    # since the odering of TrackNet is 'channels_first', so we need to change the axis
    X = np.rollaxis(img, 2, 0)
    # prdict heatmap
    pr = m.predict(np.array([X]))[0]

    # since TrackNet output is ( net_output_height*model_output_width , n_classes )
    # so we need to reshape image as ( net_output_height, model_output_width , n_classes(depth) )
    pr = pr.reshape((height, width, n_classes)).argmax(axis=2)

    # cv2 image must be numpy.uint8, convert numpy.int64 to numpy.uint8
    pr = pr.astype(np.uint8)

    # reshape the image size as original input image
    heatmap = cv2.resize(pr, (output_width, output_height))

    # heatmap is converted into a binary image by threshold method.
    ret, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)

    # find the circle in image with 2<=radius<=7
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2, maxRadius=7)

    PIL_image = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    PIL_image = Image.fromarray(PIL_image)

    # check if there have any tennis be detected
    if circles is not None:
        # if only one tennis be detected
        if len(circles) == 1:

            x = int(circles[0][0][0])
            y = int(circles[0][0][1])

            coords.append([x,y])
            t.append(time.time()-last)

            # push x,y to queue
            q.appendleft([x, y])
            # pop x,y from queue
            q.pop()

        else:
            coords.append(None)
            t.append(time.time()-last)
            # push None to queue
            q.appendleft(None)
            # pop x,y from queue
            q.pop()

    else:
        coords.append(None)
        t.append(time.time()-last)
        # push None to queue
        q.appendleft(None)
        # pop x,y from queue
        q.pop()

    # draw current frame prediction and previous 7 frames as yellow circle, total: 8 frames
    for i in range(0, 8):
        if q[i] is not None:
            draw_x = q[i][0]
            draw_y = q[i][1]
            bbox = (draw_x - 2, draw_y - 2, draw_x + 2, draw_y + 2)
            draw = ImageDraw.Draw(PIL_image)
            draw.ellipse(bbox, outline='yellow')
            del draw

    # Convert PIL image format back to opencv image format
    opencvImage = cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)

    output_video.write(opencvImage)

    # next frame
    currentFrame += 1

# everything is done, release the video
video.release()
output_video.release()


for _ in range(3):
  x, y = diff_xy(coords)
  remove_outliers(x, y, coords)

# interpolation
coords = interpolation(coords)

# velocty
Vx = []
Vy = []
V = []
frames = [*range(len(coords))]

for i in range(len(coords)-1):
  p1 = coords[i]
  p2 = coords[i+1]
  t1 = t[i]
  t2 = t[i+1]
  x = (p1[0]-p2[0])/(t1-t2)
  y = (p1[1]-p2[1])/(t1-t2)
  Vx.append(x)
  Vy.append(y)

for i in range(len(Vx)):
  vx = Vx[i]
  vy = Vy[i]
  v = (vx**2+vy**2)**0.5
  V.append(v)

xy = coords[:]
print('')
print('')
print('Coord:', xy[10:20])
print("nb de frame:",len(frames))
print('')
print('')

########################## MODEL ######################################

# Tracknet data
test_df = pd.DataFrame({'x': [coord[0] for coord in xy[:-1]], 'y':[coord[1] for coord in xy[:-1]], 'V': V})

X_TS = generate_X10(test_df)
X_ML = generate_X_ML(test_df)

# load the pre-trained classifier

model_TS = pickle.load(open('TSFClassifier2.pkl', "rb"))
model_ML = pickle.load(open('SVC2.pkl', "rb"))

# Make predictions

# pred_TS = pd.Series(model_TS.predict(X_TS))
# pred_ML = pd.Series(model_ML.predict(X_ML))
# predictions = pd.concat([pred_TS,pred_ML], axis=1)
# predictions.reset_index(inplace=True)
# predictions.columns = ['index','pred_TS', 'pred_ML']
# predictions.set_index('index', inplace=True)
# predictions['pred_ML'] = predictions['pred_ML'].fillna(value=0)
# print("")
# print("")
# print("")
# print(predictions['pred_ML'])
# print("")
# print("")
# print("")
# print(predictions['pred_TS'])
# print("")
# print("")
# print("")
# print(predictions.columns)
# print("")
# print("")
# print("")
# print(type(predictions['pred_ML']))
# print("")
# print("")
# print("")
# predcted = predictions['pred_TS'].astype(int)&predictions['pred_ML'].astype(int)
# print("")
# print("")
# print("")
# print(predcted.shape)
# print("")
# print("")
# print("")
# Trying to filter "fake" bounces
predcted = model_TS.predict(X_TS)

pred_series = pd.Series(predcted)
data_with_preds = pd.concat([test_df,pred_series],axis=1)
data_with_preds.columns=['x', 'y', 'V', 'bounce']
data_with_preds = data_with_preds.drop(['x', 'V'], axis=1)
filtered_data_with_pred = filter_fake_bounces(data_with_preds)
filtered_pred = filtered_data_with_pred['bounce']
filtered_pred = filtered_pred.to_numpy()

idx = list(np.where(filtered_pred == 1)[0])
#idx = list(np.where(predcted == 1)[0])
idx = np.array(idx) - 10


print('')
print('')
print("predict:",predcted)
print('')
print('')

video = cv2.VideoCapture(output_video_path)

output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# output_video = cv2.VideoCapture(output_video_path)

# output_width = int(output_video.get(cv2.CAP_PROP_FRAME_WIDTH))
# output_height = int(output_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(output_video.get(cv2.CAP_PROP_FPS))
# length = int(output_video.get(cv2.CAP_PROP_FRAME_COUNT))
# fourcc = cv2.VideoWriter_fourcc(*'XVID')

print(fps)
print(length)

output_video = cv2.VideoWriter('VideoOutput/final_video.mp4', fourcc, fps, (output_width, output_height))
i = 0
extremaBottom = [
    [xBottomLeft,yBottomLeft],
    [xBottomRight,yBottomRight],
    [xMiddleRight,yMiddleRight],
    [xMiddleLeft,yMiddleLeft]
]
extremaTop = [
    [xTopLeft,yTopLeft],
    [xTopRight,yTopRight],
    [xMiddleRight,yMiddleRight],
    [xMiddleLeft,yMiddleLeft]
]
score = ''
bounce_count = 0
call = ''
while True:
  ret, frame = video.read()
  fed = cv2.imread('./images/fed-nobg.png')
  nad = cv2.imread('./images/nad-nobg.png')

  scale_percent = 420 # percent of original size
  fed_width = int(fed.shape[1] * scale_percent / 100)
  fed_height = int(fed.shape[0] * scale_percent / 100)
  fed_dim = (fed_width, fed_height)

  # resize image
  federer = cv2.resize(fed, fed_dim, interpolation = cv2.INTER_AREA)

  scale_percent = 420 # percent of original size
  nad_width = int(nad.shape[1] * scale_percent / 100)
  nad_height = int(nad.shape[0] * scale_percent / 100)
  nad_dim = (nad_width, nad_height)

  # resize image
  nadal = cv2.resize(nad, nad_dim, interpolation = cv2.INTER_AREA)

  if ret:
    color_ = (0,128,0)
    cv2.rectangle(frame,(0,0),(580,150),(255,255,255),-1)
    cv2.putText(frame,f'Bounces:{bounce_count}',(10,100),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,0),10,cv2.LINE_AA)
    # cv2.putText(frame,call,(60,100),cv2.FONT_HERSHEY_SIMPLEX,3,color_,10,cv2.LINE_AA)
    if i in idx:
      color = (255, 0, 0)
      thickness = -1
      if i-1 not in idx:
        bounce_count = bounce_count + 1

      ex = extremaBottom if int(xy[i][1]) > yMiddleLeft else extremaTop
      center_coordinates = int(xy[i][0]), int(xy[i][1])
      call = judge_ball(ex, center_coordinates)

      cv2.circle(frame, center_coordinates, 10, color, thickness)
      color_ = (0,128,0) if call == 'IN' else (0, 0, 255)
      img = nadal if call == 'IN' else federer
      cv2.putText(frame,f'Bounces:{bounce_count}',(10,100),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,0),10,cv2.LINE_AA)
      output = merge(frame, img)
      output_video.write(output)
      #cv2.putText(frame,call,(60,100),cv2.FONT_HERSHEY_SIMPLEX,3,color_,10,cv2.LINE_AA)
    elif i-1 in idx or i-2 in idx or i-3 in idx or i-4 in idx or i-5 in idx or i-6 in idx or i-7 in idx or i-8 in idx or i-9 in idx or i-10 in idx:
      color_ = (0,128,0) if call == 'IN' else (0, 0, 255)
      #call = judge_ball(ex, center_coordinates)
      img = nadal if call == 'IN' else federer
      cv2.putText(frame,f'Bounces:{bounce_count}',(10,100),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,0),10,cv2.LINE_AA)
      output = merge(frame, img)
      output_video.write(output)
      #cv2.putText(frame,call,(60,100),cv2.FONT_HERSHEY_SIMPLEX,3,color_,10,cv2.LINE_AA)
    else:
      cv2.putText(frame,f'Bounces:{bounce_count}',(10,100),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,0),10,cv2.LINE_AA)
      output_video.write(frame)
      #call = ''
    i += 1
  else:
    break

video.release()
output_video.release()
