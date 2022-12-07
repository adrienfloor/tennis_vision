import argparse
import pandas as pd
import numpy as np
import sys
import sktime

from detection import *
# from sktime.datatypes._panel._convert import from_2d_array_to_nested
# from sktime.classification.interval_based import TimeSeriesForestClassifier
# from sktime.transformations.panel.compose import ColumnConcatenator
from data_generation import *
from bounce_prediction import return_bounce_prediction
from game_tracking import track_game
from court_tracking import track_court
from ball_tracking import track_ball


# parse parameters
parser = argparse.ArgumentParser()

parser.add_argument("--input_video_path", type=str)

args = parser.parse_args()

input_video_path = args.input_video_path

court_results = track_court(input_video_path)

print('')
print('COURT RESULTS')
print('')
print(court_results)
print('')
frames = court_results[0]
xTopLeft = court_results[1][0]
yTopLeft = court_results[1][1]
xBottomLeft = court_results[1][2]
yBottomLeft = court_results[1][3]
xTopRight = court_results[1][4]
yTopRight= court_results[1][5]
xBottomRight = court_results[1][6]
yBottomRight = court_results[1][7]
xMiddleLeft = court_results[1][8]
yMiddleLeft = court_results[1][9]
xMiddleRight = court_results[1][10]
yMiddleRight = court_results[1][11]

print('')
print('y midelle right')
print('')
print(yMiddleRight)
print('')

ball = track_ball(frames, input_video_path)
print('')
print(ball)
print('')
coords = ball[0]
q = ball[1]
t = ball[2]

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

idx = return_bounce_prediction(xy)

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

track_game(idx, extremaBottom, extremaTop, xy)
