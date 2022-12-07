import cv2
import os
from utils import judge_ball
from detection import merge

video_final = cv2.VideoCapture("VideoOutput/video_tracknet_output.mp4")

output_width = int(video_final.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(video_final.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_final.get(cv2.CAP_PROP_FPS))
length = int(video_final.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')

output_final_video = cv2.VideoWriter("VideoOutput/video_final_output.mp4", fourcc, fps, (output_width, output_height))

def track_game(idx, extremaBottom, extremaTop, xy):
    bounce_count = 0
    call = ''
    yMiddleLeft = extremaBottom[3][1]
    fed = cv2.imread('./images/fed-nobg.png')
    nad = cv2.imread('./images/nad-nobg.png')
    while True:
        ret, frame = video_final.read()

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
            cv2.rectangle(frame,(0,0),(580,150),(255,255,255),-1)
            cv2.putText(frame,f'Bounces:{bounce_count}',(10,100),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,0),10,cv2.LINE_AA)
            if i in idx:
                color = (255, 0, 0)
                thickness = -1
                if i-1 not in idx:
                    bounce_count = bounce_count + 1

                ex = extremaBottom if int(xy[i][1]) > yMiddleLeft else extremaTop
                center_coordinates = int(xy[i][0]), int(xy[i][1])
                call = judge_ball(ex, center_coordinates)

                cv2.circle(frame, center_coordinates, 10, color, thickness)
                img = nadal if call == 'IN' else federer
                cv2.putText(frame,f'Bounces:{bounce_count}',(10,100),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,0),10,cv2.LINE_AA)
                output = merge(frame, img)
                output_final_video.write(output)
            elif i-1 in idx or i-2 in idx or i-3 in idx or i-4 in idx or i-5 in idx or i-6 in idx or i-7 in idx or i-8 in idx or i-9 in idx or i-10 in idx:
                img = nadal if call == 'IN' else federer
                cv2.putText(frame,f'Bounces:{bounce_count}',(10,100),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,0),10,cv2.LINE_AA)
                output = merge(frame, img)
                output_final_video.write(output)
            else:
                cv2.putText(frame,f'Bounces:{bounce_count}',(10,100),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,0),10,cv2.LINE_AA)
                output_final_video.write(frame)
                i += 1
        else:
            break

    video_final.release()
    output_final_video.release()

    os.remove("/VideoOutput/video_lines_output.mp4")
    os.remove("/VideoOutput/video_tracknet_output.mp4")
