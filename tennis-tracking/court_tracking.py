import cv2
from court_detector import CourtDetector
from utils import get_video_properties

def track_court(input_video_path):
    # get video fps&video size
    video_one = cv2.VideoCapture(input_video_path)
    fps = int(video_one.get(cv2.CAP_PROP_FPS))
    print('fps : {}'.format(fps))
    output_width = int(video_one.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(video_one.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # save prediction images as videos
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_lines_video = cv2.VideoWriter("VideoOutput/video_lines_output.mp4", fourcc, fps, (output_width, output_height))

    # court
    court_detector = CourtDetector()

    # get videos properties
    fps, length, v_width, v_height = get_video_properties(video_one)

    frames = []
    frame_i = 0
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
        ret, frame = video_one.read()
        frame_i += 1

        if ret:
            if frame_i == 1:
                print('Detecting the court ...')
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
                    output_lines_video.write(new_frame)
        else:
            break
        video_one.release()
        output_lines_video.release()
        print('Lines detected')
        return [
            frames,
            [
                xTopLeft,
                yTopLeft,
                xBottomLeft,
                yBottomLeft,
                xTopRight,
                yTopRight,
                xBottomRight,
                yBottomRight,
                xMiddleLeft,
                yMiddleLeft,
                xMiddleRight,
                yMiddleRight
            ]
        ]
