import cv2 as cv
import numpy as np

cap = cv.VideoCapture('3686.mp4')

tot_frame = 0
font = cv.FONT_HERSHEY_SIMPLEX

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
scale_percent = 15
width = int(frame_width * scale_percent / 100)
length = int(frame_height * scale_percent / 100)
dim = (width, length)

fps = 30

size = (2*frame_width, frame_height)

output = cv.VideoWriter('output.mp4', cv.VideoWriter_fourcc('m','p','4','v'), fps, size, True)

if cap.isOpened() == False:
    print("video stream failed")

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        tot_frame += 1
        cv.putText(frame, 'Frames: ' + str(tot_frame), (100, 100), font, 3, (0, 255, 255), 2, cv.LINE_4)

        framer = frame.copy()
        new_frame = frame.copy()
        aframe = frame.copy()

        new_frame = cv.cvtColor(new_frame, cv.COLOR_BGR2HSV)

        lower1 = np.array([0, 100, 20])
        upper1 = np.array([5, 255, 255])

        lower2 = np.array([160, 100, 20])
        upper2 = np.array([179, 255, 255])

        lower_mask = cv.inRange(new_frame, lower1, upper1)
        upper_mask = cv.inRange(new_frame, lower2, upper2)

        full_mask = lower_mask + upper_mask

        framer = cv.bitwise_and(framer, framer, mask=full_mask)
        framer = cv.bitwise_not(framer)

        mask = np.ones(framer.shape, dtype=np.uint8)
        mask.fill(255)

        x1 = 384
        y1 = 782
        x2 = 1312
        y2 = 13
        roi_corners = np.array([[(x1, y1), (x2, y1), (x2, y2), (x1, y2)]], dtype=np.int32)
        cv.fillPoly(mask, roi_corners, 0)

        framer = cv.bitwise_or(framer, mask)
        framer = cv.bitwise_not(framer)

        gray = cv.cvtColor(framer, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, 500, 550, None, 5)
        blur = cv.blur(edges, (10, 10))

        linesP = cv.HoughLinesP(blur, cv.HOUGH_PROBABILISTIC, theta=np.pi / 180, threshold=10, minLineLength=50, maxLineGap=10)

        if linesP is not None:
            l = linesP[0][0]
            #cv.line(aframe, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 3, cv.LINE_AA)
            length = ((l[1] - l[3]) ** 2 + (l[2] - l[0]) ** 2) ** 0.5
            ang = abs((l[1] - l[3]) / length)
            if ang >= 0.62 and ang <= 1:
                aframe = cv.circle(aframe, (1700, 150), 50, (0, 255, 0), 100)
            elif ang < 0.62 or ang > 1:
                aframe = cv.circle(aframe, (1700, 150), 50, (0, 255, 255), 100)
            else:
                aframe = cv.circle(aframe, (1700, 150), 50, (0, 0, 255), 100)

        output.write(np.hstack([frame, aframe]))

        cv.resize(frame, dim, interpolation=cv.INTER_AREA)
        cv.resize(aframe, dim, interpolation=cv.INTER_AREA)

        cv.imshow('Center', np.hstack([frame, aframe]))

        if cv.waitKey(50) & 0xFF == ord('0'):
            break
    else:
        if cv.waitKey(50) & 0xFF == ord('0'):
            break

cap.release()
output.release()

cv.destroyAllWindows()
