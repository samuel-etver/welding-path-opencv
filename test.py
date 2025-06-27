import sys
import numpy as np
import cv2
import array


def filter_image(img):
    img = img.astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 5)
    (a, img) = cv2.threshold(img, 175, 255, cv2.THRESH_BINARY)
    kernel = np.ones((7, 7), np.uint8)
    img = cv2.dilate(img, kernel, iterations=5)    
    return img


def region_of_interest(img):
    mask = np.zeros_like(img)
    mask = cv2.rectangle(mask, (1200,1000), (3300,3500), (255,255,255), -1)
    img = cv2.bitwise_and(img, mask)
    return img


def find_ranges(arr):
    left_index = 0
    right_index = len(arr)

    def is_zero(v):
        return v - 1000 < 0
    
    def skip_zeroes():
        nonlocal left_index, right_index
        while left_index < right_index:
            if not is_zero(arr[left_index]):
                break
            left_index += 1
        return left_index < right_index

    def get_range():
        nonlocal left_index, right_index
        range_left_index = left_index
        while left_index < right_index:
            if is_zero(arr[left_index]):
                break
            left_index += 1
        range_right_index = left_index
        return range_left_index, range_right_index


    lines_ranges = []
    while left_index < right_index:
        if skip_zeroes():
            range_left_index, range_right_index = get_range()
            lines_ranges.append([range_left_index, range_right_index])

    return lines_ranges
            

def get_y(arr, l, r):
    n = 0
    s = 0
    for i in range(l, r):        
        n += arr[i]
        s += arr[i] * i
    return s/n

print("Python version:", sys.version)
print("OpenCV version:", cv2.__version__)

cap = cv2.VideoCapture('a.mp4')
scale = 0.2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = filter_image(frame)
    (width, height) = img.shape
    img = region_of_interest(img)

    dots_histo = array.array("i", [0 for i in range(0, width)])
    for row in range(1000, 3500):
        for col in range(1200, 3300):
            if img[row][col] > 0:
                dots_histo[col] += 1

    lines_ranges = find_ranges(dots_histo)

    y_center = None
    y_center_txt = "None"
    y_left = None
    y_right = None
    y_width = None   

    if len(lines_ranges) > 0:
        line_range = lines_ranges[0]
        y_left = get_y(dots_histo, line_range[0], line_range[1])
        line_range = lines_ranges[len(lines_ranges) - 1]
        y_right = get_y(dots_histo, line_range[0], line_range[1])
        y_width = y_right - y_left
        if y_width > 1100:
            y_center = (y_left + y_right) / 2
            y_center_txt = "{:.3f}".format(y_center)


    print(y_center_txt)
    print('---')

    
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

    font = cv2.FONT_HERSHEY_SIMPLEX
    frame = cv2.putText(frame, y_center_txt, (int(scale * width/2 - 100), int(scale * height - 100)),
                        font, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
    
    if y_center != None:
        x = int(scale * y_center)
        y = int(scale * height / 2)
        line_len = 40
        line_clr = (0, 255, 255)
        line_width = 3
        frame = cv2.line(frame, (x, y - line_len // 2), (x, y + line_len // 2), line_clr, line_width)
        frame = cv2.line(frame, (x - line_len // 2, y), (x + line_len // 2, y), line_clr, line_width)

        line_len = 500
        x = int(scale * y_left)
        frame = cv2.line(frame, (x, y - line_len // 2), (x, y + line_len // 2), line_clr, line_width)
        x = int(scale * y_right)
        frame = cv2.line(frame, (x, y - line_len // 2), (x, y + line_len // 2), line_clr, line_width)
        
    cv2.imshow('src', frame)


    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

print('ok')

