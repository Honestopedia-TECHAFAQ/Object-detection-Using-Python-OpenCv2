import cv2
from tkinter import Tk, filedialog

def choose_video_file():
    Tk().withdraw()
    file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video files", "*.mp4;*.avi")])
    return file_path

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
video_file = choose_video_file()

if not video_file:
    print("No video file selected. Exiting.")
    exit()

video_capture = cv2.VideoCapture(video_file)
if not video_capture.isOpened():
    print("Error: Could not open video.")
    exit()

ret, frame = video_capture.read()
rois = cv2.selectROIs("Select ROIs", frame, fromCenter=False, showCrosshair=True)
rois_hsv = [cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2HSV) for (x, y, w, h) in rois]

term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = video_capture.read()

    if not ret:
        print("End of video.")
        break

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi_hist_list = [cv2.calcHist([roi_hsv], [0], None, [190], [0, 180]) for roi_hsv in rois_hsv]
    for i, roi_hist in enumerate(roi_hist_list):
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        dst = cv2.calcBackProject([frame_hsv], [0], roi_hist, [0, 180], 1)
        ret, track_window = cv2.meanShift(dst, (rois[i][0], rois[i][1], rois[i][2], rois[i][3]), term_criteria)
        x, y, w, h = track_window
        result_frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(result_frame, f"{i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Object Tracking', result_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
