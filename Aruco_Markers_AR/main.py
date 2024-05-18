# Import necessary libraries
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QSlider
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
import cv2
from collections import deque
import imutils
import time
import numpy as np
import os

# Global variable to cache reference points for marker detection
CACHED_REF_PTS = None

# Function to calculate frames per second (FPS) and reset interval
def calculate_fps(start_time, num_frames, reset_interval=2.0):
    elapsed_time = time.time() - start_time
    fps = num_frames / elapsed_time

    if elapsed_time > reset_interval:
        start_time = time.time()
        num_frames = 0

    return fps, start_time, num_frames

# Function to find ArUco markers in a frame, perform perspective transformation, and create an AR effect
def find_and_warp(frame, source, cornerIDs, arucoDetector, marker_corners, marker_ids, rejected_candidates, useCache=False):
    global CACHED_REF_PTS
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (imgH, imgW) = frame.shape[:2]
    (srcH, srcW) = source.shape[:2]

    # Detect ArUco markers in the frame
    (corners, ids, rejected) = arucoDetector.detectMarkers(gray, marker_corners, np.array(marker_ids), rejected_candidates)
    ids = np.array([]) if len(corners) != 4 else ids.flatten()
    refPts = []
    
    # Extract reference points based on specified corner IDs
    for i in cornerIDs:
        j = np.squeeze(np.where(ids == i))
        if j.size == 0:
            continue
        corner = np.squeeze(corners[j])
        refPts.append(corner)
    
    # If the correct number of reference points is not found, use cached points (if available)
    if len(refPts) != 4:
        if useCache and CACHED_REF_PTS is not None:
            refPts = CACHED_REF_PTS
        else:
            return None
    
    # Update the cache if applicable
    if useCache:
        CACHED_REF_PTS = refPts
    
    # Order reference points and perform perspective transformation
    (refPtTL, refPtTR, refPtBR, refPtBL) = refPts
    dstMat = [refPtTL[0], refPtTR[1], refPtBR[2], refPtBL[3]]
    dstMat = np.array(dstMat)
    srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])
    (H, _) = cv2.findHomography(srcMat, dstMat)
    warped = cv2.warpPerspective(source, H, (imgW, imgH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # Create a mask for blending the AR effect into the original frame
    mask = np.zeros((imgH, imgW), dtype="uint8")
    cv2.fillConvexPoly(mask, dstMat.astype("int32"), 255, cv2.LINE_AA)
    mask = cv2.dilate(mask, None, iterations=2)
    mask = mask / 255.0
    mask = np.stack([mask] * 3, axis=-1)
    
    # Blend the AR effect into the original frame
    result = frame * (1 - mask) + warped * mask
    result = result.astype("uint8")

    return result

# Class for the AR application GUI
class ARApp(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize parameters and components
        self.use_cache = 1
        self.marker_ids, self.marker_corners, self.rejected_candidates = [], [], []
        self.arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        self.arucoParams = cv2.aruco.DetectorParameters()
        self.arucoDetector = cv2.aruco.ArucoDetector(self.arucoDict, self.arucoParams)

        script_dir = os.path.dirname(os.path.realpath("__file__"))
        self.vf = cv2.VideoCapture(os.path.join(script_dir, "AAA.mp4"))
        
        self.num_frames = 0
        self.start_time = time.time()

        self.Q = deque(maxlen=256)
        
        self.current_webcam_index = 0
        
        self.initUI()

    # Initialize the GUI components
    def initUI(self):
        self.setWindowTitle('Aruco Marker AR Video Embed')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.tv_placeholder = QLabel('Aruco Marker Augmented Reality')
        layout.addWidget(self.tv_placeholder)

        btn_start_cam = QPushButton('Start Cam!')
        btn_start_cam.clicked.connect(self.on_button_press)
        layout.addWidget(btn_start_cam)
        
        btn_switch_cam = QPushButton('Switch Camera')
        btn_switch_cam.clicked.connect(self.on_switch_camera)
        layout.addWidget(btn_switch_cam)
        
        # Slider for adjusting camera exposure
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setMinimum(-15)  
        self.exposure_slider.setMaximum(-1)
        self.exposure_slider.setValue(-5) 
        self.exposure_slider.valueChanged.connect(self.on_exposure_change)
        layout.addWidget(self.exposure_slider)

        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        self.video_label = QLabel()
        layout.addWidget(self.video_label)

        # Timer for updating the GUI at regular intervals
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(11) 
        
        self.fps_label = QLabel('FPS: N/A')
        layout.addWidget(self.fps_label)

        self.setLayout(layout)
        self.show()

    # Handle exposure change based on slider value
    def on_exposure_change(self, value):
        if hasattr(self, 'vs') and self.vs is not None:
            exposure_value = value
            self.vs.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
            time.sleep(0.5) 

    # Switch between different cameras (webcams)
    def on_switch_camera(self):
        self.vs.release()
        self.current_webcam_index = (self.current_webcam_index + 1) % 2
        self.vs = cv2.VideoCapture(self.current_webcam_index)
        self.vs.set(cv2.CAP_PROP_EXPOSURE, -4.5)
        time.sleep(2.0)

    # Start the webcam and AR application
    def on_button_press(self):
        self.use_cache = 1
        self.marker_ids, self.marker_corners, self.rejected_candidates = [], [], []
        self.arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        self.arucoParams = cv2.aruco.DetectorParameters()
        self.arucoDetector = cv2.aruco.ArucoDetector(self.arucoDict, self.arucoParams)

        # Stop existing video streams
        self.stop_video_streams()
        
        # Start the default webcam (index 0)
        self.vs = cv2.VideoCapture(0)
        self.vs.set(cv2.CAP_PROP_EXPOSURE, -4.5)
        time.sleep(2.0)

        self.num_frames = 0
        self.start_time = time.time()

        self.Q = deque(maxlen=256)

        self.source = None

        # Load the AR video source (AAA.mp4)
        self.vf = cv2.VideoCapture("AAA.mp4")

        (grabbed, self.source) = self.vf.read()
        self.Q.appendleft(self.source)

    # Stop all video streams when closing the application
    def stop_video_streams(self):
        if hasattr(self, 'vs') and self.vs is not None:
            self.vs.release()
            del self.vs

        if hasattr(self, 'ar_video_source') and self.ar_video_source is not None:
            self.ar_video_source.release()
            del self.ar_video_source

        if hasattr(self, 'vf') and self.vf is not None:
            self.vf.release()
            del self.vf

    # Handle the close event of the application
    def closeEvent(self, event):
        self.stop_video_streams()
        event.accept()

    # Update the GUI and perform AR processing
    def update(self):
        if hasattr(self, 'vs') and self.vs is not None:
            if len(self.Q) > 0:
                _, frame = self.vs.read()
    
                if frame is not None and frame.size > 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
                    frame_rgb = imutils.resize(frame_rgb, width=600)
    
                    live_warped = find_and_warp(
                        frame_rgb, self.source,
                        cornerIDs=(24, 42, 66, 70),
                        arucoDetector=self.arucoDetector,
                        marker_corners=self.marker_corners,
                        marker_ids=self.marker_ids,
                        rejected_candidates=self.rejected_candidates,
                        useCache=self.use_cache
                    )
    
                    if live_warped is not None:
                        frame_rgb = live_warped
                        self.source = self.Q.popleft()
    
                    if len(self.Q) != self.Q.maxlen:
                        (grabbed, nextFrame) = self.vf.read()
                        if grabbed and nextFrame is not None and nextFrame.size > 0:
                            nextFrame = cv2.cvtColor(nextFrame, cv2.COLOR_BGR2RGB)
                            self.Q.append(nextFrame)
                        else:
                            self.vf.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
                    height, width, channel = frame_rgb.shape
                    bytes_per_line = 3 * width
                    q_img = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_img)
                    self.image_label.setPixmap(pixmap)
    
                    self.num_frames += 1
                    self.fps, self.start_time, self.num_frames = calculate_fps(self.start_time, self.num_frames)
                    if self.num_frames % 30 == 0:
                        self.fps_label.setText(f'FPS: {self.fps:.2f}')

# Entry point of the application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ar_app = ARApp()
    sys.exit(app.exec_())