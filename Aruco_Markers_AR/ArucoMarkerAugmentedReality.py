# import the necessary packages
import numpy as np
import cv2
from imutils.video import VideoStream
from collections import deque
import imutils
import time


# initialize our cached reference points
CACHED_REF_PTS = None
# Function to calculate frames per second (FPS)
def calculate_fps(start_time, num_frames):
    elapsed_time = time.time() - start_time
    fps = num_frames / elapsed_time
    return fps

def find_and_warp(frame, source, cornerIDs, arucoDetector, marker_corners, marker_ids, rejected_candidates, useCache=False):
    # grab a reference to our cached reference points
    global CACHED_REF_PTS
    
    # grab the width and height of the frame and source image, respectively
    (imgH, imgW) = frame.shape[:2]
    (srcH, srcW) = source.shape[:2]
    
    # detect AruCo markers in the input frame
    (corners, ids, rejected) = arucoDetector.detectMarkers(frame, marker_corners, np.array(marker_ids), rejected_candidates)
    
    # if we *did not* find our four ArUco markers, initialize an
    # empty IDs list, otherwise flatten the ID list
    ids = np.array([]) if len(corners) != 4 else ids.flatten()
    
    # initialize our list of reference points
    refPts = []
    
    # loop over the IDs of the ArUco markers in top-left,
    # top-right, bottom-right, and bottom-left order
    for i in cornerIDs:
        # grab the index of the corner with the current ID
        j = np.squeeze(np.where(ids == i))
        
        # if we receive an empty list instead of an integer index,
        # then we could not find the marker with the current ID
        if j.size == 0:
            continue
        
        # otherwise, append the corner (x, y)-coordinates to our list
        # of reference points
        corner = np.squeeze(corners[j])
        refPts.append(corner)
    
    # check to see if we failed to find the four ArUco markers
    if len(refPts) != 4:
        # if we are allowed to use cached reference points, fall
        # back on them
        if useCache and CACHED_REF_PTS is not None:
            refPts = CACHED_REF_PTS
        # otherwise, we cannot use the cache and/or there are no
        # previous cached reference points, so return early
        else:
            return None
    
    # if we are allowed to use cached reference points, then update
    # the cache with the current set
    if useCache:
        CACHED_REF_PTS = refPts
    
    # unpack our ArUco reference points and use the reference points
    # to define the *destination* transform matrix, making sure the
    # points are specified in top-left, top-right, bottom-right, and
    # bottom-left order
    (refPtTL, refPtTR, refPtBR, refPtBL) = refPts
    dstMat = [refPtTL[0], refPtTR[1], refPtBR[2], refPtBL[3]]
    dstMat = np.array(dstMat)
    
    # define the transform matrix for the *source* image in top-left,
    # top-right, bottom-right, and bottom-left order
    srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])
    
    # compute the homography matrix and then warp the source image to
    # the destination based on the homography
    (H, _) = cv2.findHomography(srcMat, dstMat)
    warped = cv2.warpPerspective(source, H, (imgW, imgH))
    
    # construct a mask for the source image now that the perspective
    # warp has taken place (we'll need this mask to copy the source
    # image into the destination)
    mask = np.zeros((imgH, imgW), dtype="uint8")
    cv2.fillConvexPoly(mask, dstMat.astype("int32"), (255, 255, 255), cv2.LINE_AA)
    
    # this step is optional, but to give the source image a black
    # border surrounding it when applied to the source image, you
    # can apply a dilation operation
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, rect, iterations=2)
    
    # create a three channel version of the mask by stacking it
    # depth-wise, such that we can copy the warped source image
    # into the input image
    maskScaled = mask.copy() / 255.0
    maskScaled = np.dstack([maskScaled] * 3)
    
    # copy the warped source image into the input image by
    # (1) multiplying the warped image and masked together,
    # (2) then multiplying the original input image with the
    # mask (giving more weight to the input where there
    # *ARE NOT* masked pixels), and (3) adding the resulting
    # multiplications together
    warpedMultiplied = cv2.multiply(warped.astype("float"), maskScaled)
    imageMultiplied = cv2.multiply(frame.astype(float), 1.0 - maskScaled)
    output = cv2.add(warpedMultiplied, imageMultiplied)
    output = output.astype("uint8")
    
    # return the output frame to the calling function
    return output

input_path = "AAA.mp4"
use_cache = 1

# load the ArUCo dictionary and grab the ArUCo parameters
print("[INFO] initializing marker detector...")
marker_ids, marker_corners, rejected_candidates = [], [], []
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
arucoParams = cv2.aruco.DetectorParameters()
arucoDetector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

# initialize the video file stream
print("[INFO] accessing video stream...")
vf = cv2.VideoCapture(input_path)


# initialize a queue to maintain the next frame from the video stream
Q = deque(maxlen=256)

# we need to have a frame in our queue to start our augmented reality
# pipeline, so read the next frame from our video file source and add
# it to our queue
(grabbed, source) = vf.read()
Q.appendleft(source)



# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
vs.stream.set(cv2.CAP_PROP_EXPOSURE,-6)
vs.stream.set(cv2.CAP_PROP_FPS,60)
time.sleep(2.0)
# loop over the frames from the video stream
num_frames = 0
start_time = time.time()

while len(Q) > 0:
    # grab the frame from our video stream and resize it
    frame = vs.read()
    frame = imutils.resize(frame, width=600)

    # attempt to find the ArUCo markers in the frame, and provided
    # they are found, take the current source image and warp it onto
    # input frame using our augmented reality technique
    warped = find_and_warp(
        frame, source,
        cornerIDs=(24, 42, 66, 70),
        arucoDetector=arucoDetector,
        marker_corners=marker_corners,
        marker_ids=marker_ids,
        rejected_candidates=rejected_candidates,
        useCache=use_cache
    )

    # if the warped frame is not None, then we know (1) we found the
    # four ArUCo markers and (2) the perspective warp was successfully
    # applied
    if warped is not None:
        # set the frame to the output augment reality frame and then
        # grab the next video file frame from our queue
        frame = warped
        source = Q.popleft()

    # for speed/efficiency, we can use a queue to keep the next video
    # frame queue ready for us -- the trick is to ensure the queue is
    # always (or nearly full)
    if len(Q) != Q.maxlen:
        # read the next frame from the video file stream
        (grabbed, nextFrame) = vf.read()
        # if the frame was read (meaning we are not at the end of the
        # video file stream), add the frame to our queue
        if grabbed:
            Q.append(nextFrame)
        else:
            vf.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
        

    # show the output frame
    cv2.imshow("Frame", frame)

    # Calculate and display FPS
    num_frames += 1
    if num_frames % 30 == 0:  # Update FPS every 30 frames
        fps = calculate_fps(start_time, num_frames)
        print(f"FPS: {fps:.2f}")

    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
vf.release()

