import cv2
import numpy as np
from filterpy.kalman import KalmanFilter

# Kalman filter initialization
kf = KalmanFilter(dim_x=4, dim_z=2)
kf.F = np.array([[1, 0, 1, 0],
                 [0, 1, 0, 1],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])

kf.H = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0]])

kf.P *= 1e2
kf.R = 5
kf.Q *= 1e-3

def predict(kf, center):
    kf.predict()
    kf.update(center.reshape(-1, 1))  # Reshape the center to (2, 1) before updating
    return np.array([kf.x[0], kf.x[1]])

def detect_and_draw_white_cars(frame, car_cascade, lk_params, prev_pts, kf, prev_gray=None, good_pts=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 25, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    white_car_only = cv2.bitwise_and(frame, frame, mask=mask_white)
    gray = cv2.cvtColor(white_car_only, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if prev_gray is None:
        prev_gray = gray

    if good_pts is None:
        good_pts = prev_pts

    for contour in contours:
        if cv2.contourArea(contour) > 100:
            (x, y, w, h) = cv2.boundingRect(contour)
            roi = frame[y:y + h, x:x + w]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            cars = car_cascade.detectMultiScale(gray_roi, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(cars) > 0:
                center = np.array([int(x + w / 2), int(y + h / 2)])  # Convert center to NumPy array

                # Without Optical Flow: Predict using Kalman filter
                predicted_position_no_optical_flow = predict(kf, center)

                # Draw circles at the predicted positions without optical flow (X-axis and Y-axis)
                cv2.circle(frame, (int(predicted_position_no_optical_flow[0]), int(center[1])), 5, (255, 0, 0), -1)
                cv2.circle(frame, (int(center[0]), int(predicted_position_no_optical_flow[1])), 5, (255, 0, 0), -1)

                # Optical flow-based motion estimation
                pts, status, err = cv2.calcOpticalFlowPyrLK(
                    prevImg=prev_gray,
                    nextImg=gray,
                    prevPts=prev_pts,
                    nextPts=None,
                    **lk_params
                )

                # Filter out points with low status
                good_pts = pts[status.flatten() == 1]

                # Ensure prev_pts and good_pts have consistent shapes
                prev_pts = good_pts.reshape(-1, 1, 2)

                # Calculate the mean motion vector
                mean_motion = np.mean(good_pts - prev_pts, axis=0).reshape(-1, 1)

                # Ensure mean_motion has shape (2, 1)
                mean_motion = mean_motion[:2]

                # Update Kalman filter with the mean motion
                kf.update(mean_motion)

                # Predict using Kalman filter (With Optical Flow)
                predicted_position_with_optical_flow = predict(kf, center)

                # Draw circles at the predicted positions with optical flow (X-axis and Y-axis)
                cv2.circle(frame, (int(predicted_position_with_optical_flow[0]), int(center[1])), 5, (0, 255, 0), -1)
                cv2.circle(frame, (int(center[0]), int(predicted_position_with_optical_flow[1])), 5, (0, 255, 0), -1)

                # Resultant position (combination of both predictions)
                resultant_position = (predicted_position_no_optical_flow + predicted_position_with_optical_flow) / 2

                # Draw a circle at the resultant position in red
                cv2.circle(frame, (int(resultant_position[0]), int(resultant_position[1])), 5, (0, 0, 255), -1)

                # Draw a circle around the detected car
                cv2.circle(frame, (int(center[0]), int(center[1])), int((w + h) / 4), (0, 255, 0), 2)

    return frame, gray, prev_pts

car_cascade = cv2.CascadeClassifier('C:\\Users\\hp\\Desktop\\sem5\\mis project\\FINAL\\haarcascade_car.xml')
video_capture = cv2.VideoCapture('C:\\Users\\hp\\Desktop\\sem5\\mis project\\FINAL\\pv_2.mp4')

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

ret, first_frame = video_capture.read()
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
prev_pts = cv2.goodFeaturesToTrack(first_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
prev_gray = first_gray  # Initialize prev_gray

while True:
    ret, frame = video_capture.read()

    if not ret:
        break

    result_frame, prev_gray, prev_pts = detect_and_draw_white_cars(frame, car_cascade, lk_params, prev_pts, kf, prev_gray)

    cv2.imshow('White Car Detection with Kalman Filter and Optical Flow', result_frame)
    cv2.waitKey(30)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
