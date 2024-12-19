import cv2
import numpy as np
import numpy.linalg as la

class KalmanFilter:
    def __init__(self):
        self.kf = None

    def initialize(self, initial_state, initial_error_covariance):
        state_dim = len(initial_state)
        measurement_dim = 2  # Assuming 2D measurements (x, y)

        self.kf = {
            'A': np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32),
            'H': np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32),
            'Q': np.eye(state_dim) * 0.01,  # Process noise covariance
            'R': np.eye(measurement_dim) * 1,  # Measurement noise covariance
            'B': np.zeros((state_dim, 2), np.float32),
            'u': np.zeros((2, 1), np.float32),
            'mu': initial_state,
            'P': initial_error_covariance,
        }

    def predict(self):
        A = self.kf['A']
        mu = self.kf['mu']
        B = self.kf['B']
        u = self.kf['u']
        Q = self.kf['Q']

        x_pred = A @ mu + B @ u
        P_pred = A @ self.kf['P'] @ A.T + Q / 4
        return x_pred, P_pred, None, None  # Adding None values to match the expected unpacking


    def correct(self, z):
        H = self.kf['H']
        R = self.kf['R']
        P_pred = self.kf['P']

        x_pred, P_pred, _, _ = self.predict()  # Call the predict method here
        zp = H @ x_pred

        epsilon = z - zp
        k = P_pred @ H.T @ la.inv(H @ P_pred @ H.T + R)

        x_esti = x_pred + k @ epsilon
        P = (np.eye(len(P_pred)) - k @ H) @ P_pred

        self.kf['mu'] = x_esti
        self.kf['P'] = P
        return x_esti, P, zp

cursor_x = 250
cursor_y = 250
start = False

img_height = 2000
img_width = 2000

img = np.zeros((img_height, img_width, 3), np.uint8)

kf = KalmanFilter()
initial_state = np.array([[cursor_x], [cursor_y], [0], [0]], dtype=np.float32)
initial_error_covariance = np.eye(4, dtype=np.float32)
kf.initialize(initial_state, initial_error_covariance)

def update_cursor_pos(event, x, y, flags, param):
    global cursor_x, cursor_y, start, img
    if event == cv2.EVENT_LBUTTONDOWN:
        start = True
        cursor_x = x
        cursor_y = y
        kf.initialize(np.array([[x], [y], [0], [0]], dtype=np.float32), initial_error_covariance)
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        img = np.zeros((img_height, img_width, 3), np.uint8)
    elif event == cv2.EVENT_MOUSEMOVE:
        cursor_x = x
        cursor_y = y

if __name__ == "__main__":
    cv2.namedWindow("KalmanDemo")
    cv2.setMouseCallback('KalmanDemo', update_cursor_pos)

    while True:
        if start:
            measured_position = np.array([[cursor_x], [cursor_y]], dtype=np.float32)
            estimated_state, _, _ = kf.correct(measured_position)
            predicted_position, _, _, _ = kf.predict()

            x, y = int(measured_position[0]), int(measured_position[1])
            cv2.circle(img, (x, y), 2, (255, 255, 0), thickness=-1)

            x_est, y_est = int(estimated_state[0]), int(estimated_state[1])
            cv2.circle(img, (x_est, y_est), 2, (0, 255, 255), thickness=-1)

        cv2.imshow("KalmanDemo", img)

        code = cv2.waitKey(10)
        if code == 27:  # Press 'Esc' key to exit
            break

    cv2.destroyWindow("KalmanDemo")
