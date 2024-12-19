import matplotlib.pyplot as plt
import numpy as np

class KalmanFilter:
    def __init__(self):
        # Initialize Kalman filter parameters
        self.dt = 1.0  # Time step
        self.A = np.array([[1, self.dt], [0, 1]])  # State transition matrix
        self.H = np.array([[1, 0]])  # Observation matrix
        self.Q = np.eye(2)  # Process noise covariance
        self.R = np.array([[25]])  # Measurement noise covariance
        self.state = np.zeros((2, 1))  # Initial state
        self.P = np.eye(2)  # Initial covariance matrix

    def predict(self, measurement):
        # Prediction step
        self.state = np.dot(self.A, self.state)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        # Update step
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(np.dot(np.dot(self.H, self.P), self.H.T) + self.R))
        self.state = self.state + np.dot(K, (measurement - np.dot(self.H, self.state)))
        self.P = np.dot((np.eye(2) - np.dot(K, self.H)), self.P)

        # Return the predicted state
        return self.state.astype(int)

# Create KalmanFilter instance
kf = KalmanFilter()

point_positions = [(4, 300), (61, 256), (116, 214), (170, 180), (225, 148), (279, 120), (332, 97),
                   (383, 80), (434, 66), (484, 55), (535, 49), (586, 49), (634, 50),
                   (683, 58), (731, 69), (778, 82), (824, 101), (870, 124), (917, 148),
                   (962, 169), (1006, 212), (1051, 249), (1093, 290)]

# Lists to store actual and predicted positions
actual_positions = np.array(point_positions)
predicted_positions = []

# Predict the trajectory using Kalman filter
for pt in actual_positions:
    predicted = kf.predict(pt)
    predicted_positions.append(predicted.flatten())

# Convert lists to NumPy arrays
actual_positions = np.array(actual_positions)
predicted_positions = np.array(predicted_positions)

# Plotting
plt.plot(actual_positions[:, 0], actual_positions[:, 1], 'ro-', label='Actual Position')
plt.plot(predicted_positions[:, 0], predicted_positions[:, 1], 'bo-', label='Predicted Position')

plt.title('Trajectory Prediction with Kalman Filter')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True)
plt.show()
