# EKF Pose Estimation Simulation for Tricycle Robot
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

class StateEstimator:

    def __init__(self, sigma=0.1, mu=0.0):
        # State
        # TODO: change angles in degrees -> angles in radians
        self.state = np.array([[0., 0.1, np.deg2rad(30), np.deg2rad(2)]]).T
        print(self.state.shape)
        self.state_prior = np.zeros_like(self.state)
        self.u = np.zeros((2, 1))
        self.meas =  np.zeros_like(self.state)
        self.cov_prior = np.eye(4)
        self.cov = np.eye(4) # covariance of true belief
        self.Q = np.diag([0.1, 0.3, 1e-6, 1e-6]) # covariance of measurement noise (Q)
        # self.Q_decomp = np.linalg.cholesky(self.Q) # Q = AA.T --> Cholesky decomposition of covariance Q to generate zero-mean WGN
        self.R = sigma * np.random.randn(4, 4) + mu # covariance of process noise (R)
        # self.R = np.zeros((4, 4))

        # Params
        self.delta_t = 0.1
        self.duration = 10.0
        self.l = 1.0

        # Plotting
        self.history = None
        # self.prediction_history = None
        # self.measurement_history = None
        # self.estimated_history = None

    
    # Read input
    def cmd_vel(self, t):
        self.u[0] = 10*np.sin(t)
        self.u[1] = 0.01

    # Read measurement / evaluate measurement model
    def measurement_model(self, state):
        meas = np.zeros((4, 1))
        R = np.sqrt(state[0]*state[0] + state[1]*state[1])
        meas[0, 0] = R*np.cos(state[2])
        meas[1, 0] = R*np.sin(state[2])
        meas[2, 0] = state[2]
        meas[3, 0] = state[3]
        # Add zero-mean WGN
        # U = np.array([np.random.normal(0, np.sqrt(np.diag(self.Q_decomp)))]).T
        # return meas + U
        return meas

    def linearize_motion(self):
        # Jacobian of G was determined analytically
        G = np.eye(4)
        G[0, 2] = -self.delta_t*self.u[0]*np.sin(self.state_prior[2])
        G[1, 2] = self.delta_t*self.u[0]*np.cos(self.state_prior[2])
        G[2, 3] = self.delta_t*self.u[0]/(self.l * np.cos(self.state_prior[2])*np.cos(self.state_prior[2]))
        return G
  
    def linearize_meas(self):
        # Jacobian of H was determined analytically
        H = np.zeros((4, 4))
        R = np.sqrt(self.state[0]*self.state[0] + self.state[1]*self.state[1])
        H[0, 2] = -R*np.sin(self.state[2])
        H[1, 2] = R*np.cos(self.state[2])
        H[2, 2] = 1
        H[3, 3] = 1
        return H

    # Run pose estimation
    def run(self):
        for t in np.arange(0, self.duration, self.delta_t):
            # Simulate input update
            self.cmd_vel(t)

            # (1) predict means
            self.state_prior[0, 0] = self.state[0] + self.delta_t*self.u[0]*np.cos(self.state[2])
            self.state_prior[1, 0] = self.state[1] + self.delta_t*self.u[0]*np.sin(self.state[2])
            self.state_prior[2, 0] = self.state[2] + self.delta_t*self.u[0]*np.tan(self.state[3]) / self.l
            self.state_prior[3, 0] = self.state[3] + self.delta_t*self.u[1]
            # (2) predict covariance
            G = self.linearize_motion() # dim: 4x4
            self.cov_prior = np.dot(G, self.cov.dot(G.T)) + self.R
            # (3) compute kalman gain
            H = self.linearize_meas() # 2x4
            K = np.dot(np.dot(self.cov_prior, H.T), np.linalg.inv(np.dot(H, np.dot(self.cov_prior, H.T)) + self.Q)) # output dim: 4x4
            # (4) update mean
            innovation = self.measurement_model(self.state) - self.measurement_model(self.state_prior) # output dim: 4x1
            self.state = self.state_prior + np.dot(K, innovation) # output dim: 4x1
            # (5) update covariance
            self.cov = np.dot((np.eye(4) - np.dot(K, H)), self.cov_prior) # output dim: 4x4

            # Display pose estimate
            print("state: {}".format(self.state))
            if self.history is not None:
                self.history = np.append(self.history, self.state, axis=1)
            else:
                self.history = self.state


if __name__ == '__main__':

    try:
        estimator = StateEstimator()
        estimator.run()
        # Plotting
        t = np.arange(0, 10, 0.1)

        fig = plt.figure(figsize=(24,12))
        fig.add_subplot(3,2,1)
        plt.plot(t, estimator.history[0])
        plt.title("X")
        fig.add_subplot(3,2,2)
        plt.plot(t, estimator.history[1])
        plt.title("Y")
        fig.add_subplot(3,2,3)
        plt.plot(t, estimator.history[2])
        plt.title("Theta")
        fig.add_subplot(3,2,4)
        plt.plot(t, estimator.history[3])
        plt.title("Delta")
        fig.add_subplot(3,2,5)
        plt.plot(estimator.history[0], estimator.history[1])
        plt.title("XY")
        plt.show()
    except KeyboardInterrupt:
        exit(0)

