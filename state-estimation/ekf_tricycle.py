# Problem 2
# EKF Pose Estimation Simulation for Tricycle Robot
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

class StateEstimator:

    def __init__(self, process_noise=[0.001, 0.001, 0.001, 0.001]):
        # State
        self.state = np.array([[0., 0.1, np.deg2rad(30), np.deg2rad(2)]]).T
        self.gt = self.state.copy()
        self.state_prior = np.zeros_like(self.state)
        self.u = np.zeros((2, 1))
        self.cov_prior = 0.0*np.eye(4)
        self.cov = 0.0*np.eye(4) # covariance of true belief
        self.Q = np.diag([0.1, 0.3]) # covariance of measurement noise (Q)
        self.Q_decomp = np.linalg.cholesky(self.Q) # Q = AA.T --> Cholesky decomposition of covariance Q to generate zero-mean WGN
        self.R = np.diag(process_noise) # covariance of process noise (R)

        # Params
        self.delta_t = 0.1
        self.duration = 10.0
        self.l = 1.0

        # Plotting
        self.history = None
        self.ground_truth = None
        self.measurements = None

    
    # Read input
    def cmd_vel(self, t):
        self.u[0] = 10.0*np.sin(t)
        self.u[1] = 0.01


    def motion_model(self, state):
        new_state = np.zeros((4, 1))
        new_state[0, 0] = state[0, 0] + self.delta_t*self.u[0]*np.cos(state[2, 0])
        new_state[1, 0] = state[1, 0] + self.delta_t*self.u[0]*np.sin(state[2, 0])
        new_state[2, 0] = state[2, 0] + self.delta_t*self.u[0]*np.tan(state[3, 0]) / self.l
        new_state[3, 0] = state[3, 0] + self.delta_t*self.u[1]
        return new_state

    def linearize_motion(self):
        # Jacobian G was determined analytically
        G = np.eye(4)
        G[0, 2] = -self.delta_t*self.u[0]*np.sin(self.state[2])
        G[1, 2] = self.delta_t*self.u[0]*np.cos(self.state[2])
        G[2, 3] = self.delta_t*self.u[0]/(self.l * np.cos(self.state[3])**2)
        return G

    def measurement_model(self, state):
        meas = np.zeros((2, 1))
        meas[0, 0] = np.sqrt(state[0]*state[0] + state[1]*state[1])
        meas[1, 0] = np.arctan2(state[1], state[0])
        return meas
        
    def linearize_meas(self):
        # Jacobian H was determined analytically
        H = np.zeros((2, 4))
        R = np.sqrt(self.state_prior[0]*self.state_prior[0] + self.state_prior[1]*self.state_prior[1])
        H[0, 0] = self.state_prior[0] / R
        H[0, 1] = self.state_prior[1] / R
        H[1, 0] = -self.state_prior[1]/(self.state_prior[0]**2 + self.state_prior[1]**2)
        H[1, 1] = self.state_prior[0]/(self.state_prior[0]**2 + self.state_prior[1]**2)
        return H

    # Run pose estimation
    def run(self):
        for t in np.arange(0, self.duration, self.delta_t):
            # Simulate input update
            self.cmd_vel(t)

            # Ground truth update
            self.gt = self.motion_model(self.gt)
            if self.ground_truth is not None:
                self.ground_truth = np.append(self.ground_truth, self.gt,  axis=1)
            else:
                self.ground_truth = self.gt

            # Estimation
            # (1) predict means
            # Add zero-mean WGN
            epsilon_t = np.array([np.random.normal(0, np.sqrt(np.diag(self.R)))]).T
            self.state_prior = self.motion_model(self.state) + epsilon_t
            # (2) predict covariance
            G = self.linearize_motion() # dim: 4x4
            self.cov_prior = np.dot(G, self.cov.dot(G.T)) + self.R
            # (3) compute kalman gain
            H = self.linearize_meas() # 2x4
            K = np.dot(np.dot(self.cov_prior, H.T), np.linalg.inv(np.dot(H, np.dot(self.cov_prior, H.T)) + self.Q)) # output dim: 4x4
            # (4) update mean
            # Add zero-mean WGN
            delta_t = np.array([np.random.normal(0, np.sqrt(np.diag(self.Q)))]).T
            z_t = self.measurement_model(self.gt) + delta_t
            innovation = z_t - self.measurement_model(self.state_prior) # output dim: 4x1
            self.state = self.state_prior + np.dot(K, innovation) # output dim: 4x1
            # (5) update covariance
            self.cov = np.dot((np.eye(4) - np.dot(K, H)), self.cov_prior) # output dim: 4x4

            # Record state and measurement for plotting
            print("state: {}".format(self.state))
            if self.history is not None:
                self.history = np.append(self.history, self.state, axis=1)
            else:
                self.history = self.state

            if self.measurements is not None:
                self.measurements = np.append(self.measurements, z_t, axis=1)
            else:
                self.measurements = z_t


if __name__ == '__main__':

    try:
        cov_R = [0.00, 0.00, 0.00, 0.00]
        # cov_R = [0.001, 0.001, 0.001, 0.001]
        # cov_R = [0.005, 0.005, 0.01, 0.01]
        # cov_R = [0.1, 0.1, 0.05, 0.05]
        estimator = StateEstimator(process_noise=cov_R)
        estimator.run()
        # Plotting
        t = np.arange(0, 10, 0.1)

        fig = plt.figure(figsize=(24,12))
        fig.add_subplot(3,2,1)
        plt.scatter(t, estimator.history[0], marker="+", label="state")
        plt.scatter(t, estimator.measurements[0], marker="x", label="m_range")
        plt.scatter(t, estimator.measurements[1], marker="*", label="m_bearing")
        plt.plot(t, estimator.ground_truth[0], c='r', label="ground truth")
        plt.legend()
        plt.xlabel("Time, t [s]",fontsize=16)
        plt.ylabel("X [m]", fontsize=16)
        plt.title("X", fontsize=16)

        fig.add_subplot(3,2,2)
        plt.scatter(t, estimator.history[1], marker="+", label="state")
        plt.scatter(t, estimator.measurements[0], marker="x", label="m_range")
        plt.scatter(t, estimator.measurements[1], marker="*", label="m_bearing")
        plt.plot(t, estimator.ground_truth[1], c='r', label="ground truth")
        plt.legend()
        plt.xlabel("Time, t [s]",fontsize=16)
        plt.ylabel("Y [m]", fontsize=16)
        plt.title("Y", fontsize=16)

        fig.add_subplot(3,2,3)
        plt.scatter(t, estimator.history[2], marker="+", label="state")
        plt.plot(t, estimator.ground_truth[2], c='r', label="ground truth")
        plt.legend()
        plt.xlabel("Time, t [s]",fontsize=16)
        plt.ylabel("Theta [rad]", fontsize=16)
        plt.title("Theta", fontsize=16)

        fig.add_subplot(3,2,4)
        plt.scatter(t, estimator.history[3], marker="+", label="state")
        plt.plot(t, estimator.ground_truth[3], c='r', label="ground truth")
        plt.legend()
        plt.xlabel("Time, t [s]",fontsize=16)
        plt.ylabel("Delta [rad]", fontsize=16)
        plt.title("Delta", fontsize=16)

        fig.add_subplot(3,2,5)
        plt.plot(estimator.history[0], estimator.history[1], c="c", label="state")
        plt.plot(estimator.ground_truth[0], estimator.ground_truth[1], c='r', label="ground truth")
        plt.legend()
        plt.xlabel("X, [m]",fontsize=16)
        plt.ylabel("Y, [m]", fontsize=16)
        plt.title("XY", fontsize=16)
        # title = "Pose estimation with R=diag(" + str(cov_R) +")"
        # fig.suptitle(title, fontsize=20)
        fig.tight_layout()
        plt.show()
    except KeyboardInterrupt:
        exit(0)

