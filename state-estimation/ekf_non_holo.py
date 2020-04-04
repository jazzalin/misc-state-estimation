# Problem 4
# EKF Pose Estimation Simulation for non-holonomic unicycle model
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

class StateEstimator:

    def __init__(self,
                duration,
                step,
                process_noise=[0.0, 0.0, 0.0],
                measurement_noise=[0., 0., 0.],
                nonlinear=False,
                random=False): # nonlinear --> param to select between measurement models
        # State
        self.state = np.array([[0.1, 0.1, np.deg2rad(90)]]).T
        self.gt = self.state.copy()
        self.state_prior = np.zeros_like(self.state)
        self.u = np.zeros((2, 1))
        self.cov_prior = 0.0*np.eye(3)
        self.cov = 0.0*np.eye(3) # covariance of true belief
        if nonlinear:
            self.Q = np.diag([0.5, 1.]) # covariance of measurement noise (Q)
        else:
            self.Q = np.diag([0.5, 0.5, 1.])
        if (np.array(measurement_noise)!=False).all(): # override
            self.Q = np.diag(measurement_noise)
        self.R = np.diag(process_noise) # covariance of process noise (R)

        # Params
        self.nonlinear = nonlinear
        self.random = random
        self.delta_t = step
        self.duration = duration

        # Plotting
        self.history = None
        self.ground_truth = None
        self.measurements = None

    
    # Mock inputs: try different control signals
    # e.g. random, sinusoidal
    def cmd_vel(self, t):
        # Velocity ranges
        lin_vel = 10.0
        ang_vel = 1.0
        if self.random:
            self.u[0] = 2*lin_vel*np.random.random_sample() - lin_vel
            self.u[1] = 2*ang_vel*np.random.random_sample() - ang_vel
        else:
            self.u[0] = lin_vel*np.sin(t)
            self.u[1] = ang_vel*np.sin(t)

    # Unicycle motion model
    def motion_model(self, state):
        new_state = np.zeros((3, 1))
        new_state[0, 0] = state[0, 0] + self.delta_t*self.u[0]*np.cos(state[2, 0])
        new_state[1, 0] = state[1, 0] + self.delta_t*self.u[0]*np.sin(state[2, 0])
        new_state[2, 0] = state[2, 0] + self.delta_t*self.u[1]
        return new_state

    # Linearized unicycle motion model
    def linearize_motion(self):
        # Jacobian G was determined analytically
        G = np.eye(3)
        G[0, 2] = -self.delta_t*self.u[0]*np.sin(self.state[2, 0])
        G[1, 2] = self.delta_t*self.u[0]*np.cos(self.state[2, 0])
        return G

    # Measurement model 1
    def measurement_model1(self, state):
        meas = np.zeros((3, 1))
        meas[0, 0] = state[0, 0]
        meas[1, 0] = state[1, 0]
        meas[2, 0] = state[2, 0]
        return meas
    
    # Linearized measurement model 1
    def linearize_meas1(self):
        # Jacobian H was determined analytically
        H = np.eye(3)
        return H

    # Measurement model 2
    def measurement_model2(self, state):
        meas = np.zeros((2, 1))
        meas[0, 0] = np.sqrt(state[0]*state[0] + state[1]*state[1])
        meas[1, 0] = np.tan(state[1]/(state[0]+1e-6))
        return meas
    
    # Linearized measurement model 2
    def linearize_meas2(self):
        # Jacobian H was determined analytically
        H = np.zeros((2, 3))
        R = np.sqrt(self.state_prior[0]*self.state_prior[0] + self.state_prior[1]*self.state_prior[1])
        H[0, 0] = self.state_prior[0] / R
        H[0, 1] = self.state_prior[1] / R
        H[1, 2] = 1./(np.cos(self.state_prior[2])**2)
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
            G = self.linearize_motion()
            self.cov_prior = np.dot(G, self.cov.dot(G.T)) + self.R
            # (3) compute kalman gain
            if self.nonlinear:
                H = self.linearize_meas2()
            else:
                H = self.linearize_meas1()
            K = np.dot(np.dot(self.cov_prior, H.T), np.linalg.inv(np.dot(H, np.dot(self.cov_prior, H.T)) + self.Q))
            # (4) update mean
            # Add zero-mean WGN
            delta_t = np.array([np.random.normal(0, np.sqrt(np.diag(self.Q)))]).T
            if self.nonlinear:
                z_t = self.measurement_model2(self.gt) + delta_t
                innovation = z_t - self.measurement_model2(self.state_prior)
            else:
                z_t = self.measurement_model1(self.gt) + delta_t
                innovation = z_t - self.measurement_model1(self.state_prior)
            self.state = self.state_prior + np.dot(K, innovation)
            # (5) update covariance
            self.cov = np.dot((np.eye(3) - np.dot(K, H)), self.cov_prior)

            # Display pose estimate
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
        # Sim params
        duration = 20
        step = 0.1
        cov_R = [0.001, 0.001, 0.001]
        cov_Q = [1./2, 1./2]
        # cov_R = [0.005, 0.005, 0.01]
        # cov_R = [0.1, 0.1, 0.05]
        
        estimator = StateEstimator(duration, step, process_noise=cov_R, measurement_noise=cov_Q, nonlinear=True, random=False)
        estimator.run()
        # Plotting
        t = np.arange(0, duration, step)
        fig = plt.figure(figsize=(24,12))

        fig.add_subplot(2,2,1)
        plt.scatter(t, estimator.history[0], marker="+", label="state")
        plt.scatter(t, estimator.measurements[0], marker="x", label="m_range")
        plt.scatter(t, estimator.measurements[1], marker="*", label="m_heading")
        # plt.scatter(t, estimator.measurements[2], marker="*", label="m_theta")
        plt.plot(t, estimator.ground_truth[0], c='r', label="ground truth")
        plt.legend()
        plt.xlabel("Time, t [s]",fontsize=16)
        plt.ylabel("X [m]", fontsize=16)
        plt.title("X", fontsize=16)

        fig.add_subplot(2,2,2)
        plt.scatter(t, estimator.history[1], marker="+", label="state")
        plt.scatter(t, estimator.measurements[0], marker="x", label="m_range")
        plt.scatter(t, estimator.measurements[1], marker="*", label="m_heading")
        # plt.scatter(t, estimator.measurements[2], marker="*", label="m_theta")
        plt.plot(t, estimator.ground_truth[1], c='r', label="ground truth")
        plt.legend()
        plt.xlabel("Time, t [s]",fontsize=16)
        plt.ylabel("Y [m]", fontsize=16)
        plt.title("Y", fontsize=16)

        fig.add_subplot(2,2,3)
        plt.scatter(t, estimator.history[2], marker="+", label="state")
        plt.plot(t, estimator.ground_truth[2], c='r', label="ground truth")
        plt.legend()
        plt.xlabel("Time, t [s]",fontsize=16)
        plt.ylabel("Theta [rad]", fontsize=16)
        plt.title("Theta", fontsize=16)

        fig.add_subplot(2,2,4)
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