# Problem 3
# Unicycle Robot Pose Estimation from Encoder Profiles
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

class PoseEstimator:

    def __init__(self, duration, step):
        # State
        self.initialize_state()
        self.Dl = None
        self.Dr = None

        # Params
        self.duration = duration
        self.delta_t = step
        self.b = 0.45

    def initialize_state(self):
        self.state = np.array([[0.1, 0.2, np.deg2rad(45)]]).T
        self.history = None
    
    # Generate profiles for Left and Right encoders
    def init_profiles(self, lerr, rerr):
        t = np.arange(0, self.duration, self.delta_t)
        # self.Dl = perror*(np.sin(t) + 1) + np.sin(t) + 1
        self.Dl = lerr*(np.log(t+1) + 1) + np.log(t+1) + 1
        self.Dr = rerr*(np.sin(t) + 1) + np.sin(t) + 1
        # Reset 
        self.initialize_state()


    def motion_model(self, state, D_l, D_r):
        new_state = np.zeros((3, 1))
        delta_center = (D_l + D_r)/2.
        delta_yaw = (D_r - D_l)/self.b
        new_state[0, 0] = state[0, 0] + delta_center*np.cos(state[2, 0] + delta_yaw/2.)
        new_state[1, 0] = state[1, 0] + delta_center*np.sin(state[2, 0] + delta_yaw/2.)
        new_state[2, 0] = state[2, 0] + delta_yaw
        return new_state

    # Run pose construction from encoder profiles
    def run(self, left_error=0.0, right_error=0.0):
        self.init_profiles(left_error, right_error)
        print("Reconstructing pose with error (left error, right error): ({}%, {}%)".format(left_error*100, right_error*100))
        for t in range(0, int(self.duration/self.delta_t)):
            self.state = self.motion_model(self.state, self.Dl[t], self.Dr[t])

            # Record pose update
            # print("state {}: {}".format(t, self.state))
            if self.history is not None:
                self.history = np.append(self.history, self.state, axis=1)
            else:
                self.history = self.state
        return (self.history, self.Dl, self.Dr)

def total_rmse(gt, est, N):
    return np.sqrt(np.mean((est - gt)**2))

def running_rmse(gt, est):
    rmse_x = np.zeros((gt.shape[1], 1))
    rmse_y = np.zeros((gt.shape[1], 1))
    for i in range(1, rmse_x.shape[0]):
        rmse_x[i, 0] = total_rmse(gt[0][:i], est[0][:i], i)
        rmse_y[i, 0] = total_rmse(gt[1][:i], est[1][:i], i)
    return (rmse_x, rmse_y)

if __name__ == '__main__':

    try:
        # Sim params
        duration = 3
        step = 0.1
        t = np.arange(0, duration, step)

        estimator = PoseEstimator(duration, step)
        # Ground truth
        gt_state, gt_dl, gt_dr = estimator.run()
        print("Shape")
        print(gt_state[0].shape)

        # 2% systematic left and right encoder error
        f2_state, f2_dl, f2_dr = estimator.run(left_error=0.02, right_error=0.02)
        f2_rmse_x, f2_rmse_y = running_rmse(gt_state, f2_state)

        # 2% systematic right encoder error
        # 3% systematic left encoder error
        f3_state, f3_dl, f3_dr = estimator.run(left_error=0.03, right_error=0.02)
        f3_rmse_x, f3_rmse_y = running_rmse(gt_state, f3_state)

        # Plotting
        fig = plt.figure(figsize=(24,12))
        fig.add_subplot(4,1,1)
        plt.plot(t, gt_dl, c="r", label="ground truth")
        # plt.plot(t, f2_dl, c="c", label="2% error")
        plt.plot(t, f3_dl, c="c", label="3% error")
        plt.legend()
        plt.xlabel("Time, t [s]", fontsize=16)
        plt.ylabel("D_l [m]", fontsize=16)
        plt.title("Left encoder profile", fontsize=16)
        
        fig.add_subplot(4,1,2)
        plt.plot(t, gt_dr, c="r", label="ground truth")
        # plt.plot(t, f2_dr, c="c", label="2% error")
        plt.plot(t, f3_dr, c="c", label="2% error")
        plt.legend()
        plt.xlabel("Time, t [s]", fontsize=16)
        plt.ylabel("D_r [m]", fontsize=16)
        plt.title("Right encoder profile", fontsize=16)

        fig.add_subplot(4,1,3)
        plt.plot(gt_state[0], gt_state[1], c="r", label="ground truth")
        # plt.plot(f2_state[0], f2_state[1], c="c", label="2% error")
        plt.plot(f3_state[0], f3_state[1], c="c", label="2% and 3% error")
        plt.legend()
        plt.xlabel("X_t [m]",fontsize=16)
        plt.ylabel("Y_t [m]", fontsize=16)
        plt.title("Pose reconstruction with 3% (left) and 2% (right) systematic error", fontsize=16)
        
        fig.add_subplot(4,1,4)
        # plt.plot(t, f2_rmse_x, c='g', label="RMSE X")
        # plt.plot(t, f2_rmse_y, c='b', label="RMSE Y")
        plt.plot(t, f3_rmse_x, c='g', label="RMSE X")
        plt.plot(t, f3_rmse_y, c='b', label="RMSE Y")
        plt.legend()
        plt.xlabel("Time, t [s]",fontsize=16)
        plt.ylabel("RMSE", fontsize=16)
        plt.title("Running RMSE as a function of time", fontsize=16)



        fig.tight_layout()
        plt.show()
    except KeyboardInterrupt:
        exit

