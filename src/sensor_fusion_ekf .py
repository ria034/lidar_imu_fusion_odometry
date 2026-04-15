import numpy as np
from imu_preintegration import GRAVITY, exp_so3, skew


def log_so3(R):
    # logarithm map for SO(3), returns rotation vector
    cos_angle = (np.trace(R) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    if angle < 1e-10:
        return np.zeros(3)

    log = angle / (2.0 * np.sin(angle)) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ])
    return log


class EKF:

    def __init__(self, init_R, init_v, init_p, init_bg, init_ba):

        # state
        self.R  = init_R.copy()   # 3x3 rotation matrix (world frame)
        self.v  = init_v.copy()   # 3,  velocity        (world frame)
        self.p  = init_p.copy()   # 3,  position        (world frame)
        self.bg = init_bg.copy()  # 3,  gyro  bias
        self.ba = init_ba.copy()  # 3,  accel bias

        # 15x15 covariance: [delta_R(3), delta_v(3), delta_p(3), bg(3), ba(3)]
        self.P = np.eye(15) * 1e-4

        # store previous state for measurement model
        self.R_prev = init_R.copy()
        self.p_prev = init_p.copy()

        # trajectory log
        self.trajectory = [init_p.copy()]


    def predict(self, preint):

        dt       = preint["dt"]
        delta_R  = preint["delta_R"]
        delta_v  = preint["delta_v"]
        delta_p  = preint["delta_p"]
        cov_preint = preint["cov"]

        J_R_bg = preint["J_R_bg"]
        J_v_bg = preint["J_v_bg"]
        J_v_ba = preint["J_v_ba"]
        J_p_bg = preint["J_p_bg"]
        J_p_ba = preint["J_p_ba"]

        # save previous state for update step
        self.R_prev = self.R.copy()
        self.p_prev = self.p.copy()

        # --- state propagation ---
        p_new = self.p + self.v * dt + 0.5 * GRAVITY * dt**2 + self.R @ delta_p
        v_new = self.v + GRAVITY * dt + self.R @ delta_v
        R_new = self.R @ delta_R

        # --- build F (15x15 state transition Jacobian) ---
        F = np.eye(15)

        # d(delta_R) / d(delta_R)
        F[0:3, 0:3]   = delta_R.T

        # d(delta_v) / d(delta_R)
        F[3:6, 0:3]   = -self.R @ skew(delta_v)

        # d(delta_v) / d(bg)
        F[3:6, 9:12]  = self.R @ J_v_bg

        # d(delta_v) / d(ba)
        F[3:6, 12:15] = self.R @ J_v_ba

        # d(delta_p) / d(delta_R)
        F[6:9, 0:3]   = -self.R @ skew(delta_p)

        # d(delta_p) / d(delta_v)
        F[6:9, 3:6]   = np.eye(3) * dt

        # d(delta_p) / d(bg)
        F[6:9, 9:12]  = self.R @ J_p_bg

        # d(delta_p) / d(ba)
        F[6:9, 12:15] = self.R @ J_p_ba

        # biases are modelled as random walk, no cross terms

        # --- process noise from preintegration covariance ---
        Q = cov_preint

        # --- covariance propagation ---
        self.P = F @ self.P @ F.T + Q

        # --- apply propagated state ---
        self.R = R_new
        self.v = v_new
        self.p = p_new

        self.trajectory.append(self.p.copy())


    def update(self, T_icp, icp_noise_scale=0.1):

        R_icp = T_icp[:3, :3]
        t_icp = T_icp[:3,  3]

        # --- predicted relative transform from EKF state ---
        R_rel_pred = self.R_prev.T @ self.R
        t_rel_pred = self.R_prev.T @ (self.p - self.p_prev)

        # --- innovation ---
        # translation residual (3,)
        res_t = t_icp - t_rel_pred

        # rotation residual as rotation vector (3,)
        R_res = R_icp @ R_rel_pred.T
        res_R = log_so3(R_res)

        # stack into 6-vector [rot(3), trans(3)]
        innovation = np.concatenate([res_R, res_t])

        # --- measurement Jacobian H (6x15) ---
        H = np.zeros((6, 15))

        # d(rot_residual) / d(delta_R)  — identity in tangent space
        H[0:3, 0:3]  =  self.R_prev.T

        # d(trans_residual) / d(delta_R)
        H[3:6, 0:3]  = -self.R_prev.T @ skew(self.p - self.p_prev)

        # d(trans_residual) / d(delta_p)
        H[3:6, 6:9]  =  self.R_prev.T

        # --- measurement noise (6x6) ---
        R_mat = np.eye(6) * icp_noise_scale
        R_mat[0:3, 0:3] *= 0.1   # rotation noise tighter than translation

        # --- kalman gain ---
        S = H @ self.P @ H.T + R_mat
        K = self.P @ H.T @ np.linalg.inv(S)

        # --- state correction ---
        dx = K @ innovation

        # apply correction to rotation via exponential map
        self.R  = self.R @ exp_so3(dx[0:3])
        self.v  = self.v  + dx[3:6]
        self.p  = self.p  + dx[6:9]
        self.bg = self.bg + dx[9:12]
        self.ba = self.ba + dx[12:15]

        # --- covariance update (Joseph form for numerical stability) ---
        I_KH    = np.eye(15) - K @ H
        self.P  = I_KH @ self.P @ I_KH.T + K @ R_mat @ K.T

        # update trajectory with corrected position
        self.trajectory[-1] = self.p.copy()

    def correct_preintegration(self, preint):

        d_bg = self.bg - preint.get("bg_prev", self.bg)
        d_ba = self.ba - preint.get("ba_prev", self.ba)

        delta_R_corrected = preint["delta_R"] @ exp_so3(preint["J_R_bg"] @ d_bg)
        delta_v_corrected = preint["delta_v"] + preint["J_v_bg"] @ d_bg + preint["J_v_ba"] @ d_ba
        delta_p_corrected = preint["delta_p"] + preint["J_p_bg"] @ d_bg + preint["J_p_ba"] @ d_ba

        corrected = preint.copy()
        corrected["delta_R"] = delta_R_corrected
        corrected["delta_v"] = delta_v_corrected
        corrected["delta_p"] = delta_p_corrected

        return corrected

    def get_biases(self):
        return self.bg.copy(), self.ba.copy()

    def get_pose(self):
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3,  3] = self.p
        return T

    def get_trajectory(self):
        return np.array(self.trajectory)



if __name__ == "__main__":

    from kitti_loader import KittiLoader
    from imu_preintegration import IMUPreintegrator
    from lidar_scan_matching import Scan_Matching, numpy_to_o3d
    import matplotlib.pyplot as plt

    print("loading kitti...")
    loader  = KittiLoader()
    matcher = Scan_Matching(voxel_size=0.5)

    # initialise EKF from first ground truth pose
    T0    = loader.get_ground_truth_pose_imu(0)
    R0    = T0[:3, :3]
    p0    = T0[:3,  3]
    v0    = np.zeros(3)
    bg0   = np.zeros(3)
    ba0   = np.zeros(3)

    ekf   = EKF(R0, v0, p0, bg0, ba0)
    preint = IMUPreintegrator(bg=bg0, ba=ba0)

    N = 200   # number of frames to process, increase as needed

    for k in range(N - 1):

        # --- IMU preintegration between lidar frames k and k+1 ---
        imu_batch = loader.get_imu_between(k, k + 1)
        preint.reset()
        preint.bg = ekf.bg.copy()
        preint.ba = ekf.ba.copy()
        result = preint.integrate_between(imu_batch)

        # --- EKF predict ---
        ekf.predict(result)

        # --- build ICP initial guess from predicted relative transform ---
        T_init     = np.eye(4)
        T_init[:3, :3] = ekf.R_prev.T @ ekf.R
        T_init[:3,  3] = ekf.R_prev.T @ (ekf.p - ekf.p_prev)

        # --- scan matching ---
        scan_k  = loader.get_lidar_scan(k)
        scan_k1 = loader.get_lidar_scan(k + 1)

        src = numpy_to_o3d(scan_k)
        tgt = numpy_to_o3d(scan_k1)

        src_down = matcher.preprocess_point_cloud(src)
        tgt_down = matcher.preprocess_point_cloud(tgt)

        import open3d as o3d
        icp_result = o3d.pipelines.registration.registration_icp(
            src_down, tgt_down,
            matcher.voxel_size * 1.5,
            T_init,   # <-- IMU prediction as initial guess
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        T_icp = icp_result.transformation

        # --- EKF update ---
        ekf.update(T_icp)

        if k % 20 == 0:
            print("frame %d  p=%s  fitness=%.4f" % (k, np.round(ekf.p, 2), icp_result.fitness))

    # --- plot trajectory vs ground truth ---
    traj_ekf = ekf.get_trajectory()

    gt = np.array([loader.get_ground_truth_pose_imu(k)[:3, 3] for k in range(N)])

    plt.figure(figsize=(10, 6))
    plt.plot(gt[:, 0],      gt[:, 1],      "g-",  label="ground truth")
    plt.plot(traj_ekf[:, 0], traj_ekf[:, 1], "b--", label="EKF estimate")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("EKF Trajectory vs Ground Truth")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("trajectory.png", dpi=150)
    plt.show()
    print("done. trajectory.png saved.")