import numpy as np
import os


#configurable base paths, adjust if kitti zips extracted differently
ODOM_SEQ_DIR  = os.path.join(os.path.dirname(__file__), "00")
RAW_DATE_DIR  = os.path.join(os.path.dirname(__file__), "2011_10_03")
RAW_DRIVE_DIR = os.path.join(RAW_DATE_DIR, "2011_10_03_drive_0027_sync")

#odometry has 4541 velodyne frames
N_ODOM_FRAMES = 4541

#earth radius for mercator projection
EARTH_R = 6378137.0

#oxts field names in order
OXTS_FIELDS = [
    "lat", "lon", "alt", "roll", "pitch", "yaw",
    "vn", "ve", "vf", "vl", "vu",
    "ax", "ay", "az", "af", "al", "au",
    "wx", "wy", "wz", "wf", "wl", "wu",
    "pos_accuracy", "vel_accuracy",
    "navstat", "numsats", "posmode", "velmode", "orimode"
]


def _parse_RT(calib_path):

    #parse R and T from a kitti raw calibration file
    R = None
    t = None

    with open(calib_path, "r") as f:

        for line in f:

            if line.startswith("R:"):
                R = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 3)

            elif line.startswith("T:"):
                t = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 1)

    if R is None or t is None:
        raise ValueError("R or T not found in %s" % calib_path)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3:] = t
    return T


def _parse_cam_to_cam(calib_path):

    #parse R_rect_00 from calib_cam_to_cam.txt
    with open(calib_path, "r") as f:

        for line in f:

            if line.startswith("R_rect_00:"):

                vals = [float(x) for x in line.strip().split()[1:]]
                R_rect = np.eye(4)
                R_rect[:3, :3] = np.array(vals).reshape(3, 3)
                return R_rect

    raise ValueError("R_rect_00 not found in %s" % calib_path)


def _load_timestamps(ts_path):

    #load timestamps, handles datetime format from raw data
    from datetime import datetime

    timestamps = []

    with open(ts_path, "r") as f:

        for line in f:

            line = line.strip()
            if not line:
                continue

            #try plain float first (odometry times.txt)
            try:
                timestamps.append(float(line))
                continue
            except ValueError:
                pass

            #datetime format, truncate to 6 decimal places (microseconds)
            parts = line.split(".")
            frac = parts[1][:6] if len(parts) > 1 else "0"
            trimmed = parts[0] + "." + frac
            dt = datetime.strptime(trimmed, "%Y-%m-%d %H:%M:%S.%f")
            timestamps.append(dt.timestamp())

    return np.array(timestamps)


def _load_single_oxts(filepath):

    #parse one oxts txt file into a dict
    with open(filepath, "r") as f:
        vals = [float(x) for x in f.read().strip().split()]

    data = {}
    for i, name in enumerate(OXTS_FIELDS):
        data[name] = vals[i]

    return data


def _latlon_to_mercator(lat, lon):

    #convert lat/lon (degrees) to mercator x/y (meters)
    mx = EARTH_R * np.radians(lon)
    my = EARTH_R * np.log(np.tan(np.pi / 4.0 + np.radians(lat) / 2.0))
    return mx, my


def _rot_from_oxts(roll, pitch, yaw):

    #rotation matrix from euler angles (roll pitch yaw)
    #Rx(roll) * Ry(pitch) * Rz(yaw) following kitti convention
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    Rz = np.array([[cy, -sy, 0],
                    [sy,  cy, 0],
                    [0,   0,  1]])

    Ry = np.array([[ cp, 0, sp],
                    [  0, 1,  0],
                    [-sp, 0, cp]])

    Rx = np.array([[1,  0,   0],
                    [0, cr, -sr],
                    [0, sr,  cr]])

    return Rz @ Ry @ Rx


def _poses_from_oxts(oxts_list):

    #compute 4x4 poses in local ENU frame from oxts gps/ins
    #first frame becomes the origin
    poses = []
    origin_set = False
    mx0 = 0.0
    my0 = 0.0

    for ox in oxts_list:

        mx, my = _latlon_to_mercator(ox["lat"], ox["lon"])

        if not origin_set:
            mx0 = mx
            my0 = my
            origin_set = True

        #position in local frame
        tx = mx - mx0
        ty = my - my0
        tz = ox["alt"]

        #rotation from euler angles
        R = _rot_from_oxts(ox["roll"], ox["pitch"], ox["yaw"])

        T = np.eye(4)
        T[:3, :3] = R
        T[0, 3]   = tx
        T[1, 3]   = ty
        T[2, 3]   = tz

        poses.append(T)

    return poses


class KittiLoader:

    def __init__(self, odom_dir=None, raw_date_dir=None, raw_drive_dir=None):

        #allow overriding default paths
        self.odom_dir      = odom_dir or ODOM_SEQ_DIR
        self.raw_date_dir  = raw_date_dir or RAW_DATE_DIR
        self.raw_drive_dir = raw_drive_dir or RAW_DRIVE_DIR

        #load calibration transforms
        self._load_calibration()

        #load all oxts data (all readings, not just synced)
        self._load_oxts()

        #load timestamps for both sensors
        self._load_timestamps()

        #compute ground truth poses from oxts
        self._compute_gt_poses()

    def _load_calibration(self):

        #imu to velo from raw calibration
        imu_velo_path = os.path.join(self.raw_date_dir, "calib_imu_to_velo.txt")
        self.T_imu_to_velo = _parse_RT(imu_velo_path)

        #velo to unrectified cam0 from raw calibration
        velo_cam_path = os.path.join(self.raw_date_dir, "calib_velo_to_cam.txt")
        T_velo_to_cam_unrect = _parse_RT(velo_cam_path)

        #rectification rotation for cam0
        cam_cam_path = os.path.join(self.raw_date_dir, "calib_cam_to_cam.txt")
        R_rect = _parse_cam_to_cam(cam_cam_path)

        #velo to rectified cam0 (this is what odometry calib.txt calls Tr)
        self.T_velo_to_cam = R_rect @ T_velo_to_cam_unrect

        #derived transforms
        self.T_velo_to_imu = np.linalg.inv(self.T_imu_to_velo)
        self.T_imu_to_cam  = self.T_velo_to_cam @ self.T_imu_to_velo
        self.T_cam_to_imu  = np.linalg.inv(self.T_imu_to_cam)

        #store in dict for easy access
        self.transforms = {
            "T_velo_to_cam":  self.T_velo_to_cam,
            "T_imu_to_velo":  self.T_imu_to_velo,
            "T_velo_to_imu":  self.T_velo_to_imu,
            "T_imu_to_cam":   self.T_imu_to_cam,
            "T_cam_to_imu":   self.T_cam_to_imu,
        }

    def _load_oxts(self):

        #load ALL oxts readings (works for both synced and unsynced)
        oxts_dir = os.path.join(self.raw_drive_dir, "oxts", "data")
        files = sorted(os.listdir(oxts_dir))

        self.oxts_raw = []
        for fname in files:

            data = _load_single_oxts(os.path.join(oxts_dir, fname))
            self.oxts_raw.append(data)

        self.n_imu = len(self.oxts_raw)

    def _load_timestamps(self):

        #imu timestamps from oxts (all readings)
        imu_ts_path = os.path.join(self.raw_drive_dir, "oxts", "timestamps.txt")
        self.imu_timestamps = _load_timestamps(imu_ts_path)

        #lidar timestamps: try velodyne_points/timestamps.txt first (unsynced)
        #then fall back to oxts timestamps (synced, 1:1 with imu)
        velo_ts_path = os.path.join(self.raw_drive_dir, "velodyne_points", "timestamps.txt")

        if os.path.exists(velo_ts_path):
            self.lidar_timestamps = _load_timestamps(velo_ts_path)[:N_ODOM_FRAMES]
        else:
            self.lidar_timestamps = self.imu_timestamps[:N_ODOM_FRAMES]

        #build mapping from lidar frame index to imu index range
        #for each lidar frame k, find all imu readings between lidar k and k+1
        self._build_imu_index_map()

    def _build_imu_index_map(self):

        #for each lidar frame k, find imu indices whose timestamps
        #fall in [lidar_t[k], lidar_t[k+1])
        #works for both synced (1:1) and unsynced (many:1)
        self.imu_ranges = []

        for k in range(N_ODOM_FRAMES - 1):

            t_start = self.lidar_timestamps[k]
            t_end   = self.lidar_timestamps[k + 1]

            #binary search for start and end indices in imu timestamps
            i_start = np.searchsorted(self.imu_timestamps, t_start, side="left")
            i_end   = np.searchsorted(self.imu_timestamps, t_end, side="left")

            self.imu_ranges.append((i_start, i_end))

    def _compute_gt_poses(self):

        #compute ground truth poses from oxts gps/ins data
        #use only the readings that correspond to lidar frames
        #for synced data this is 1:1, for unsynced we pick closest
        lidar_oxts = []

        for k in range(N_ODOM_FRAMES):

            #find closest imu reading to this lidar timestamp
            idx = np.argmin(np.abs(self.imu_timestamps - self.lidar_timestamps[k]))
            lidar_oxts.append(self.oxts_raw[idx])

        #store the lidar aligned oxts indices for get_imu_data
        self._lidar_to_imu_idx = []
        for k in range(N_ODOM_FRAMES):

            idx = np.argmin(np.abs(self.imu_timestamps - self.lidar_timestamps[k]))
            self._lidar_to_imu_idx.append(idx)

        #poses in imu/body frame (ENU world coordinates)
        self.gt_poses_imu = _poses_from_oxts(lidar_oxts)

        #also store in cam frame for compatibility
        self.gt_poses = []
        for T_w_imu in self.gt_poses_imu:

            T_w_cam = T_w_imu @ self.T_cam_to_imu
            self.gt_poses.append(T_w_cam)

    def n_frames(self):

        return N_ODOM_FRAMES

    def get_lidar_scan(self, k):

        #load velodyne bin file for frame k, returns Nx4 (x,y,z,reflectance)
        assert 0 <= k < N_ODOM_FRAMES, "frame %s out of range" % k

        fname = "%06d.bin" % k
        velo_dir = os.path.join(self.odom_dir, "velodyne")
        fpath = os.path.join(velo_dir, fname)

        scan = np.fromfile(fpath, dtype=np.float32).reshape(-1, 4)
        return scan

    def get_imu_data(self, k):

        #get imu reading closest to lidar frame k in body frame
        assert 0 <= k < N_ODOM_FRAMES, "frame %s out of range" % k

        idx = self._lidar_to_imu_idx[k]
        raw = self.oxts_raw[idx]

        #oxts body frame: af/al/au are forward/left/up accel
        #wf/wl/wu are forward/left/up angular rates
        accel_body = np.array([raw["af"], raw["al"], raw["au"]])
        gyro_body  = np.array([raw["wf"], raw["wl"], raw["wu"]])

        result = {
            "accel":     accel_body,
            "gyro":      gyro_body,
            "timestamp": self.imu_timestamps[idx],
            "raw":       raw,
        }
        return result

    def get_imu_between(self, k_start, k_end):

        #get all imu measurements between lidar frames k_start and k_end
        #for synced data returns 1 measurement per interval
        #for unsynced data returns all high rate imu readings in that window
        assert 0 <= k_start < k_end <= N_ODOM_FRAMES

        measurements = []

        for k in range(k_start, k_end):

            i_start, i_end = self.imu_ranges[k]

            #if no imu readings in this interval, use the closest one
            if i_start >= i_end:

                idx = self._lidar_to_imu_idx[k]
                dt = self.lidar_timestamps[k + 1] - self.lidar_timestamps[k]

                measurements.append({
                    "accel":     np.array([self.oxts_raw[idx]["af"],
                                           self.oxts_raw[idx]["al"],
                                           self.oxts_raw[idx]["au"]]),
                    "gyro":      np.array([self.oxts_raw[idx]["wf"],
                                           self.oxts_raw[idx]["wl"],
                                           self.oxts_raw[idx]["wu"]]),
                    "dt":        dt,
                    "timestamp": self.imu_timestamps[idx],
                })
                continue

            for i in range(i_start, i_end):

                raw = self.oxts_raw[i]

                #dt to next imu reading, or to lidar frame end
                if i < i_end - 1:
                    dt = self.imu_timestamps[i + 1] - self.imu_timestamps[i]
                else:
                    dt = self.lidar_timestamps[k + 1] - self.imu_timestamps[i]

                #clamp dt to avoid negatives from timestamp jitter
                dt = max(dt, 1e-6)

                measurements.append({
                    "accel":     np.array([raw["af"], raw["al"], raw["au"]]),
                    "gyro":      np.array([raw["wf"], raw["wl"], raw["wu"]]),
                    "dt":        dt,
                    "timestamp": self.imu_timestamps[i],
                })

        return measurements

    def get_ground_truth_pose(self, k):

        #get 4x4 ground truth pose for frame k (world to cam0 frame)
        assert 0 <= k < len(self.gt_poses), "frame %s out of range" % k
        return self.gt_poses[k].copy()

    def get_ground_truth_pose_imu(self, k):

        #get ground truth pose in imu/body frame (from oxts directly)
        assert 0 <= k < len(self.gt_poses_imu), "frame %s out of range" % k
        return self.gt_poses_imu[k].copy()

    def get_transform(self, name):

        #get a named calibration transform
        if name not in self.transforms:
            raise KeyError("unknown transform %s, available: %s" % (name, list(self.transforms.keys())))

        return self.transforms[name].copy()

    def get_lidar_timestamp(self, k):

        assert 0 <= k < N_ODOM_FRAMES
        return self.lidar_timestamps[k]

    def get_imu_timestamp(self, k):

        assert 0 <= k < N_ODOM_FRAMES
        return self.imu_timestamps[self._lidar_to_imu_idx[k]]


if __name__ == "__main__":

    print("loading kitti data...")
    loader = KittiLoader()

    print("frames: %s" % loader.n_frames())
    print("total imu readings: %s" % loader.n_imu)
    print("T_imu_to_velo:")
    print(loader.get_transform("T_imu_to_velo"))

    #test loading first scan
    scan = loader.get_lidar_scan(0)
    print("scan 0 shape: %s" % str(scan.shape))

    #test imu data
    imu = loader.get_imu_data(0)
    print("imu 0 accel: %s" % str(imu["accel"]))
    print("imu 0 gyro: %s" % str(imu["gyro"]))

    #test imu between, print count for a few intervals
    for k in [0, 100, 1000, 4000]:

        between = loader.get_imu_between(k, k + 1)
        print("imu readings between frames %s and %s: %s" % (k, k + 1, len(between)))

    #test ground truth
    pose = loader.get_ground_truth_pose_imu(0)
    print("gt pose imu frame 0:\n%s" % str(pose))

    #print timestamp range
    t0 = loader.get_lidar_timestamp(0)
    t1 = loader.get_lidar_timestamp(loader.n_frames() - 1)
    print("total time: %.1f seconds" % (t1 - t0))
