import open3d as o3d
import numpy as np
import glob
import os
import copy

bin_path = r"C:\Users\lenovo\Downloads\lidar-imu-fusion-odometry\data\velodyne_00_sequence"

def load_lidar_bins(bin_path, i):
    files = sorted(glob.glob(os.path.join(bin_path, "*.bin")))
    source = np.fromfile(files[i],   dtype=np.float32).reshape(-1, 4)
    target = np.fromfile(files[i+1], dtype=np.float32).reshape(-1, 4)
    return source, target

def numpy_to_o3d(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    return pcd

class Scan_Matching():
    def __init__(self, voxel_size=0.05):
        self.voxel_size = voxel_size

    def preprocess_point_cloud(self, pcd):
        pcd_down = pcd.voxel_down_sample(self.voxel_size)
        radius_normal = self.voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        return pcd_down

    def execute_global_registration(self, source, target):
        source_down = self.preprocess_point_cloud(source)
        target_down = self.preprocess_point_cloud(target)
        distance_threshold = self.voxel_size * 1.5
        result = o3d.pipelines.registration.registration_icp(
            source_down, target_down, distance_threshold, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        translation = result.transformation[:3, 3]
        rotation    = result.transformation[:3, :3]
        return result.transformation, translation, rotation

    def draw_registration_result(self, source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0, 0])  # red
        target_temp.paint_uniform_color([0, 0, 1])  # blue
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])

if __name__ == "__main__":
    scan_matching = Scan_Matching(voxel_size=0.5)

    for i in range(670):
        source_np, target_np = load_lidar_bins(bin_path, i)
        source = numpy_to_o3d(source_np)
        target = numpy_to_o3d(target_np)

        transformation, translation, rotation = scan_matching.execute_global_registration(source, target)
        print(f"Scan {i} -> {i+1}")
        print(f"Translation:\n{translation}")
        print(f"Rotation:\n{rotation}\n")

        scan_matching.draw_registration_result(source, target, transformation)