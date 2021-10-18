import os

import cv2
import numpy as np
import time
import open3d as o3d
from data.kitti_Dataset import Kitti_Dataset


from pathlib import Path
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path_dataset', type=str, default=None, help='dir for the label data', required=True)
args = parser.parse_args()

# 根据偏航角计算旋转矩阵（逆时针旋转）
def rot_y(rotation_y):
    cos = np.cos(rotation_y)
    sin = np.sin(rotation_y)
    R = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
    return R



def draw_3dframeworks(vis,points, color):

    position = points
    points_box = np.transpose(position)

    lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                          [0, 4], [1, 5], [2, 6], [3, 7], [0, 5], [1, 4]])

    colors = np.array([color for j in range(len(lines_box))])
    line_set = o3d.geometry.LineSet()

    line_set.points = o3d.utility.Vector3dVector(points_box)
    line_set.lines = o3d.utility.Vector2iVector(lines_box)
    line_set.colors = o3d.utility.Vector3dVector(colors)


    vis.add_geometry(line_set)
    render_option = vis.get_render_option()
    render_option.point_size = 1
    render_option.background_color = np.asarray([0, 0, 0])


if __name__ == "__main__":
    # dir_path ="G:\\czq\\tsinghua\\label_test"

    dir_path = Path(args.path_dataset)
    split = "training"
    dataset = Kitti_Dataset(dir_path, split=split)

    split_real = "kitti"
    dataset_real = Kitti_Dataset(dir_path, split = split_real)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1440, height=1080)

    for index in range(100):
        obj = dataset.get_labels(index)
        img3_d = dataset.get_rgb(index)
        calib1 = dataset.get_calib(index)
        pc = dataset.get_pcs(index)

        obj_real = dataset_real.get_labels(index)

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pc)

        vis.add_geometry(point_cloud)
        for obj_index in range(len(obj)):
            R = rot_y(obj[obj_index].rotation_y)
            h, w, l = obj[obj_index].dimensions[0], obj[obj_index].dimensions[1], obj[obj_index].dimensions[2]
            x = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
            y = [0, 0, 0, 0, -h, -h, -h, -h]
            # y = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
            z = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
            # 得到目标物体经过旋转之后的实际尺寸（得到其在相机坐标系下的实际尺寸）
            corner_3d = np.vstack([x, y, z])
            corner_3d = np.dot(R, corner_3d)

            # 将该物体移动到相机坐标系下的原点处（涉及到坐标的移动，直接相加就行）
            corner_3d[0, :] += obj[obj_index].location[0]
            corner_3d[1, :] += obj[obj_index].location[1]
            corner_3d[2, :] += obj[obj_index].location[2]
            corner_3d = np.vstack((corner_3d, np.zeros((1, corner_3d.shape[-1]))))
            corner_3d[-1][-1] = 1



            inv_Tr = np.zeros_like(calib1.Tr_velo_to_cam)
            inv_Tr[0:3, 0:3] = np.transpose(calib1.Tr_velo_to_cam[0:3, 0:3])
            inv_Tr[0:3, 3] = np.dot(-np.transpose(calib1.Tr_velo_to_cam[0:3, 0:3]), calib1.Tr_velo_to_cam[0:3, 3])

            Y = np.dot(inv_Tr, corner_3d)

            draw_3dframeworks(vis, Y, color = [1.0, 0., 0.])
        for obj_real_index in range(len(obj_real)):

            R_real = rot_y(obj_real[obj_real_index].rotation_y)
            h_real, w_real, l_real = obj_real[obj_real_index].dimensions[0], obj_real[obj_real_index].dimensions[1], obj_real[obj_real_index].dimensions[2]
            x_real = [l_real / 2, l_real / 2, -l_real / 2, -l_real / 2, l_real / 2, l_real / 2, -l_real / 2, -l_real / 2]
            y_real = [0, 0, 0, 0, -h_real, -h_real, -h_real, -h_real]
            z_real = [w_real/ 2, -w_real / 2, -w_real / 2, w_real / 2, w_real / 2, -w_real / 2, -w_real / 2, w_real / 2]

            corner_3d_real = np.vstack([x_real, y_real, z_real])
            corner_3d_real = np.dot(R_real, corner_3d_real)

            corner_3d_real[0, :] += obj_real[obj_real_index].location[0]
            corner_3d_real[1, :] += obj_real[obj_real_index].location[1]
            corner_3d_real[2, :] += obj_real[obj_real_index].location[2]
            corner_3d_real = np.vstack((corner_3d_real, np.zeros((1, corner_3d_real.shape[-1]))))
            corner_3d_real[-1][-1] = 1

            inv_Tr = np.zeros_like(calib1.Tr_velo_to_cam)
            inv_Tr[0:3, 0:3] = np.transpose(calib1.Tr_velo_to_cam[0:3, 0:3])
            inv_Tr[0:3, 3] = np.dot(-np.transpose(calib1.Tr_velo_to_cam[0:3, 0:3]), calib1.Tr_velo_to_cam[0:3, 3])

            Y_real = np.dot(inv_Tr, corner_3d_real)

            draw_3dframeworks(vis, Y_real,color = [0, 1., 0])

        vis.get_render_option().load_from_json('renderoption.json')
        param = o3d.io.read_pinhole_camera_parameters('json/viewpoint1.json')

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.update_renderer()
        vis.run()
        time.sleep(0.5)
        vis.clear_geometries()