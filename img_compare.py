import cv2
import numpy as np
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

if __name__ == "__main__":
    # 读取的数据的文件夹
    # 文件夹分为三级 dir_path\ training or test \ calib,label,bin,image
    # G:\czq\tsinghua\label_test
    # dir_path = 'G:\\czq\\tsinghua\\2_caozhenqiang'
    # dir_path =Path(args.path_label)

    dir_path = Path(args.path_dataset)
    # 读取训练集文件夹
    split = "training"
    dataset = Kitti_Dataset(dir_path, split=split)
    split_real = "kitti"
    dataset_real = Kitti_Dataset(dir_path, split=split_real)
    k = 0
    max_num = 100
    # 逐张读入图片
    while True:


        img3_d = dataset.get_rgb(k)
        calib = dataset.get_calib(k)
        # 获取标签数据
        obj = dataset.get_labels(k)
        obj_real = dataset_real.get_labels(k)
        # 逐个读入一副图片中的所有object的标签
        for num in range(len(obj)):
            if obj[num].name == "Car" or obj[num].name == "Pedestrian" or obj[num].name == "Cyclist":

                # step1 得到rot_y旋转矩阵 3*3
                R = rot_y(obj[num].rotation_y)
                # 读取obect物体的高宽长信息
                h, w, l = obj[num].dimensions[0], obj[num].dimensions[1], obj[num].dimensions[2]
                x = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
                y = [0, 0, 0, 0, -h, -h, -h, -h]
                z = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
                # 将xyz转化成3*8的矩阵
                corner_3d = np.vstack([x, y, z])
                # R * X
                corner_3d = np.dot(R, corner_3d)

                # 将该物体移动到相机坐标系下的原点处（涉及到坐标的移动，直接相加就行）
                corner_3d[0, :] += obj[num].location[0]
                corner_3d[1, :] += obj[num].location[1]
                corner_3d[2, :] += obj[num].location[2]

                # 将3d的bbox转换到2d坐标系中（需要用到内参矩阵)
                corner_3d = np.vstack((corner_3d, np.zeros((1, corner_3d.shape[-1]))))
                corner_2d = np.dot(calib.P2, corner_3d)
                # 在像素坐标系下，横坐标x = corner_2d[0, :] /= corner_2d[2, :]
                # 纵坐标的值以此类推
                corner_2d[0, :] /= corner_2d[2, :]
                corner_2d[1, :] /= corner_2d[2, :]

                corner_2d = np.array(corner_2d, dtype=np.int)

                # 绘制立方体边界框
                color1 = [0, 255, 0]
                # 线宽
                thickness = 2

                #绘制3d框
                for corner_i in range(0, 4):
                    i, j = corner_i, (corner_i + 1) % 4
                    cv2.line(img3_d, (corner_2d[0, i], corner_2d[1, i]), (corner_2d[0, j], corner_2d[1, j]), color1, thickness)
                    i, j = corner_i + 4, (corner_i + 1) % 4 + 4
                    cv2.line(img3_d, (corner_2d[0, i], corner_2d[1, i]), (corner_2d[0, j], corner_2d[1, j]), color1, thickness)
                    i, j = corner_i, corner_i + 4
                    cv2.line(img3_d, (corner_2d[0, i], corner_2d[1, i]), (corner_2d[0, j], corner_2d[1, j]), color1, thickness)



                cv2.line(img3_d,(corner_2d[0, 0],corner_2d[1, 0]), (corner_2d[0, 5], corner_2d[1, 5]),color1, thickness)
                cv2.line(img3_d, (corner_2d[0, 1], corner_2d[1, 1]), (corner_2d[0, 4], corner_2d[1, 4]), color1, thickness)
            # print("red")

        for obj_real_index in range(len(obj_real)):
            if obj_real[obj_real_index].name == "Car" or obj_real[obj_real_index].name == "Pedestrian" or obj_real[obj_real_index].name == "Cyclist":


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

                # 将3d的bbox转换到2d坐标系中（需要用到内参矩阵)
                corner_3d_real = np.vstack((corner_3d_real, np.zeros((1, corner_3d_real.shape[-1]))))
                corner_2d_real = np.dot(calib.P2, corner_3d_real)
                # 在像素坐标系下，横坐标x = corner_2d[0, :] /= corner_2d[2, :]
                # 纵坐标的值以此类推
                corner_2d_real[0, :] /= corner_2d_real[2, :]
                corner_2d_real[1, :] /= corner_2d_real[2, :]

                corner_2d_real = np.array(corner_2d_real, dtype=np.int)

                # 绘制立方体边界框
                color = [0, 0, 255]
                # 线宽
                thickness = 2

                #绘制3d框
                for corner_i in range(0, 4):
                    i, j = corner_i, (corner_i + 1) % 4
                    cv2.line(img3_d, (corner_2d_real[0, i], corner_2d_real[1, i]), (corner_2d_real[0, j], corner_2d_real[1, j]), color, thickness)
                    i, j = corner_i + 4, (corner_i + 1) % 4 + 4
                    cv2.line(img3_d, (corner_2d_real[0, i], corner_2d_real[1, i]), (corner_2d_real[0, j], corner_2d_real[1, j]), color, thickness)
                    i, j = corner_i, corner_i + 4
                    cv2.line(img3_d, (corner_2d_real[0, i], corner_2d_real[1, i]), (corner_2d_real[0, j], corner_2d_real[1, j]), color, thickness)


                cv2.line(img3_d,(corner_2d_real[0, 0],corner_2d_real[1, 0]), (corner_2d_real[0, 5], corner_2d_real[1, 5]),color, thickness)
                cv2.line(img3_d, (corner_2d_real[0, 1], corner_2d_real[1, 1]), (corner_2d_real[0, 4], corner_2d_real[1, 4]), color, thickness)
                # print("green")
        cv2.imshow("{}".format(k), img3_d)
        # cv2.waitKey(0)

        key = cv2.waitKey(100) & 0xFF
        if key == ord('d'):
            k += 1
            cv2.destroyAllWindows()
            # if idx == 104:
            #     idx += 1
        if key == ord('a'):
            k -= 1
            cv2.destroyAllWindows()
        if key == ord("s"):
            cv2.imwrite("save_img\\{}.jpg".format(k), img3_d)
        if key == ord('q'):
            break
        if k >= max_num:
            k = max_num - 1
        if k < 0:
            k = 0
        # 读入图片信息
        # print(k)

        # cv2.destroyAllWindows()
