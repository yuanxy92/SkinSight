import numpy as np
import argparse

import os
import glob
import threading
import torch
from tqdm.auto import tqdm
import cv2
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
base_models_path = os.path.join(current_dir, 'base_models')
if base_models_path not in sys.path:
    sys.path.append(base_models_path)

from base_models.base_model import Pi3Adapter

import numpy as np

from loop_utils.sim3loop import Sim3LoopOptimizer
from loop_utils.sim3utils import *
from datetime import datetime

from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

from loop_utils.config_utils import load_config
from pathlib import Path
import time
# import utils3d
# from geometry_torch import recover_focal_shift

def remove_duplicates(data_list):
    """
        data_list: [(67, (3386, 3406), 48, (2435, 2455)), ...]
    """
    seen = {} 
    result = []
    
    for item in data_list:
        if item[0] == item[2]:
            continue

        key = (item[0], item[2])
        
        if key not in seen.keys():
            seen[key] = True
            result.append(item)
    
    return result


def extract_p2_k_matrix(calib_path):
    """from calib.txt get K  (kitti)"""

    calib_path = Path(calib_path)
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    with open(calib_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('P2:'):
                values = line.split(':')[1].split()
                values = [float(v) for v in values]
                p2_matrix = np.array(values).reshape(3, 4)
                k_matrix = p2_matrix[:3, :3]
                return k_matrix, p2_matrix

    raise ValueError("P2 not found in calibration file")

import multiprocessing as mp
from multiprocessing import shared_memory, Queue, Process, resource_tracker

def pack_shm(data_dict):
    shm_metadata = {}
    shm_objects = []
    for key, arr in data_dict.items():
        if (not isinstance(arr, np.ndarray)) or arr.ndim == 0:
            shm_metadata[key] = {"type": "raw", "data": arr}
            continue
        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        shm_objects.append(shm)
        shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        shm_arr[:] = arr[:]
        shm_metadata[key] = {
            "type": "shm",
            "name": shm.name,
            "shape": arr.shape,
            "dtype": str(arr.dtype)
        }
    return shm_metadata, shm_objects

def unpack_shm(shm_metadata):
    result_dict = {}
    shm_refs = []
    for key, info in shm_metadata.items():
        if info["type"] == "raw":
            result_dict[key] = info["data"]
            continue
        shm = shared_memory.SharedMemory(name=info["name"])
        shm_refs.append(shm)
        resource_tracker.unregister(shm._name, "shared_memory") 
        arr = np.ndarray(info["shape"], dtype=info["dtype"], buffer=shm.buf)
        result_dict[key] = arr
    return result_dict, shm_refs


def compute_alignment(chunk_data1, chunk_data2, overlap, config):
# 提取两个分块在重叠区域（Overlap）的点云和置信度
            point_map1 = chunk_data1['world_points'][-overlap:]
            point_map2 = chunk_data2['world_points'][:overlap]
            conf1 = chunk_data1['world_points_conf'][-overlap:]
            conf2 = chunk_data2['world_points_conf'][:overlap]

            # 如果存在掩码（如天空掩码），则在对齐时考虑掩码
            mask = None
            if chunk_data1["mask"] is not None:
                mask1 = chunk_data1["mask"][-overlap:]
                mask2 = chunk_data2["mask"][:overlap]
                mask = mask1.squeeze() & mask2.squeeze()

            # 根据置信度中值设置过滤阈值，剔除低质量点
            if config['Model']['Pointcloud_Save'].get('use_conf_filter', True):
                conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.1
            else:
                conf_threshold = -1.0
            
            # 使用加权 Umeyama 算法计算 Sim3 变换（缩放 s, 旋转 R, 平移 t）
            s, R, t = weighted_align_point_maps(point_map1, 
                                                conf1, 
                                                point_map2, 
                                                conf2,
                                                mask,
                                                conf_threshold=conf_threshold,
                                                config=config)
            
            return s, R, t


def alignment_process(queue1, queue2):
    print("Alignment 进程启动")
    f = open(os.devnull, 'w')
    sys.stdout = f
    sys.stderr = f
    while True:
        task = queue1.get()
        data1, shm_refs1 = unpack_shm(task["data1"])
        data2, shm_refs2 = unpack_shm(task["data2"])
        result = compute_alignment(data1, data2, task["overlap"], task["config"])
        for s in shm_refs1 + shm_refs2:
            s.close() 
        queue2.put({
            "srt": result,
            "idx": task["idx"]
        })
        

class LongSeqResult:
    def __init__(self):
        self.combined_extrinsics = []
        self.combined_intrinsics = []
        self.combined_depth_maps = []
        self.combined_depth_confs = []
        self.combined_world_points = []
        self.combined_world_points_confs = []
        self.all_camera_poses = []
        self.all_camera_intrinsics = [] 

class SkinSightRecon:
    def __init__(self, image_dir, save_dir, config):
        self.config = config

        self.chunk_size = self.config['Model']['chunk_size']
        self.overlap = self.config['Model']['overlap']
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.sky_mask = False
        self.useDBoW = self.config['Model']['useDBoW']

        self.img_dir = image_dir
        self.img_list = None
        self.output_dir = save_dir

        self.result_unaligned_dir = os.path.join(save_dir, '_tmp_results_unaligned')
        self.result_aligned_dir = os.path.join(save_dir, '_tmp_results_aligned')
        self.result_loop_dir = os.path.join(save_dir, '_tmp_results_loop')
        self.pcd_dir = os.path.join(save_dir, 'pcd')
        os.makedirs(self.result_unaligned_dir, exist_ok=True)
        os.makedirs(self.result_aligned_dir, exist_ok=True)
        os.makedirs(self.result_loop_dir, exist_ok=True)
        os.makedirs(self.pcd_dir, exist_ok=True)
        
        self.all_camera_poses = []
        self.all_camera_intrinsics = [] 
        
        self.delete_temp_files = self.config['Model']['delete_temp_files']

        if self.config['Weights']['model'] == 'Pi3':
            self.model = Pi3Adapter(self.config)
        else:
            raise ValueError(f"Unsupported model: {self.config['Weights']['model']}. ")

        self.skyseg_session = None
        
        self.chunk_indices = None # [(begin_idx, end_idx), ...]

        self.loop_list = [] # e.g. [(1584, 139), ...]

        self.loop_optimizer = Sim3LoopOptimizer(self.config)

        self.sim3_list = [] # [(s [1,], R [3,3], T [3,]), ...]

        self.loop_sim3_list = [] # [(chunk_idx_a, chunk_idx_b, s [1,], R [3,3], T [3,]), ...]

        self.loop_predict_list = []

        self.loop_enable = self.config['Model']['loop_enable']

        print('init done.')

    def process_single_chunk(self, range_1, chunk_idx=None, range_2=None, is_loop=False):
        start_idx, end_idx = range_1
        chunk_image_paths = self.img_list[start_idx:end_idx]
        if range_2 is not None:
            start_idx, end_idx = range_2
            chunk_image_paths += self.img_list[start_idx:end_idx]

        predictions = self.model.infer_chunk(chunk_image_paths)

        

        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)
        
        # Save predictions to disk instead of keeping in memory
        if is_loop:
            save_dir = self.result_loop_dir
            filename = f"loop_{range_1[0]}_{range_1[1]}_{range_2[0]}_{range_2[1]}.npy"
        else:
            if chunk_idx is None:
                raise ValueError("chunk_idx must be provided when is_loop is False")
            save_dir = self.result_unaligned_dir
            filename = f"chunk_{chunk_idx}.npy"
        
        save_path = os.path.join(save_dir, filename)
                    
        if not is_loop and range_2 is None:
            extrinsics = predictions['extrinsic']
            intrinsics = predictions['intrinsic']
            
            chunk_range = self.chunk_indices[chunk_idx]
            self.all_camera_poses.append((chunk_range, extrinsics))
            self.all_camera_intrinsics.append((chunk_range, intrinsics))

        predictions['depth'] = np.squeeze(predictions['depth'])

        #np.save(save_path, predictions)
        
        #return predictions if is_loop or range_2 is not None else None

        return predictions
    
    def process_long_sequence(self):
        # --- 计算分块索引 ---
        if self.overlap >= self.chunk_size:
            raise ValueError(f"[SETTING ERROR] Overlap ({self.overlap}) must be less than chunk size ({self.chunk_size})")
        
        if len(self.img_list) <= self.chunk_size:
            num_chunks = 1
            self.chunk_indices = [(0, len(self.img_list))]
        else:
            step = self.chunk_size - self.overlap
            num_chunks = (len(self.img_list) - self.overlap + step - 1) // step
            self.chunk_indices = []
            for i in range(num_chunks):
                start_idx = i * step
                end_idx = min(start_idx + self.chunk_size, len(self.img_list))
                self.chunk_indices.append((start_idx, end_idx))

        # --- 顺序推理同时计算对齐 ---
        chunk_lst = [] # 主线程里使用的chunk
        chunk_shm_lst = [] # 被复制到共享内存的chunk的标识符
        shm_obj_lst = [] # 共享内存对象
        
        queue1 = Queue() # 输入相邻chunk
        queue2 = Queue() # 输出相邻变换
        process_align_num = 3
        process_align_lst = [Process(target=alignment_process, args=(queue1, queue2)) for _ in range(process_align_num)]
        for process in process_align_lst: process.start()

        for chunk_idx in range(len(self.chunk_indices)):
            print(f'[Progress]: {chunk_idx}/{len(self.chunk_indices)-1}')
            start = time.perf_counter()
            # 处理单个chunk
            chunk = self.process_single_chunk(self.chunk_indices[chunk_idx], chunk_idx=chunk_idx)
            torch.cuda.empty_cache()
            chunk_lst.append(chunk)
            # 复制到共享内存
            chunk_shm, shm_objs = pack_shm(chunk)
            chunk_shm_lst.append(chunk_shm)
            shm_obj_lst += shm_objs
            # 第2个chunk开始向进程发送任务
            if chunk_idx >= 1:
                chunk_shm_data1 = chunk_shm_lst[chunk_idx-1]
                chunk_shm_data2 = chunk_shm
                task = {
                    "data1": chunk_shm_data1,
                    "data2": chunk_shm_data2,
                    "overlap": self.overlap,
                    "config": self.config,
                    "idx": chunk_idx
                }
                queue1.put(task)

            end = time.perf_counter()
            print(f"Chunk {chunk_idx} processed in {end - start:.2f} seconds")
        # 清理torch内存
        del self.model 
        torch.cuda.empty_cache()

        start = time.perf_counter()
        res_lst = []
        for i in range(len(self.chunk_indices) - 1):
            res = queue2.get()
            res_lst.append(res)
            print(f"Alignment {res['idx']} processed")
        res_lst.sort(key=lambda res: res["idx"])
        self.sim3_list = [res["srt"] for res in res_lst]
        end = time.perf_counter()
        print(f"Alignment processed in {end - start:.2f} seconds")

        # 清理线程与共享内存
        for process in process_align_lst: process.terminate()
        for shm in shm_obj_lst:
            shm.close()
            shm.unlink()
        chunk_shm_lst = []
        shm_obj_lst = []


        # --- 应用变换并保存结果 ---
        print('Apply alignment')
        # 将局部 Sim3 变换累加，得到各分块相对于第一块的全局变换矩阵
        self.sim3_list = accumulate_sim3_transforms(self.sim3_list)

        with open(os.path.join(self.output_dir, "sim3res.txt"), "xt") as temp:
            temp.write("\n".join([str(x) for x in self.sim3_list]))

        for chunk_idx in range(len(self.chunk_indices) - 1):
            print(f'Applying {chunk_idx + 1} -> {chunk_idx} (Total {len(self.chunk_indices) - 1})')
            s, R, t = self.sim3_list[chunk_idx]

            # 加载待对齐的分块数据
            chunk_data = chunk_lst[chunk_idx + 1]

            # 对该块的所有 3D 世界坐标点应用 Sim3 变换
            chunk_data['world_points'] = apply_sim3_direct(chunk_data['world_points'], s, R, t)

            # 保存对齐后的结果
            #aligned_path = os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx + 1}.npy")
            #np.save(aligned_path, chunk_data)

            # 特殊处理：保存第一个分块（它是参考坐标系，无需变换）
            if chunk_idx == 0:
                chunk_data_first = chunk_lst[0]
                #np.save(os.path.join(self.result_aligned_dir, "chunk_0.npy"), chunk_data_first)

                # 将第一个块转换为 PLY 点云文件保存
                points_first = chunk_data_first['world_points'].reshape(-1, 3)
                colors_first = (chunk_data_first['images'].transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
                confs_first = chunk_data_first['world_points_conf'].reshape(-1)
                ply_path_first = os.path.join(self.pcd_dir, f'0_pcd.ply')
                save_confident_pointcloud_batch(
                    points=points_first,
                    colors=colors_first,
                    confs=confs_first,
                    output_path=ply_path_first,
                    conf_threshold=(np.mean(confs_first) * self.config['Model']['Pointcloud_Save']['conf_threshold_coef']
                        if self.config['Model']['Pointcloud_Save'].get('use_conf_filter', True) else -1.0),
                    sample_ratio=self.config['Model']['Pointcloud_Save']['sample_ratio']
                )

            # 将后续对齐的分块转换为 PLY 点云文件保存
            #aligned_chunk_data = np.load(os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx+1}.npy"),
            #                                 allow_pickle=True).item() if chunk_idx > 0 else chunk_data_first
            #aligned_chunk_data = np.load(os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx+1}.npy"), allow_pickle=True).item()
            aligned_chunk_data = chunk_data

            points = aligned_chunk_data['world_points'].reshape(-1, 3)
            colors = (aligned_chunk_data['images'].transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
            confs = aligned_chunk_data['world_points_conf'].reshape(-1)
            ply_path = os.path.join(self.pcd_dir, f'{chunk_idx + 1}_pcd.ply')
            save_confident_pointcloud_batch(
                points=points,
                colors=colors,
                confs=confs,
                output_path=ply_path,
                conf_threshold=(np.mean(confs) * self.config['Model']['Pointcloud_Save']['conf_threshold_coef']
                    if self.config['Model']['Pointcloud_Save'].get('use_conf_filter', True) else -1.0),
                sample_ratio=self.config['Model']['Pointcloud_Save']['sample_ratio']
            )

        # 保存相机轨迹位姿
        self.save_camera_poses()
        
        print('Done.')

    def run(self):
        print(f"Loading images from {self.img_dir}...")
        self.img_list = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")) +
                               glob.glob(os.path.join(self.img_dir, "*.png")))
        # print(self.img_list)
        if len(self.img_list) == 0:
            raise ValueError(f"[DIR EMPTY] No images found in {self.img_dir}!")
        print(f"Found {len(self.img_list)} images")

        if self.loop_enable:
            self.get_loop_pairs()

            if self.useDBoW:
                self.retrieval.close()  # Save CPU Memory
                gc.collect()
            else:
                del self.loop_detector  # Save GPU Memory
        torch.cuda.empty_cache()
        print('Loading model...')
        self.model.load()

        if self.config['Model']['calib']:
            calib_path = Path(self.img_dir).parent / 'calib.txt'
            k, p2_matrix = extract_p2_k_matrix(calib_path)
            self.model.k = k

        self.process_long_sequence()

    def save_camera_poses(self):
        '''
        Save camera poses from all chunks to txt and ply files
        - txt file: Each line contains a 4x4 C2W matrix flattened into 16 numbers
        - ply file: Camera poses visualized as points with different colors for each chunk
        '''
        chunk_colors = [
            [255, 0, 0],  # Red
            [0, 255, 0],  # Green
            [0, 0, 255],  # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [128, 0, 0],  # Dark Red
            [0, 128, 0],  # Dark Green
            [0, 0, 128],  # Dark Blue
            [128, 128, 0],  # Olive
        ]
        print("Saving all camera poses to txt file...")

        all_poses = [None] * len(self.img_list)
        all_intrinsics = [None] * len(self.img_list)

        first_chunk_range, first_chunk_extrinsics = self.all_camera_poses[0]
        _, first_chunk_intrinsics = self.all_camera_intrinsics[0]
        for i, idx in enumerate(range(first_chunk_range[0], first_chunk_range[1])):
            c2w = first_chunk_extrinsics[i]
            all_poses[idx] = c2w
            if first_chunk_intrinsics is not None:
                all_intrinsics[idx] = first_chunk_intrinsics[i]

        for chunk_idx in range(1, len(self.all_camera_poses)):
            chunk_range, chunk_extrinsics = self.all_camera_poses[chunk_idx]
            _, chunk_intrinsics = self.all_camera_intrinsics[chunk_idx]
            s, R, t = self.sim3_list[
                chunk_idx - 1]  # When call self.save_camera_poses(), all the sim3 are aligned to the first chunk.

            S = np.eye(4)
            S[:3, :3] = s * R
            S[:3, 3] = t

            for i, idx in enumerate(range(chunk_range[0], chunk_range[1])):
                c2w = chunk_extrinsics[i]  #

                transformed_c2w = S @ c2w  # Be aware of the left multiplication!
                transformed_c2w[:3, :3] /= s  # Normalize rotation

                all_poses[idx] = transformed_c2w
                if chunk_intrinsics is not None:
                    all_intrinsics[idx] = chunk_intrinsics[i]

        poses_path = os.path.join(self.output_dir, 'camera_poses.txt')
        with open(poses_path, 'w') as f:
            for pose in all_poses:
                flat_pose = pose.flatten()
                f.write(' '.join([str(x) for x in flat_pose]) + '\n')

        print(f"Camera poses saved to {poses_path}")
        if all_intrinsics[0] is not None:
            intrinsics_path = os.path.join(self.output_dir, 'intrinsic.txt')
            with open(intrinsics_path, 'w') as f:
                for intrinsic in all_intrinsics:
                    fx = intrinsic[0, 0]
                    fy = intrinsic[1, 1]
                    cx = intrinsic[0, 2]
                    cy = intrinsic[1, 2]
                    f.write(f'{fx} {fy} {cx} {cy}\n')
            print(f"Camera intrinsics saved to {intrinsics_path}")

        ply_path = os.path.join(self.output_dir, 'camera_poses.ply')
        with open(ply_path, 'w') as f:
            # Write PLY header
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {len(all_poses)}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('end_header\n')

            color = chunk_colors[0]
            for pose in all_poses:
                position = pose[:3, 3]
                f.write(f'{position[0]} {position[1]} {position[2]} {color[0]} {color[1]} {color[2]}\n')

        print(f"Camera poses visualization saved to {ply_path}")

    def close(self):
        if not self.delete_temp_files:
            return
        
        total_space = 0

        print(f'Deleting the temp files under {self.result_unaligned_dir}')
        for filename in os.listdir(self.result_unaligned_dir):
            file_path = os.path.join(self.result_unaligned_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)

        print(f'Deleting the temp files under {self.result_aligned_dir}')
        for filename in os.listdir(self.result_aligned_dir):
            file_path = os.path.join(self.result_aligned_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)

        print(f'Deleting the temp files under {self.result_loop_dir}')
        for filename in os.listdir(self.result_loop_dir):
            file_path = os.path.join(self.result_loop_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)
        print('Deleting temp files done.')

        print(f"Saved disk space: {total_space/1024/1024/1024:.4f} GiB")


import shutil
def copy_file(src_path, dst_dir):
    try:
        os.makedirs(dst_dir, exist_ok=True)
        
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        
        shutil.copy2(src_path, dst_path)
        print(f"config yaml file has been copied to: {dst_path}")
        return dst_path
        
    except FileNotFoundError:
        print("File Not Found")
    except PermissionError:
        print("Permission Error")
    except Exception as e:
        print(f"Copy Error: {e}")

if __name__ == '__main__':
    # python skinsight_recon.py --image_dir ../data/fig3 --config ./configs/base_config.yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Image path')
    parser.add_argument('--config', type=str, required=False, default='./configs/base_config.yaml',
                        help='config path')
    args = parser.parse_args()

    config = load_config(args.config)

    image_dir = args.image_dir
    path = image_dir.split("/")
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    save_dir = '../Results/'
    

    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
        print(f'The exp will be saved under dir: {save_dir}')
        copy_file(args.config, save_dir)

    if config['Model']['align_method'] == 'numba':
        warmup_numba()

    ss = SkinSightRecon(image_dir, save_dir, config)
    ss.run()
    ss.close()

    del ss
    torch.cuda.empty_cache()
    gc.collect()

    all_ply_path = os.path.join(save_dir, f'pcd/combined_pcd.ply')
    input_dir = os.path.join(save_dir, f'pcd')
    print("Saving all the point clouds")
    merge_ply_files(input_dir, all_ply_path)
    print('All done.')
    sys.exit()