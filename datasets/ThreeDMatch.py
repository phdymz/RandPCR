import os
import os.path
from os.path import join, exists
import numpy as np
import json
import pickle
import random
import open3d as o3d
from utils.pointcloud import make_point_cloud
import torch.utils.data as data
from scipy.spatial.distance import cdist
import glob
from torch.utils.data import Dataset
import open3d
from scipy.spatial.transform import Rotation
import torch, copy


def get_correspondences(src_pcd, tgt_pcd, trans, search_voxel_size, K=None):
    src_pcd.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(tgt_pcd)

    correspondences = []
    for i, point in enumerate(src_pcd.points):
        [count, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            correspondences.append([i, j])

    correspondences = np.array(correspondences)
    correspondences = torch.from_numpy(correspondences)
    return correspondences

def to_array(tensor):
    """
    Conver tensor to array
    """
    if(not isinstance(tensor,np.ndarray)):
        if(tensor.device == torch.device('cpu')):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor

def to_o3d_pcd(xyz):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(to_array(xyz))
    return pcd

def rotation_matrix(augment_axis, augment_rotation):
    angles = np.random.rand(3) * 2 * np.pi * augment_rotation
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    # R = Rx @ Ry @ Rz
    if augment_axis == 1:
        return random.choice([Rx, Ry, Rz]) 
    return Rx @ Ry @ Rz
    
def translation_matrix(augment_translation):
    T = np.random.rand(3) * augment_translation
    return T

class ThreeDMatchDataset(data.Dataset):
    __type__ = 'descriptor'
    
    def __init__(self, 
                 root, 
                 split='train', 
                 num_node=16, 
                 downsample=0.03, 
                 self_augment=False, 
                 augment_noise=0.005,
                 augment_axis=1, 
                 augment_rotation=1.0,
                 augment_translation=0.001,
                 config=None,
                 ):
        self.root = root
        self.split = split
        self.num_node = num_node
        self.downsample = downsample
        self.self_augment = self_augment
        self.augment_noise = augment_noise
        self.augment_axis = augment_axis
        self.augment_rotation = augment_rotation
        self.augment_translation = augment_translation
        self.config = config

        # assert self_augment == False
        
        # containers
        self.ids = []
        self.points = []
        self.src_to_tgt = {}
        
        # load data
        pts_filename = join(self.root, f'3DMatch_{split}_{self.downsample:.3f}_points.pkl')
        keypts_filename = join(self.root, f'3DMatch_{split}_{self.downsample:.3f}_keypts.pkl')

        if exists(pts_filename) and exists(keypts_filename):
            with open(pts_filename, 'rb') as file:
                data = pickle.load(file)
                self.points = [*data.values()]
                self.ids_list = [*data.keys()]
            with open(keypts_filename, 'rb') as file:
                self.correspondences = pickle.load(file)
            print(f"Load PKL file from {pts_filename}")
        else:
            print("PKL file not found.")
            return

        for idpair in self.correspondences.keys():
            src = idpair.split("@")[0]
            tgt = idpair.split("@")[1]
            # add (key -> value)  src -> tgt 
            if src not in self.src_to_tgt.keys():
                self.src_to_tgt[src] = [tgt]
            else:
                self.src_to_tgt[src] += [tgt]

    def __getitem__(self, index):
        src_id = list(self.src_to_tgt.keys())[index]
        
        if random.random() > 0.5:
            tgt_id = self.src_to_tgt[src_id][0]
        else:
            tgt_id = random.choice(self.src_to_tgt[src_id])
            
        src_ind = self.ids_list.index(src_id)
        tgt_ind = self.ids_list.index(tgt_id)
        src_pcd = make_point_cloud(self.points[src_ind])
        if self.self_augment:
            tgt_pcd = make_point_cloud(self.points[src_ind])
            N_src = self.points[src_ind].shape[0]
            N_tgt = self.points[src_ind].shape[0]
            corr = np.array([np.arange(N_src), np.arange(N_src)]).T
        else:
            tgt_pcd = make_point_cloud(self.points[tgt_ind])
            N_src = self.points[src_ind].shape[0]
            N_tgt = self.points[tgt_ind].shape[0]
            corr = self.correspondences[f"{src_id}@{tgt_id}"]
        if N_src > 50000 or N_tgt > 50000:
            return self.__getitem__(int(np.random.choice(self.__len__(), 1)))

        # data augmentation
        gt_trans = np.eye(4).astype(np.float32)
        R = rotation_matrix(self.augment_axis, self.augment_rotation)
        T = translation_matrix(self.augment_translation)
        gt_trans[0:3, 0:3] = R
        gt_trans[0:3, 3] = T
        tgt_pcd.transform(gt_trans)
        src_points = np.array(src_pcd.points)
        tgt_points = np.array(tgt_pcd.points)
        src_points += np.random.rand(src_points.shape[0], 3) * self.augment_noise
        tgt_points += np.random.rand(tgt_points.shape[0], 3) * self.augment_noise
        

        if len(corr) > self.num_node:
            sel_corr = corr[np.random.choice(len(corr), self.num_node, replace=False)]
        else:
            sel_corr = corr

        sel_P_src = src_points[sel_corr[:,0], :].astype(np.float32)
        sel_P_tgt = tgt_points[sel_corr[:,1], :].astype(np.float32)
        dist_keypts = cdist(sel_P_src, sel_P_src)
        # sel_P_src = np.array(src_pcd.points)[sel_src, :].astype(np.float32)
        # sel_P_tgt = np.array(tgt_pcd.points)[sel_tgt, :].astype(np.float32)
                
        pts0 = src_points 
        pts1 = tgt_points
        feat0 = np.ones_like(pts0[:, :1]).astype(np.float32)
        feat1 = np.ones_like(pts1[:, :1]).astype(np.float32)
        if self.self_augment:
            feat0[np.random.choice(pts0.shape[0],int(pts0.shape[0] * 0.99),replace=False)] = 0
            feat1[np.random.choice(pts1.shape[0],int(pts1.shape[0] * 0.99),replace=False)] = 0
        
        return pts0, pts1, feat0, feat1, sel_corr, dist_keypts
            
    def __len__(self):
        return len(self.src_to_tgt.keys())

class ThreeDMatchTestset(data.Dataset):
    __type__ = 'descriptor'
    def __init__(self, 
                root, 
                downsample=0.03, 
                config=None,
                last_scene=False,
                ):
        self.root = root
        self.downsample = downsample
        self.config = config
        
        # contrainer
        self.points = []
        self.ids_list = []
        self.num_test = 0
        
        self.scene_list = [
            '7-scenes-redkitchen',
            'sun3d-home_at-home_at_scan1_2013_jan_1',
            'sun3d-home_md-home_md_scan9_2012_sep_30',
            'sun3d-hotel_uc-scan3',
            'sun3d-hotel_umd-maryland_hotel1',
            'sun3d-hotel_umd-maryland_hotel3',
            'sun3d-mit_76_studyroom-76-1studyroom2',
            'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
        ]
        if last_scene == True:
            self.scene_list = self.scene_list[-1:]
        for scene in self.scene_list:
            self.test_path = f'../data/fragments/{scene}'
            pcd_list = [filename for filename in os.listdir(self.test_path) if filename.endswith('ply')]
            self.num_test += len(pcd_list)

            pcd_list = sorted(pcd_list, key=lambda x: int(x[:-4].split("_")[-1]))
            for i, ind in enumerate(pcd_list):
                pcd = o3d.io.read_point_cloud(join(self.test_path, ind))
                pcd = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size=downsample)
                
                # Load points and labels
                points = np.array(pcd.points)

                self.points += [points]
                self.ids_list += [scene + '/' + ind]
        return

    def __getitem__(self, index):
        pts = self.points[index].astype(np.float32)
        feat = np.ones_like(pts[:, :1]).astype(np.float32)
        return pts, pts, feat, feat, np.array([]), np.array([])

    def __len__(self):
        return self.num_test

class ETHTestset(data.Dataset):
    __type__ = 'descriptor'

    def __init__(self,
                 root,
                 downsample=0.065,
                 config=None
                 ):
        self.root = root
        self.downsample = downsample
        self.config = config

        # contrainer
        self.points = []
        self.ids_list = []
        self.num_test = 0

        pcd_list = glob.glob(root + '*.ply')
        self.num_test += len(pcd_list)
        pcd_list = sorted(pcd_list, key=lambda x: int(x[:-4].split("_")[-1]))

        for i, ind in enumerate(pcd_list):
            pcd = o3d.io.read_point_cloud(ind)
            pcd = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size=downsample)

            # Load points and labels
            points = np.array(pcd.points)
            self.points += [points]


    def __getitem__(self, index):
        pts = self.points[index].astype(np.float32)
        feat = np.ones_like(pts[:, :1]).astype(np.float32)
        return pts, pts, feat, feat, np.array([]), np.array([])

    def __len__(self):
        return self.num_test


class KITTIDataset(Dataset):
    """
    We follow D3Feat to add data augmentation part.
    We first voxelize the pcd and get matches
    Then we apply data augmentation to pcds. KPConv runs over processed pcds, but later for loss computation, we use pcds before data augmentation
    """
    DATA_FILES = {
        'train': './configs/train_kitti.txt',
        'val': './configs/val_kitti.txt',
        'test': './configs/test_kitti.txt'
    }

    def __init__(self, config, split, data_augmentation=True):
        super(KITTIDataset, self).__init__()
        self.config = config
        self.root = os.path.join(config.root, 'dataset')
        self.icp_path = os.path.join(config.root, 'icp')
        if not os.path.exists(self.icp_path):
            os.makedirs(self.icp_path)
        self.voxel_size = config.first_subsampling_dl
        self.data_augmentation = data_augmentation
        self.IS_ODOMETRY = True

        # Initiate containers
        self.files = []
        self.kitti_icp_cache = {}
        self.kitti_cache = {}
        self.prepare_kitti_ply(split)
        self.split = split

    def prepare_kitti_ply(self, split):
        assert split in ['train', 'val', 'test']

        subset_names = open(self.DATA_FILES[split]).read().split()
        for dirname in subset_names:
            drive_id = int(dirname)
            fnames = glob.glob(self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)
            assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
            inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            # get one-to-one distance by comparing the translation vector
            all_odo = self.get_video_odometry(drive_id, return_all=True)
            all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
            Ts = all_pos[:, :3, 3]
            pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
            pdist = np.sqrt(pdist.sum(-1))

            ######################################
            # D3Feat script to generate test pairs
            more_than_10 = pdist > 10
            curr_time = inames[0]
            while curr_time in inames:
                next_time = np.where(more_than_10[curr_time][curr_time:curr_time + 100])[0]
                if len(next_time) == 0:
                    curr_time += 1
                else:
                    next_time = next_time[0] + curr_time - 1

                if next_time in inames:
                    self.files.append((drive_id, curr_time, next_time))
                    curr_time = next_time + 1

        # remove bad pairs
        if split == 'test':
            self.files.remove((8, 15, 58))
        print(f'Num_{split}: {len(self.files)}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        drive = self.files[idx][0]
        t0, t1 = self.files[idx][1], self.files[idx][2]
        all_odometry = self.get_video_odometry(drive, [t0, t1])
        positions = [self.odometry_to_positions(odometry) for odometry in all_odometry]
        fname0 = self._get_velodyne_fn(drive, t0)
        fname1 = self._get_velodyne_fn(drive, t1)

        # XYZ and reflectance
        xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
        xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)

        xyz0 = xyzr0[:, :3]
        xyz1 = xyzr1[:, :3]

        # use ICP to refine the ground_truth pose, for ICP we don't voxllize the point clouds
        key = '%d_%d_%d' % (drive, t0, t1)
        filename = self.icp_path + '/' + key + '.npy'
        if key not in self.kitti_icp_cache:
            if not os.path.exists(filename):
                print('missing ICP files, recompute it')
                M = (self.velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T)
                     @ np.linalg.inv(self.velo2cam)).T
                xyz0_t = self.apply_transform(xyz0, M)
                pcd0 = to_o3d_pcd(xyz0_t)
                pcd1 = to_o3d_pcd(xyz1)
                reg = open3d.registration.registration_icp(pcd0, pcd1, 0.2, np.eye(4),
                                                           open3d.registration.TransformationEstimationPointToPoint(),
                                                           open3d.registration.ICPConvergenceCriteria(
                                                               max_iteration=200))
                pcd0.transform(reg.transformation)
                M2 = M @ reg.transformation
                np.save(filename, M2)
            else:
                M2 = np.load(filename)
            self.kitti_icp_cache[key] = M2
        else:
            M2 = self.kitti_icp_cache[key]

        # refined pose is denoted as trans
        tsfm = M2
        rot = tsfm[:3, :3]
        trans = tsfm[:3, 3][:, None]

        # voxelize the point clouds here
        pcd0 = to_o3d_pcd(xyz0)
        pcd1 = to_o3d_pcd(xyz1)
        pcd0 = pcd0.voxel_down_sample(self.voxel_size)
        pcd1 = pcd1.voxel_down_sample(self.voxel_size)
        src_pcd = np.array(pcd0.points)
        tgt_pcd = np.array(pcd1.points)

        # Get matches
        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)

        rot = rot.astype(np.float32)
        trans = trans.astype(np.float32)

        # add data augmentation
        src_pcd_input = copy.deepcopy(src_pcd)
        tgt_pcd_input = copy.deepcopy(tgt_pcd)

        return src_pcd_input, tgt_pcd_input, src_feats, tgt_feats, rot, trans


    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    @property
    def velo2cam(self):
        try:
            velo2cam = self._velo2cam
        except AttributeError:
            R = np.array([
                7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
                -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
            ]).reshape(3, 3)
            T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
            velo2cam = np.hstack([R, T])
            self._velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
        return self._velo2cam

    def get_video_odometry(self, drive, indices=None, ext='.txt', return_all=False):
        if self.IS_ODOMETRY:
            data_path = self.root + '/poses/%02d.txt' % drive
            if data_path not in self.kitti_cache:
                self.kitti_cache[data_path] = np.genfromtxt(data_path)
            if return_all:
                return self.kitti_cache[data_path]
            else:
                return self.kitti_cache[data_path][indices]

    def odometry_to_positions(self, odometry):
        if self.IS_ODOMETRY:
            T_w_cam0 = odometry.reshape(3, 4)
            T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
            return T_w_cam0

    def _get_velodyne_fn(self, drive, t):
        if self.IS_ODOMETRY:
            fname = self.root + '/sequences/%02d/velodyne/%06d.bin' % (drive, t)
        return fname

    def get_position_transform(self, pos0, pos1, invert=False):
        T0 = self.pos_transform(pos0)
        T1 = self.pos_transform(pos1)
        return (np.dot(T1, np.linalg.inv(T0)).T if not invert else np.dot(
            np.linalg.inv(T1), T0).T)


class RedWood(data.Dataset):
    __type__ = 'descriptor'

    def __init__(self,
                 root,
                 scene,
                 downsample=0.03,
                 config=None,
                 last_scene=False,
                 ):
        self.root = root
        self.downsample = downsample
        self.config = config

        # contrainer
        self.points = []
        self.ids_list = []
        self.num_test = 0
        self.scene = scene
        self.pcd_list = []
        scene_path = os.path.join(root, scene + '/fragments')
        pcd_list = os.listdir(scene_path)

        for pcd_path in pcd_list:
            if not pcd_path.endswith('.ply'):
                continue
            full_path = os.path.join(scene_path, pcd_path)
            orig_pcd = o3d.io.read_point_cloud(full_path)
            pcd = orig_pcd.voxel_down_sample(voxel_size=self.downsample)
            points = np.array(pcd.points)
            self.points += [points]
            self.pcd_list.append(pcd_path)

        print()

    def __getitem__(self, index):
        pts = self.points[index].astype(np.float32)
        feat = np.ones_like(pts[:, :1]).astype(np.float32)
        return pts, pts, feat, feat, np.array([]), np.array([])

    def __len__(self):
        return len(self.points)



if __name__ == "__main__":
    dset = ThreeDMatchDataset(root='/data/3DMatch/', split='train', num_node=64, downsample=0.05, self_augment=True)
    dset[0]
    import pdb
    pdb.set_trace()
