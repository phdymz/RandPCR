import os
import open3d as o3d
import argparse
import json
import importlib
import logging
import torch
import numpy as np
from multiprocessing import Process, Manager
from functools import partial
from easydict import EasyDict as edict
from utils.pointcloud import make_point_cloud
from models.architectures import KPFCNN
from utils.timer import Timer, AverageMeter
from datasets.ThreeDMatch import ThreeDMatchTestset, KITTIDataset
from datasets.dataloader import get_dataloader, get_dataloader_kitti
from geometric_registration.common import get_pcd, get_keypts, get_desc, get_scores, loadlog, build_correspondence, \
    build_correspondence_single
from utils.uio import compute_transform_error, read_info_file
import os.path as osp
from tqdm import tqdm


def to_tensor(array):
    """
    Convert array to tensor
    """
    if(not isinstance(array,torch.Tensor)):
        return torch.from_numpy(array).float()
    else:
        return array

def mutual_selection(score_mat):
    """
    Return a {0,1} matrix, the element is 1 if and only if it's maximum along both row and column

    Args: np.array()
        score_mat:  [B,N,N]
    Return:
        mutuals:    [B,N,N]
    """
    score_mat = to_array(score_mat)
    if (score_mat.ndim == 2):
        score_mat = score_mat[None, :, :]

    mutuals = np.zeros_like(score_mat)
    for i in range(score_mat.shape[0]):  # loop through the batch
        c_mat = score_mat[i]
        flag_row = np.zeros_like(c_mat)
        flag_column = np.zeros_like(c_mat)

        max_along_row = np.argmax(c_mat, 1)[:, None]
        max_along_column = np.argmax(c_mat, 0)[None, :]
        np.put_along_axis(flag_row, max_along_row, 1, 1)
        np.put_along_axis(flag_column, max_along_column, 1, 0)
        mutuals[i] = (flag_row.astype(np.bool_)) & (flag_column.astype(np.bool_))
    return mutuals.astype(np.bool_)

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

def get_angle_deviation(R_pred,R_gt):
    """
    Calculate the angle deviation between two rotaion matrice
    The rotation error is between [0,180]
    Input:
        R_pred: [B,3,3]
        R_gt  : [B,3,3]
    Return:
        degs:   [B]
    """
    R=np.matmul(R_pred,R_gt.transpose(0,2,1))
    tr=np.trace(R,0,1,2)
    rads=np.arccos(np.clip((tr-1)/2,-1,1))  # clip to valid range
    degs=rads/np.pi*180

    return degs

def ransac_pose_estimation(src_pcd, tgt_pcd, src_feat, tgt_feat, mutual=False, distance_threshold=0.05, ransac_n=3):
    """
    RANSAC pose estimation with two checkers
    We follow D3Feat to set ransac_n = 3 for 3DMatch and ransac_n = 4 for KITTI.
    For 3DMatch dataset, we observe significant improvement after changing ransac_n from 4 to 3.
    """
    if (mutual):
        if (torch.cuda.device_count() >= 1):
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        src_feat, tgt_feat = to_tensor(src_feat), to_tensor(tgt_feat)
        scores = torch.matmul(src_feat.cuda(), tgt_feat.transpose(0, 1).cuda()).cpu()
        selection = mutual_selection(scores[None, :, :])[0]
        row_sel, col_sel = np.where(selection)
        corrs = o3d.utility.Vector2iVector(np.array([row_sel, col_sel]).T)
        src_pcd = to_o3d_pcd(src_pcd)
        tgt_pcd = to_o3d_pcd(tgt_pcd)
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            source=src_pcd, target=tgt_pcd, corres=corrs,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000))
    else:
        src_pcd = to_o3d_pcd(src_pcd)
        tgt_pcd = to_o3d_pcd(tgt_pcd)
        src_feats = to_o3d_feats(src_feat)
        tgt_feats = to_o3d_feats(tgt_feat)

        # result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        #     src_pcd, tgt_pcd, src_feats, tgt_feats, distance_threshold,
        #     o3d.pipelines.registration.TransformationEstimationPointToPoint(False), ransac_n,
        #     [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        #      o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        #     o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000))
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source=src_pcd,
            target=tgt_pcd,
            source_feature=src_feats,
            target_feature=tgt_feats,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000),
            mutual_filter=False
        )

    return result_ransac.transformation

def to_o3d_pcd(xyz):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(to_array(xyz))
    return pcd

def to_o3d_feats(embedding):
    """
    Convert tensor/array to open3d features
    embedding:  [N, 3]
    """
    feats = o3d.pipelines.registration.Feature()
    feats.data = to_array(embedding).T
    return feats




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chosen_snapshot', default='D3Feat02011008', type=str, help='snapshot dir')
    parser.add_argument('--inlier_ratio_threshold', default=0.05, type=float)
    parser.add_argument('--distance_threshold', default=0.10, type=float)
    parser.add_argument('--random_points', default=False, action='store_true')
    parser.add_argument('--num_points', default=5000, type=int)
    parser.add_argument('--generate_features', default=True, action='store_true')
    parser.add_argument('--num_operator', default=3, type=int)
    parser.add_argument('--multi', default=True, type=bool)


    args = parser.parse_args()
    if args.random_points:
        log_filename = f'geometric_registration/{args.chosen_snapshot}-rand-{args.num_points}.log'
    else:
        log_filename = f'geometric_registration/{args.chosen_snapshot}-pred-{args.num_points}.log'
    logging.basicConfig(level=logging.INFO,
                        filename=log_filename,
                        filemode='w',
                        format="")

    config_path = f'./configs/config.json'
    config = json.load(open(config_path, 'r'))
    config = edict(config)

    config.root = '/media/ymz/2b933929-0294-4162-9385-4fe3eec72189/distillation/Kitti/Kitti'
    config.first_subsampling_dl = 0.3
    config.conv_radius = 4.25
    # config.use_batch_norm = True
    # config.first_features_dim = 64


    # create model
    config.architecture = [
        'simple',
        'resnetb',
    ]
    for i in range(config.num_layers - 1):
        config.architecture.append('resnetb_strided')
        config.architecture.append('resnetb')
        config.architecture.append('resnetb')
    for i in range(config.num_layers - 2):
        config.architecture.append('nearest_upsample')
        config.architecture.append('unary')
    config.architecture.append('nearest_upsample')
    config.architecture.append('last_unary')

    models = []
    print('Load {} operators'.format(args.num_operator))
    for i in range(args.num_operator):
        torch.manual_seed(i * 100)
        models.append(KPFCNN(config))
        models[-1].eval().cuda()

    dset = KITTIDataset(config, 'test', data_augmentation=False)
    dloader, _ = get_dataloader_kitti(dataset=dset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=config.num_workers,
                                )

    tsfm_est = []
    num_iter = int(len(dset) // config.batch_size)
    c_loader_iter = dloader.__iter__()

    rot_gt, trans_gt = [], []
    with torch.no_grad():
        for _ in tqdm(range(num_iter)):  # loop through this epoch
            inputs = c_loader_iter.next()
            ###############################################
            # forward pass
            for k, v in inputs.items():
                if type(v) == list:
                    inputs[k] = [item.cuda() for item in v]
                else:
                    inputs[k] = v.cuda()

            len_src = inputs['stack_lengths'][0][0]
            c_rot, c_trans = inputs['rot'], inputs['trans']
            rot_gt.append(c_rot.cpu().numpy())
            trans_gt.append(c_trans.cpu().numpy())
            src_pcd, tgt_pcd = inputs['points'][0][:len_src], inputs['points'][0][len_src:]
            n_points = args.num_points

            correspondences = []

            for i in range(args.num_operator):
                model = models[i]
                features, scores = model(inputs)  # [N1, C1], [N2, C2]
                scores = scores.detach().cpu()

                src_feats, tgt_feats = features[:len_src], features[len_src:]
                src_saliency, tgt_saliency = scores[:len_src], scores[len_src:]

                src_scores =  src_saliency
                tgt_scores =  tgt_saliency

                if (src_pcd.size(0) > n_points):
                    idx = torch.argsort(src_scores.squeeze())[-args.num_points:]
                    source_keypts, source_desc = src_pcd[idx], src_feats[idx]
                    source_keypts, source_desc = source_keypts.cpu().numpy(), source_desc.cpu().numpy()

                    # src_feats = src_feats[idx]

                if (tgt_pcd.size(0) > n_points):
                    idx = torch.argsort(tgt_scores.squeeze())[-args.num_points:]
                    target_keypts, target_desc = tgt_pcd[idx], tgt_feats[idx]
                    target_keypts, target_desc = target_keypts.cpu().numpy(), target_desc.cpu().numpy()

                    # tgt_feats = tgt_feats[idx]

                # scores = torch.matmul(src_feats, tgt_feats.transpose(0, 1)).cpu()
                # selection = mutual_selection(scores[None, :, :])[0]
                # row_sel, col_sel = np.where(selection)
                # corr = np.array([row_sel, col_sel]).T
                # src_kpts = source_keypts[corr[:, 0]]
                # tgt_kpts = target_keypts[corr[:, 1]]

                corr = build_correspondence_single(source_desc, target_desc)
                src_kpts = source_keypts[corr[:, 0]]
                tgt_kpts = target_keypts[corr[:, 1]]

                correspondences.append(np.concatenate([src_kpts, tgt_kpts], axis=-1))
            ########################################
            # run ransac
            if not args.multi:
                src_ref = correspondences[0][:, :3]
                tgt_ref = correspondences[0][:, 3:]

                src_set = src_ref[None,]
                tgt_set = tgt_ref[None,]

                for i in range(1, args.num_operator):
                    src_associate_i = correspondences[i][:, :3]
                    tgt_associate_i = correspondences[i][:, 3:]

                    dis_i = (src_ref ** 2).sum(-1).reshape(-1, 1) + (src_associate_i ** 2).sum(-1).reshape(1,
                                                                                                           -1) - 2 * np.matmul(
                        src_ref, src_associate_i.T)
                    idx = dis_i.argmin(axis=-1)

                    src_set = np.concatenate([src_set, src_associate_i[idx][None]], axis=0)
                    tgt_set = np.concatenate([tgt_set, tgt_associate_i[idx][None]], axis=0)

                src_dist = ((src_ref[None] - src_set[1:]) ** 2).sum(-1).T
                tgt_dist = ((tgt_ref[None] - tgt_set[1:]) ** 2).sum(-1).T

                compatibility = np.exp(-abs(src_dist - tgt_dist) / args.distance_threshold).mean(-1)
                mask = compatibility > np.exp(-1.0)
                if mask.sum() < 4:
                    frag1 = src_ref
                    frag2 = tgt_ref
                else:
                    frag1 = src_ref[mask]
                    frag2 = tgt_ref[mask]
            else:
                # use multi
                frag1 = []
                frag2 = []

                for j in range(len(correspondences)):
                    index = []
                    for i in range(len(correspondences)):
                        if i != j:
                            index.append(i)
                    src_ref = correspondences[j][:, :3]
                    tgt_ref = correspondences[j][:, 3:]

                    src_set = src_ref[None,]
                    tgt_set = tgt_ref[None,]

                    for i in index:
                        src_associate_i = correspondences[i][:, :3]
                        tgt_associate_i = correspondences[i][:, 3:]

                        dis_i = (src_ref ** 2).sum(-1).reshape(-1, 1) + (src_associate_i ** 2).sum(-1).reshape(1,
                                                                                                               -1) - 2 * np.matmul(
                            src_ref, src_associate_i.T)
                        idx = dis_i.argmin(axis=-1)

                        src_set = np.concatenate([src_set, src_associate_i[idx][None]], axis=0)
                        tgt_set = np.concatenate([tgt_set, tgt_associate_i[idx][None]], axis=0)

                    src_dist = ((src_ref[None] - src_set[1:]) ** 2).sum(-1).T
                    tgt_dist = ((tgt_ref[None] - tgt_set[1:]) ** 2).sum(-1).T

                    compatibility = np.exp(-abs(src_dist - tgt_dist) / args.distance_threshold).mean(-1)
                    mask = compatibility > np.exp(-1.0)
                    if mask.sum() < 4:
                        pass
                    else:
                        frag1.append(src_ref[mask])
                        frag2.append(tgt_ref[mask])

            if frag1 == []:
                frag1 = src_ref
                frag2 = tgt_ref
            else:
                frag1 = np.vstack(frag1)
                frag2 = np.vstack(frag2)

            frag1_pc = o3d.geometry.PointCloud()
            frag1_pc.points = o3d.utility.Vector3dVector(frag1)

            frag2_pc = o3d.geometry.PointCloud()
            frag2_pc.points = o3d.utility.Vector3dVector(frag2)

            corrs = o3d.utility.Vector2iVector(
                np.concatenate([np.arange(0, len(frag1))[:, None], np.arange(0, len(frag1))[:, None]], axis=-1))
            distance_threshold = 0.3

            result_ransac = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                source=frag1_pc, target=frag2_pc, corres=corrs,
                max_correspondence_distance=distance_threshold,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=4,
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000))

            tsfm_est.append(result_ransac.transformation)

    tsfm_est = np.array(tsfm_est)
    rot_est = tsfm_est[:, :3, :3]
    trans_est = tsfm_est[:, :3, 3]
    rot_gt = np.array(rot_gt)
    trans_gt = np.array(trans_gt)[:, :, 0]

    rot_threshold = 5
    trans_threshold = 2

    r_deviation = get_angle_deviation(rot_est, rot_gt)
    translation_errors = np.linalg.norm(trans_est - trans_gt, axis=-1)

    flag_1 = r_deviation < rot_threshold
    flag_2 = translation_errors < trans_threshold
    correct = (flag_1 & flag_2).sum()
    precision = correct / rot_gt.shape[0]

    message = f'\n Registration recall: {100*precision:.2f}\n'

    r_deviation = r_deviation[flag_1]
    translation_errors = translation_errors[flag_2]

    errors = dict()
    errors['rot_mean'] = round(np.mean(r_deviation), 2)
    errors['rot_std'] = round(np.std(r_deviation), 2)

    # errors['rot_median'] = round(np.median(r_deviation), 3)
    errors['trans_rmse'] = round(100 * np.mean(translation_errors), 2)
    errors['trans_std'] = round(100 * np.std(translation_errors), 2)
    # errors['trans_rmedse'] = round(np.median(translation_errors), 3)

    message += str(errors)
    print(message)
