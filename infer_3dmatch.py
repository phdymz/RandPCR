import os
import time

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
from datasets.ThreeDMatch import ThreeDMatchTestset
from datasets.dataloader import get_dataloader
from geometric_registration.common import get_pcd, get_keypts, get_desc, get_scores, loadlog, build_correspondence, build_correspondence_single
from utils.uio import compute_transform_error, read_info_file
import os.path as osp




def register_one_scene(inlier_ratio_threshold, distance_threshold, save_path, args, scene):
    gt_matches = 0
    pred_matches = 0
    accepted = 0
    keyptspath = f"{save_path}/keypoints/{scene}"
    descpath = f"{save_path}/descriptors/{scene}"
    scorepath = f"{save_path}/scores/{scene}"
    gtpath = f'geometric_registration/gt_result/{scene}-evaluation/'
    gtLog = loadlog(gtpath)
    infos = read_info_file(osp.join(f'../data/fragments/{scene}-evaluation', 'gt.info'))
    covariances = {}
    for item in infos:
        pair1, pair2 = item['test_pair']
        covariances[f'{pair1}_{pair2}'] = item['covariance']

    times = []
    times_corr = []
    memorys = []

    inlier_num_meter, inlier_ratio_meter = AverageMeter(), AverageMeter()
    pcdpath = f"../data/fragments/{scene}/"
    num_frag = len([filename for filename in os.listdir(pcdpath) if filename.endswith('ply')])
    for id1 in range(num_frag):
        for id2 in range(id1 + 1, num_frag):
            cloud_bin_s = f'cloud_bin_{id1}'
            cloud_bin_t = f'cloud_bin_{id2}'
            key = f"{id1}_{id2}"
            if key not in gtLog.keys():
                # skip the pairs that have less than 30% overlap.
                num_inliers = 0
                inlier_ratio = 0
                gt_flag = 0
            else:
                now = time.time()

                correspondences = []

                for i in range(args.num_operator):
                    cloud_bin_s_i = cloud_bin_s + f'_{i}'
                    cloud_bin_t_i = cloud_bin_t + f'_{i}'

                    source_keypts = get_keypts(keyptspath, cloud_bin_s_i)
                    target_keypts = get_keypts(keyptspath, cloud_bin_t_i)
                    source_desc = get_desc(descpath, cloud_bin_s_i, 'D3Feat')
                    target_desc = get_desc(descpath, cloud_bin_t_i, 'D3Feat')
                    source_score = get_scores(scorepath, cloud_bin_s_i, 'D3Feat').squeeze()
                    target_score = get_scores(scorepath, cloud_bin_t_i, 'D3Feat').squeeze()
                    source_desc = np.nan_to_num(source_desc)
                    target_desc = np.nan_to_num(target_desc)

                    # randomly select 5000 keypts
                    if args.random_points:
                        source_indices = np.random.choice(range(source_keypts.shape[0]), args.num_points)
                        target_indices = np.random.choice(range(target_keypts.shape[0]), args.num_points)
                    else:
                        source_indices = np.argsort(source_score)[-args.num_points:]
                        target_indices = np.argsort(target_score)[-args.num_points:]
                    source_keypts = source_keypts[source_indices, :]
                    source_desc = source_desc[source_indices, :]
                    target_keypts = target_keypts[target_indices, :]
                    target_desc = target_desc[target_indices, :]

                    corr = build_correspondence(source_desc, target_desc)
                    # corr = build_correspondence_single(source_desc, target_desc)

                    src_kpts = source_keypts[corr[:,0]]
                    tgt_kpts = target_keypts[corr[:,1]]

                    correspondences.append(np.concatenate([src_kpts, tgt_kpts], axis = -1))

                times_corr.append(time.time() - now)

                #sample knn
                if not args.multi:
                    src_ref = correspondences[0][:,:3]
                    tgt_ref = correspondences[0][:,3:]

                    src_set = src_ref[None,]
                    tgt_set = tgt_ref[None,]

                    for i in range(1, args.num_operator):
                        src_associate_i = correspondences[i][:,:3]
                        tgt_associate_i = correspondences[i][:,3:]

                        dis_i = (src_ref**2).sum(-1).reshape(-1,1) + (src_associate_i**2).sum(-1).reshape(1,-1) - 2 * np.matmul(src_ref, src_associate_i.T)
                        idx = dis_i.argmin(axis = -1)

                        src_set = np.concatenate([src_set, src_associate_i[idx][None]], axis = 0)
                        tgt_set = np.concatenate([tgt_set, tgt_associate_i[idx][None]], axis = 0)


                    src_dist = ((src_ref[None] - src_set[1:])**2).sum(-1).T
                    tgt_dist = ((tgt_ref[None] - tgt_set[1:])**2).sum(-1).T

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
                            src_associate_i = correspondences[i][:,:3]
                            tgt_associate_i = correspondences[i][:,3:]

                            dis_i = (src_ref**2).sum(-1).reshape(-1,1) + (src_associate_i**2).sum(-1).reshape(1,-1) - 2 * np.matmul(src_ref, src_associate_i.T)
                            idx = dis_i.argmin(axis = -1)


                            src_set = np.concatenate([src_set, src_associate_i[idx][None]], axis = 0)
                            tgt_set = np.concatenate([tgt_set, tgt_associate_i[idx][None]], axis = 0)

                        src_dist = ((src_ref[None] - src_set[1:])**2).sum(-1).T
                        tgt_dist = ((tgt_ref[None] - tgt_set[1:])**2).sum(-1).T

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

                times.append(time.time() - now)


                gt_trans = gtLog[key]
                frag1_pc = o3d.geometry.PointCloud()
                frag1_pc.points = o3d.utility.Vector3dVector(frag1)

                frag2_pc = o3d.geometry.PointCloud()
                frag2_pc.points = o3d.utility.Vector3dVector(frag2)

                corrs = o3d.utility.Vector2iVector(np.concatenate([np.arange(0,len(frag1))[:,None], np.arange(0,len(frag1))[:,None]], axis = -1))



                #test RANSAC
                # result_ransac = o3d.registration.registration_ransac_based_on_correspondence(
                #     source=frag1_pc, target=frag2_pc, corres=corrs,
                #     max_correspondence_distance=distance_threshold,
                #     estimation_method=o3d.registration.TransformationEstimationPointToPoint(False),
                #     ransac_n=4,
                #     criteria=o3d.registration.RANSACConvergenceCriteria(50000, 1000))

                result_ransac = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                    source=frag1_pc, target=frag2_pc, corres=corrs,
                    max_correspondence_distance=distance_threshold,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                    ransac_n=4,
                    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000))

                error = compute_transform_error(np.linalg.inv(gt_trans), covariances[key], result_ransac.transformation)
                accepted += (error < (0.2 ** 2))

                frag2_pc.transform(gt_trans)
                frag2 = np.asarray(frag2_pc.points)
                distance = np.sqrt(np.sum(np.power(frag1 - frag2, 2), axis=1))
                num_inliers = np.sum(distance < distance_threshold)
                inlier_ratio = num_inliers / len(distance)
                if inlier_ratio > inlier_ratio_threshold:
                    pred_matches += 1
                gt_matches += 1
                inlier_num_meter.update(num_inliers)
                inlier_ratio_meter.update(inlier_ratio)


    recall = pred_matches * 100.0 / gt_matches
    rr = accepted * 100.0 / gt_matches
    logging.info(
        f"{scene}: Recall={recall:.2f}%, inlier ratio={inlier_ratio_meter.avg * 100:.2f}%, inlier num={inlier_num_meter.avg:.2f}, "
        f"registration recall={rr:.2f}")
    return recall, inlier_num_meter.avg, inlier_ratio_meter.avg, rr, accepted, np.mean(times_corr), np.mean(times)


def generate_features(models, dloader, args):

    times = []
    memorys = []

    dataloader_iter = dloader.__iter__()

    descriptor_path = f'{save_path}/descriptors'
    keypoint_path = f'{save_path}/keypoints'
    score_path = f'{save_path}/scores'
    if not os.path.exists(descriptor_path):
        os.mkdir(descriptor_path)
    if not os.path.exists(keypoint_path):
        os.mkdir(keypoint_path)
    if not os.path.exists(score_path):
        os.mkdir(score_path)

    # generate descriptors
    recall_list = []
    for scene in dset.scene_list:
        descriptor_path_scene = os.path.join(descriptor_path, scene)
        keypoint_path_scene = os.path.join(keypoint_path, scene)
        score_path_scene = os.path.join(score_path, scene)
        if not os.path.exists(descriptor_path_scene):
            os.mkdir(descriptor_path_scene)
        if not os.path.exists(keypoint_path_scene):
            os.mkdir(keypoint_path_scene)
        if not os.path.exists(score_path_scene):
            os.mkdir(score_path_scene)
        pcdpath = f"../data/fragments/{scene}/"
        num_frag = len([filename for filename in os.listdir(pcdpath) if filename.endswith('ply')])
        # generate descriptors for each fragment
        for ids in range(num_frag):
            inputs = dataloader_iter.next()
            for k, v in inputs.items():  # load inputs to device.
                if type(v) == list:
                    inputs[k] = [item.cuda() for item in v]
                else:
                    inputs[k] = v.cuda()

            now = time.time()
            torch.cuda.synchronize(device=inputs['points'][0].device)
            torch.cuda.reset_max_memory_allocated(device=inputs['points'][0].device)

            for i in range(args.num_operator):
                model = models[i]
                features, scores = model(inputs)
                pcd_size = inputs['stack_lengths'][0][0]
                pts = inputs['points'][0][:int(pcd_size)]
                features, scores = features[:int(pcd_size)], scores[:int(pcd_size)]
                # scores = torch.ones_like(features[:, 0:1])
                np.save(f'{descriptor_path_scene}/cloud_bin_{ids}_{i}.D3Feat',
                        features.detach().cpu().numpy().astype(np.float32))
                np.save(f'{keypoint_path_scene}/cloud_bin_{ids}_{i}', pts.detach().cpu().numpy().astype(np.float32))
                np.save(f'{score_path_scene}/cloud_bin_{ids}_{i}', scores.detach().cpu().numpy().astype(np.float32))

            torch.cuda.synchronize(device=inputs['points'][0].device)
            memory_usage = torch.cuda.max_memory_allocated(device=inputs['points'][0].device)
            times.append(time.time() - now)
            memorys.append(memory_usage/1024/1024)

            print(f"Generate cloud_bin_{ids} for {scene}")

    print('average time on feature extraction', 1000*np.mean(times))
    print(np.mean(memorys))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chosen_snapshot', default='D3Feat02011008', type=str, help='snapshot dir')
    parser.add_argument('--inlier_ratio_threshold', default=0.05, type=float)
    parser.add_argument('--distance_threshold', default=0.10, type=float)
    parser.add_argument('--random_points', default=False, action='store_true')
    parser.add_argument('--num_points', default=1000, type=int)
    parser.add_argument('--generate_features', default=False, action='store_true')
    parser.add_argument('--num_operator', default=6, type=int)
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


    save_path = f'./geometric_registration/{args.chosen_snapshot}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if args.generate_features:
        dset = ThreeDMatchTestset(root=config.root,
                                  downsample=config.downsample,
                                  config=config,
                                  last_scene=False,
                                  )
        dloader, _ = get_dataloader(dataset=dset,
                                    batch_size=config.batch_size,
                                    shuffle=False,
                                    num_workers=config.num_workers,
                                    )
        generate_features(models, dloader, args)

    # register each pair of fragments in scenes using multiprocessing.
    scene_list = [
        '7-scenes-redkitchen',
        'sun3d-home_at-home_at_scan1_2013_jan_1',
        'sun3d-home_md-home_md_scan9_2012_sep_30',
        'sun3d-hotel_uc-scan3',
        'sun3d-hotel_umd-maryland_hotel1',
        'sun3d-hotel_umd-maryland_hotel3',
        'sun3d-mit_76_studyroom-76-1studyroom2',
        'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
    ]
    # return_dict = Manager().dict()
    # register_one_scene(args.inlier_ratio_threshold, args.distance_threshold, save_path, return_dict, scene_list[0])
    # jobs = []
    recalls = []
    inlier_nums = []
    inlier_ratios = []
    registration_recalls = []
    accepts = 0

    for scene in scene_list:
        recall, inlier_num_meter, inlier_ratio_meter, rr, accepted, time_cost_corr, time_cost_total = register_one_scene(args.inlier_ratio_threshold,
                                                                          args.distance_threshold, save_path, args, scene)
        recalls.append(recall)
        inlier_nums.append(inlier_num_meter)
        inlier_ratios.append(inlier_ratio_meter)
        registration_recalls.append(rr)
        # accepts += accepted


    print(f"All 8 scene, average recall: {np.mean(recalls):.2f}% +- {np.std(recalls):.2f}%")
    # print(f"All 8 scene, average num inliers: {np.mean(inlier_nums):.2f}")
    print(f"All 8 scene, average num inliers ratio: {np.mean(inlier_ratios) * 100:.2f}%")
    print(f"All 8 scene, average registration recall: {np.mean(registration_recalls):.2f}%")
    # print(f"All 8 scene, mean registration recall: {accepts/1623*100}")
    print(f"All 8 scene, average time cost correspondence: {1000 * np.mean(time_cost_corr):.2f}ms")
    print(f"All 8 scene, average time cost filter: {1000 * (np.mean(time_cost_total) - np.mean(time_cost_corr)):.2f}ms")
