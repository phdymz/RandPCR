import torch
import numpy as np
from nibabel import quaternions as nq




def get_rotation_translation_from_transform(transform: np.ndarray):
    r"""Get rotation matrix and translation vector from rigid transform matrix.
    Args:
        transform (array): (4, 4)
    Returns:
        rotation (array): (3, 3)
        translation (array): (3,)
    """
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return rotation, translation





def compute_transform_error(transform, covariance, estimated_transform):
    relative_transform = np.matmul(np.linalg.inv(transform), estimated_transform)
    R, t = get_rotation_translation_from_transform(relative_transform) # tor trans
    q = nq.mat2quat(R)
    er = np.concatenate([t, q[1:]], axis=0)
    p = er.reshape(1, 6) @ covariance @ er.reshape(6, 1) / covariance[0, 0]
    return p.item()



def read_info_file(file_name):
    with open(file_name) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    test_pairs = []
    num_pairs = len(lines) // 7
    for i in range(num_pairs):
        line_id = i * 7
        split_line = lines[line_id].split()
        test_pair = [int(split_line[0]), int(split_line[1])]
        num_fragments = int(split_line[2])
        info = []
        for j in range(1, 7):
            info.append(lines[line_id + j].split())
        info = np.array(info, dtype=np.float32)
        test_pairs.append(dict(test_pair=test_pair, num_fragments=num_fragments, covariance=info))
    return test_pairs


