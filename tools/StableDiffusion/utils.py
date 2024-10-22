import os
import cv2
import math
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from copy import deepcopy
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

def gen_config(cfg_path):
    return OmegaConf.load(cfg_path)

def get_focal_from_fov(new_fov, H, W):
    # NOTE: top-left pixel should be (0,0)
    if W >= H:
        f = (W / 2.0) / np.tan(np.deg2rad(new_fov / 2.0))
    else:
        f = (H / 2.0) / np.tan(np.deg2rad(new_fov / 2.0))
    return f

def get_intrins_from_fov(new_fov, H, W, device):
    # NOTE: top-left pixel should be (0,0)
    f = get_focal_from_fov(new_fov,H,W)

    new_cu = (W / 2.0) - 0.5
    new_cv = (H / 2.0) - 0.5

    new_intrins = torch.tensor([
        [f,         0,     new_cu  ],
        [0,         f,     new_cv  ],
        [0,         0,     1       ]
    ], dtype=torch.float32, device=device)

    return new_intrins

def dpt2xyz(dpt,intrinsic):
    # get grid
    height, width = dpt.shape[0:2]
    grid_u = np.arange(width)[None,:].repeat(height,axis=0)
    grid_v = np.arange(height)[:,None].repeat(width,axis=1)
    grid = np.concatenate([grid_u[:,:,None],grid_v[:,:,None],np.ones_like(grid_v)[:,:,None]],axis=-1)
    uvz = grid * dpt[:,:,None]
    # inv intrinsic
    inv_intrinsic = np.linalg.inv(intrinsic)
    xyz = np.einsum(f'ab,hwb->hwa',inv_intrinsic,uvz)
    return xyz

def dpt2xyz_torch(dpt,intrinsic):
    # get grid
    height, width = dpt.shape[0:2]
    grid_u = torch.arange(width)[None,:].repeat(height,1)
    grid_v = torch.arange(height)[:,None].repeat(1,width)
    grid = torch.concatenate([grid_u[:,:,None],grid_v[:,:,None],torch.ones_like(grid_v)[:,:,None]],axis=-1).to(dpt)
    uvz = grid * dpt[:,:,None]
    # inv intrinsic
    inv_intrinsic = torch.linalg.inv(intrinsic)
    xyz = torch.einsum(f'ab,hwb->hwa',inv_intrinsic,uvz)
    return xyz

def visual_pcd(xyz, color=None, normal = True):
    if hasattr(xyz,'ndim'):
        xyz_norm = np.mean(np.sqrt(np.sum(np.square(xyz),axis=1)))
        xyz = xyz / xyz_norm
        xyz = xyz.reshape(-1,3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
    else: pcd = xyz
    if color is not None:
        color = color.reshape(-1,3)
        pcd.colors = o3d.utility.Vector3dVector(color)
    if normal:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.2, 20))
    o3d.visualization.draw_geometries([pcd])

def visual_pcds(xyzs, normal = True):
    pcds = []
    for xyz in xyzs:
        if hasattr(xyz,'ndim'):
            # xyz_norm = np.mean(np.sqrt(np.sum(np.square(xyz),axis=1)))
            # xyz = xyz / xyz_norm
            xyz = xyz.reshape(-1,3)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.paint_uniform_color(np.random.rand(3))
        else: pcd = xyz
        if normal:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.2, 20))
        pcds.append(pcd)
    o3d.visualization.draw_geometries(pcds)
    
def save_pic(input_pic:np.array,save_fn,normalize=True):
    # avoid replace
    pic = deepcopy(input_pic).astype(np.float32)
    if normalize:
        vmin = np.percentile(pic, 2)
        vmax = np.percentile(pic, 98)
        pic = (pic - vmin) / (vmax - vmin)
    pic = (pic * 255.0).clip(0, 255)
    if save_fn is not None:
        pic_save = Image.fromarray(pic.astype(np.uint8))
        pic_save.save(save_fn)
    return pic

def dpt_normalize(dpt, sky_mask = None):
    if sky_mask is not None:
        pic = dpt[~sky_mask]
    else:
        pic = dpt
    vmin = np.percentile(pic, 2)
    vmax = np.percentile(pic, 98)
    dpt = (dpt - vmin) / (vmax - vmin)
    if sky_mask is not None:
        dpt[sky_mask] = 1.
    return dpt

def points_to_hpoints(points):
    ones = np.ones_like(points)[...,-1:]
    return np.concatenate([points,ones],-1)

def hpoints_to_points(hpoints):
    return hpoints[...,:-1]/hpoints[...,-1:]

def transform_points(pts,transform):
    h,w=transform.shape
    if h==3 and w==3:
        return pts @ transform.T
    if h==3 and w==4:
        return pts @ transform[:,:3].T + transform[:,3:].T
    elif h==4 and w==4:
        return hpoints_to_points(points_to_hpoints(pts) @ transform.T)
    else: raise NotImplementedError

def get_nml_from_quant(quant):
    '''
    input N*4
    outut N*3
    follow https://arxiv.org/pdf/2404.17774
    '''
    w=quant[:,0]
    x=quant[:,1]
    y=quant[:,2]
    z=quant[:,3]
    n0 = 2*x*z+2*y*w
    n1 = 2*y*z-2*x*w
    n2 = 1-2*x*x-2*y*y
    nml = torch.cat((n0[:,None],n1[:,None],n2[:,None]),dim=1)
    return nml

def quaternion_from_matrix(M):
    m00 = M[..., 0, 0]
    m01 = M[..., 0, 1]
    m02 = M[..., 0, 2]
    m10 = M[..., 1, 0]
    m11 = M[..., 1, 1]
    m12 = M[..., 1, 2]
    m20 = M[..., 2, 0]
    m21 = M[..., 2, 1]
    m22 = M[..., 2, 2]   
    K = torch.zeros((len(M),4,4)).to(M)
    K[:,0,0] = m00 - m11 - m22
    K[:,1,0] = m01 + m10
    K[:,1,1] = m11 - m00 - m22
    K[:,2,0] = m02 + m20
    K[:,2,1] = m12 + m21
    K[:,2,2] = m22 - m00 - m11
    K[:,3,0] = m21 - m12
    K[:,3,1] = m02 - m20
    K[:,3,2] = m10 - m01
    K[:,3,3] = m00 + m11 + m22
    K = K/3
    # quaternion is eigenvector of K that corresponds to largest eigenvalue
    w, V = torch.linalg.eigh(K)
    q = V[torch.arange(len(V)),:,torch.argmax(w,dim=1)]
    q = q[:,[3, 0, 1, 2]]
    for i in range(len(q)):
        if q[i,0]<0.:
            q[i] = -q[i]
    return q

def numpy_quaternion_from_matrix(M):
    H,W = M.shape[0:2]
    M = M.reshape(-1,3,3)
    m00 = M[..., 0, 0]
    m01 = M[..., 0, 1]
    m02 = M[..., 0, 2]
    m10 = M[..., 1, 0]
    m11 = M[..., 1, 1]
    m12 = M[..., 1, 2]
    m20 = M[..., 2, 0]
    m21 = M[..., 2, 1]
    m22 = M[..., 2, 2]   
    K = np.zeros((len(M),4,4))
    K[...,0,0] = m00 - m11 - m22
    K[...,1,0] = m01 + m10
    K[...,1,1] = m11 - m00 - m22
    K[...,2,0] = m02 + m20
    K[...,2,1] = m12 + m21
    K[...,2,2] = m22 - m00 - m11
    K[...,3,0] = m21 - m12
    K[...,3,1] = m02 - m20
    K[...,3,2] = m10 - m01
    K[...,3,3] = m00 + m11 + m22
    K = K/3
    # quaternion is eigenvector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    q = V[np.arange(len(V)),:,np.argmax(w,axis=1)]
    q = q[...,[3, 0, 1, 2]]
    for i in range(len(q)):
        if q[i,0]<0.:
            q[i] = -q[i]
    q = q.reshape(H,W,4)
    return q

def numpy_normalize(input):
    input = input / (np.sqrt(np.sum(np.square(input),axis=-1,keepdims=True))+1e-5)
    return input

import torch.nn.functional as F
def nei_delta(input,pad=2):
    if not type(input) is torch.Tensor:
        input = torch.from_numpy(input.astype(np.float32))
    if len(input.shape) < 3:
        input = input[:,:,None]
    h,w,c = input.shape
    # reshape
    input = input.permute(2,0,1)[None]
    input = F.pad(input, pad=(pad,pad,pad,pad), mode='replicate')
    kernel = 2*pad + 1
    input = F.unfold(input,[kernel,kernel],padding=0)
    input = input.reshape(c,-1,h,w).permute(2,3,0,1).squeeze() # hw(3)*25
    return torch.amax(input,dim=-1),torch.amin(input,dim=-1),input

def inpaint_mask(render_dpt,render_rgb):
    
    # edge filter delta thres
    valid_dpt = render_dpt[render_dpt>1e-3]
    valid_dpt = torch.sort(valid_dpt).values
    max = valid_dpt[int(.85*len(valid_dpt))]
    min = valid_dpt[int(.15*len(valid_dpt))]
    ths = (max-min) * 0.2
    # nei check
    nei_max, nei_min, _ = nei_delta(render_dpt,pad=1)
    edge_mask = (nei_max - nei_min) > ths
    # render hole
    hole_mask = render_dpt < 1e-3
    # whole mask -- original noise and sparse
    mask = edge_mask | hole_mask
    mask = mask.cpu().float().numpy()
    
    # modify rgb sightly for small holes : blur and sharpen
    render_rgb       = render_rgb.detach().cpu().numpy()
    render_rgb       = (render_rgb*255).astype(np.uint8)
    render_rgb_blur  = cv2.medianBlur(render_rgb,5)
    render_rgb[mask>.5] = render_rgb_blur[mask>.5]  # blur and replace small holes
    render_rgb       = torch.from_numpy((render_rgb/255).astype(np.float32)).to(render_dpt)
    
    # slightly clean mask
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.erode(mask,kernel,iterations=2)
    mask = cv2.dilate(mask,kernel,iterations=7)
    mask = mask > 0.5

    return mask,render_rgb

def radius_filter(xyz, neighbor_num = 3):
    H,W = xyz.shape[0:2]
    max_bound_div = (H+W)/neighbor_num
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.reshape(-1,3))
    max_bound = pcd.get_max_bound()
    ball_radius = np.linalg.norm(max_bound) / max_bound_div
    pcd_filter, idx_filter = pcd.remove_radius_outlier(neighbor_num, ball_radius)
    mask = np.zeros((H,W)).reshape(-1)
    mask[idx_filter] = 1
    edge_mask = mask.reshape(H,W) < .5
    return edge_mask

# def edge_filter(metric_dpt,times=0.2):
#     nei_max,nei_min,_ = nei_delta(metric_dpt)
#     delta = (nei_max-nei_min).numpy()
#     edge = delta > times*metric_dpt
#     kernel = np.ones((5,5),np.uint8)
#     edge = cv2.dilate(edge.astype(np.float32),kernel,iterations=1) > .5
#     return edge

def edge_filter(metric_dpt,times=0.1):
    _max = np.percentile(metric_dpt,95)
    _min = np.percentile(metric_dpt,5)
    _range = _max - _min
    nei_max,nei_min,_ = nei_delta(metric_dpt)
    delta = (nei_max-nei_min).numpy()
    edge = delta > times*_range
    return edge

def fill_mask_with_nearest(imgs, mask):
    # mask and un-mask pixel coors
    mask_coords = np.column_stack(np.where(mask > .5))
    non_mask_coords = np.column_stack(np.where(mask < .5))
    # kd-tree on un-masked pixels
    tree = cKDTree(non_mask_coords)
    # nn search of masked pixels
    _, idxs = tree.query(mask_coords)
    # replace and fill
    for i, coord in enumerate(mask_coords):
        nearest_coord = non_mask_coords[idxs[i]]
        for img in imgs:
            img[coord[0], coord[1]] = img[nearest_coord[0], nearest_coord[1]]
    return imgs

def edge_rectify(metric_dpt,rgb):
    edge = edge_filter(metric_dpt)
    metric_dpt,rgb = fill_mask_with_nearest([metric_dpt,rgb],edge)
    return metric_dpt,rgb
