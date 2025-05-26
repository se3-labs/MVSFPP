import torch
import torch.nn.functional as F


def parse_intrinsics(intrinsics):
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    return fx, fy, cx, cy


def expand_as(x, y):
    if len(x.shape) == len(y.shape):
        return x

    for i in range(len(y.shape) - len(x.shape)):
        x = x.unsqueeze(-1)

    return x


def lift(x, y, z, intrinsics, homogeneous=False):
    '''

    :param self:
    :param x: Shape (batch_size, num_points)
    :param y:
    :param z:
    :param intrinsics:
    :return:
    '''
    fx, fy, cx, cy = parse_intrinsics(intrinsics)

    x_lift = (x - expand_as(cx, x)) / expand_as(fx, x) * z
    y_lift = (y - expand_as(cy, y)) / expand_as(fy, y) * z

    if homogeneous:
        return torch.stack((x_lift, y_lift, z, torch.ones_like(z, device=intrinsics.device)), dim=-1)
    else:
        return torch.stack((x_lift, y_lift, z), dim=-1)


def world_from_xy_depth(xy, depth, cam2world, intrinsics):
    '''Translates meshgrid of xy pixel coordinates plus depth to  world coordinates.
    '''
    batch_size, ndepths = depth.size(0), depth.size(1)
    # height, width = img_shape
    # y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=depth.device),
    #                        torch.arange(0, width, dtype=torch.float32, device=depth.device)])
    # y, x = y.contiguous(), x.contiguous()
    # y, x = y.view(height * width), x.view(height * width)

    x_cam = xy[..., 0].view(batch_size, -1)  # ndepths, -1)
    y_cam = xy[..., 1].view(batch_size, -1)  # ndepths, -1)
    z_cam = depth.view(batch_size, -1)  # ndepths, -1)

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics, homogeneous=True)  # (batch_size, -1, 4)

    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)  # 1, 3, 2)

    world_coords = torch.bmm(cam2world, pixel_points_cam).permute(0, 2, 1)[:, :, :3]  # (batch_size, -1, 3)
    # world_coords = torch.matmul(cam2world.unsqueeze(1), pixel_points_cam).permute(0, 1, 3, 2)[..., :3]

    return world_coords

def inverse_4x4(m: torch.Tensor) -> torch.Tensor:
    # Get inverse avoiding inv, det operations that are not supported by either ONNX or TensorRT
    # m: (B, 4, 4)
    
    a, b, c, d = m[:, 0], m[:, 1], m[:, 2], m[:, 3]  # (B, 4)

    s0 = a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]
    s1 = a[:, 0]*b[:, 2] - a[:, 2]*b[:, 0]
    s2 = a[:, 0]*b[:, 3] - a[:, 3]*b[:, 0]
    s3 = a[:, 1]*b[:, 2] - a[:, 2]*b[:, 1]
    s4 = a[:, 1]*b[:, 3] - a[:, 3]*b[:, 1]
    s5 = a[:, 2]*b[:, 3] - a[:, 3]*b[:, 2]

    c5 = c[:, 2]*d[:, 3] - c[:, 3]*d[:, 2]
    c4 = c[:, 1]*d[:, 3] - c[:, 3]*d[:, 1]
    c3 = c[:, 1]*d[:, 2] - c[:, 2]*d[:, 1]
    c2 = c[:, 0]*d[:, 3] - c[:, 3]*d[:, 0]
    c1 = c[:, 0]*d[:, 2] - c[:, 2]*d[:, 0]
    c0 = c[:, 0]*d[:, 1] - c[:, 1]*d[:, 0]

    inv_det = 1.0 / (
        s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0
    )

    inv = torch.zeros_like(m)
    inv[:, 0, 0] =  (b[:, 1]*c5 - b[:, 2]*c4 + b[:, 3]*c3) * inv_det
    inv[:, 0, 1] = -(a[:, 1]*c5 - a[:, 2]*c4 + a[:, 3]*c3) * inv_det
    inv[:, 0, 2] =  (d[:, 1]*s5 - d[:, 2]*s4 + d[:, 3]*s3) * inv_det
    inv[:, 0, 3] = -(c[:, 1]*s5 - c[:, 2]*s4 + c[:, 3]*s3) * inv_det

    inv[:, 1, 0] = -(b[:, 0]*c5 - b[:, 2]*c2 + b[:, 3]*c1) * inv_det
    inv[:, 1, 1] =  (a[:, 0]*c5 - a[:, 2]*c2 + a[:, 3]*c1) * inv_det
    inv[:, 1, 2] = -(d[:, 0]*s5 - d[:, 2]*s2 + d[:, 3]*s1) * inv_det
    inv[:, 1, 3] =  (c[:, 0]*s5 - c[:, 2]*s2 + c[:, 3]*s1) * inv_det

    inv[:, 2, 0] =  (b[:, 0]*c4 - b[:, 1]*c2 + b[:, 3]*c0) * inv_det
    inv[:, 2, 1] = -(a[:, 0]*c4 - a[:, 1]*c2 + a[:, 3]*c0) * inv_det
    inv[:, 2, 2] =  (d[:, 0]*s4 - d[:, 1]*s2 + d[:, 3]*s0) * inv_det
    inv[:, 2, 3] = -(c[:, 0]*s4 - c[:, 1]*s2 + c[:, 3]*s0) * inv_det

    inv[:, 3, 0] = -(b[:, 0]*c3 - b[:, 1]*c1 + b[:, 2]*c0) * inv_det
    inv[:, 3, 1] =  (a[:, 0]*c3 - a[:, 1]*c1 + a[:, 2]*c0) * inv_det
    inv[:, 3, 2] = -(d[:, 0]*s3 - d[:, 1]*s1 + d[:, 2]*s0) * inv_det
    inv[:, 3, 3] =  (c[:, 0]*s3 - c[:, 1]*s1 + c[:, 2]*s0) * inv_det
    return inv

def homo_warping_3D_with_mask(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    # height, width = src_fea.shape[2], src_fea.shape[3]
    height = torch.tensor([src_fea.shape[2]], dtype=torch.int32, device=src_fea.device)[0]
    width = torch.tensor([src_fea.shape[3]], dtype=torch.int32, device=src_fea.device)[0]

    with torch.no_grad():
        # ref_proj_inv, _ = torch.linalg.inv_ex(ref_proj)
        ref_proj_inv = inverse_4x4(ref_proj)

        proj = torch.matmul(src_proj, ref_proj_inv).to(torch.float16)
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        # y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
        #                        torch.arange(0, width, dtype=torch.float32, device=src_fea.device)], indexing='ij')
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float16, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float16, device=src_fea.device)], indexing='ij')
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth, -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / (proj_xyz[:, 2:3, :, :] + 1e-6)  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    X_mask = ((proj_x_normalized > 1) + (proj_x_normalized < -1)).detach()
    Y_mask = ((proj_y_normalized > 1) + (proj_y_normalized < -1)).detach()
    proj_mask = ((X_mask + Y_mask) > 0).view(batch, num_depth, height, width)
    z = proj_xyz[:, 2:3, :, :].view(batch, num_depth, height, width)
    proj_mask = (proj_mask + (z <= 0)) > 0

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros', align_corners=True)
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea, proj_mask


def diff_homo_warping_3D_with_mask(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    # with torch.no_grad():
    proj = torch.matmul(src_proj, torch.inverse(ref_proj))
    rot = proj[:, :3, :3]  # [B,3,3]
    trans = proj[:, :3, 3:4]  # [B,3,1]

    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                           torch.arange(0, width, dtype=torch.float32, device=src_fea.device)], indexing='ij')
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(height * width), x.view(height * width)
    xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
    xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
    rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
    rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth, -1)  # [B, 3, Ndepth, H*W]
    proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
    proj_xy = proj_xyz[:, :2, :, :] / (proj_xyz[:, 2:3, :, :] + 1e-6)  # [B, 2, Ndepth, H*W]
    proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
    proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
    proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
    grid = proj_xy

    X_mask = ((proj_x_normalized > 1) + (proj_x_normalized < -1)).detach()
    Y_mask = ((proj_y_normalized > 1) + (proj_y_normalized < -1)).detach()
    proj_mask = ((X_mask + Y_mask) > 0).view(batch, num_depth, height, width)
    z = proj_xyz[:, 2:3, :, :].view(batch, num_depth, height, width)
    proj_mask = (proj_mask + (z <= 0)) > 0

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros', align_corners=True)
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea, proj_mask.detach()


def homo_warping_3D(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)], indexing='ij')
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth, -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / (proj_xyz[:, 2:3, :, :] + 1e-6)  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros', align_corners=True)
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea
