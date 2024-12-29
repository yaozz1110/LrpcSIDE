import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MonodepthLoss(nn.modules.Module):
    def __init__(self, input_size, n=4,
                SSIM_w=0.85,
                disp_gradient_w=0.1, lr_w=1,
                focal_x=743, focal_y=743,
                delta_cos=0.867, delta_diff_x=0.01,
                delta_diff_y=0.01, delta_diff_z=0.01,
                delta_z=0.0001, sample_ratio=0.15):
        super(MonodepthLoss, self).__init__()
        self.SSIM_w = SSIM_w
        self.disp_gradient_w = disp_gradient_w
        self.lr_w = lr_w
        self.n = n
        self.input_size = input_size
        self.fx = torch.tensor([focal_x], dtype=torch.float32).to(device)
        self.fy = torch.tensor([focal_y], dtype=torch.float32).to(device)
        self.input_size = input_size
        self.u0 = torch.tensor(input_size[1] // 2, dtype=torch.float32).to(device)
        self.v0 = torch.tensor(input_size[0] // 2, dtype=torch.float32).to(device)
        self.init_image_coor()
        self.delta_cos = delta_cos
        self.delta_diff_x = delta_diff_x
        self.delta_diff_y = delta_diff_y
        self.delta_diff_z = delta_diff_z
        self.delta_z = delta_z
        self.sample_ratio = sample_ratio

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = img.size()
        h = s[2]
        w = s[3]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(nn.functional.interpolate(img,
                               size=[nh, nw], mode='bilinear',
                               align_corners=True))
        return scaled_imgs

    def gradient_x(self, img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
        return gx

    def gradient_y(self, img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
        return gy

    def apply_disparity(self, img, disp):
        batch_size, _, height, width = img.size()

        # Original coordinates of pixels
        x_base = torch.linspace(0, 1, width).repeat(batch_size,
                    height, 1).type_as(img)
        y_base = torch.linspace(0, 1, height).repeat(batch_size,
                    width, 1).transpose(1, 2).type_as(img)

        # Apply shift in X direction
        x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
        # In grid_sample coordinates are assumed to be between -1 and 1
        output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear',
                               padding_mode='zeros')

        return output

    def generate_image_left(self, img, disp):
        return self.apply_disparity(img, -disp)

    def generate_image_right(self, img, disp):
        return self.apply_disparity(img, disp)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)

    def disp_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [torch.exp(-torch.mean(torch.abs(g), 1,
                     keepdim=True)) for g in image_gradients_x]
        weights_y = [torch.exp(-torch.mean(torch.abs(g), 1,
                     keepdim=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i]
                        for i in range(self.n)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i]
                        for i in range(self.n)]

        return [torch.abs(smoothness_x[i]) + torch.abs(smoothness_y[i])
                for i in range(self.n)]

    def compute_chamfer_distance(self, pointcloud1, pointcloud2):
        B, H, W, _ = pointcloud1.shape
        pointcloud1 = pointcloud1.to(device)
        pointcloud2 = pointcloud2.to(device)
        pointcloud1 = pointcloud1.view(B, -1, 3)
        pointcloud2 = pointcloud2.view(B, -1, 3)
        # 计算 pointcloud1 到 pointcloud2 的距离
        dist1 = torch.sum((pointcloud1[:, :, None, :] - pointcloud2[:, None, :, :]) ** 2, dim=-1)
        dist1, _ = torch.min(dist1, dim=-1)
        dist1 = torch.mean(dist1)
        # 计算 pointcloud2 到 pointcloud1 的距离
        dist2 = torch.sum((pointcloud2[:, :, None, :] - pointcloud1[:, None, :, :]) ** 2, dim=-1)
        dist2, _ = torch.min(dist2, dim=-1)
        dist2 = torch.mean(dist2)
        return dist1 + dist2

    def init_image_coor(self):
        x_row = np.arange(0, self.input_size[1])
        x = np.tile(x_row, (self.input_size[0], 1))
        x = x[np.newaxis, :, :]
        x = x.astype(np.float32)
        x = torch.from_numpy(x.copy()).to(device)
        self.u_u0 = x - self.u0

        y_col = np.arange(0, self.input_size[0])  # y_col = np.arange(0, height)
        y = np.tile(y_col, (self.input_size[1], 1)).T
        y = y[np.newaxis, :, :]
        y = y.astype(np.float32)
        y = torch.from_numpy(y.copy()).to(device)
        self.v_v0 = y - self.v0

    def transfer_xyz(self, depth):
        u_u0_resized = F.interpolate(self.u_u0.unsqueeze(0), size=depth.shape[-2:], mode='bilinear',
                                     align_corners=False).squeeze(0)
        v_v0_resized = F.interpolate(self.v_v0.unsqueeze(0), size=depth.shape[-2:], mode='bilinear',
                                     align_corners=False).squeeze(0)
        x = u_u0_resized * torch.abs(depth) / self.fx
        y = v_v0_resized * torch.abs(depth) / self.fy
        z = depth
        x = x.unsqueeze(1)  # [1, 1, 256, 512]
        y = y.unsqueeze(1)  # [1, 1, 256, 512]
        z = z.unsqueeze(1)  # [1, 1, 256, 512]
        pw = torch.cat([x, y, z], 1).permute(0, 2, 3, 1)  # [b, h, w, c]
        return pw

    def select_index(self):
        valid_width = self.input_size[1]
        valid_height = self.input_size[0]
        num = valid_width * valid_height
        p1 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p1)
        p2 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p2)
        p3 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p3)

        p1_x = p1 % self.input_size[1]
        p1_y = (p1 / self.input_size[1]).astype(np.int)

        p2_x = p2 % self.input_size[1]
        p2_y = (p2 / self.input_size[1]).astype(np.int)

        p3_x = p3 % self.input_size[1]
        p3_y = (p3 / self.input_size[1]).astype(np.int)
        p123 = {'p1_x': p1_x, 'p1_y': p1_y, 'p2_x': p2_x, 'p2_y': p2_y, 'p3_x': p3_x, 'p3_y': p3_y}
        return p123

    def form_pw_groups(self, p123, pw):
        """
        Form 3D points groups, with 3 points in each grouup.
        :param p123: points index
        :param pw: 3D points
        :return:
        """
        scale_y, scale_x = pw.size(1) / 256.0, pw.size(2) / 512.0
        # apply the scales to the indices
        p1_x = (torch.from_numpy(p123['p1_x']) * scale_x).long()
        p1_y = (torch.from_numpy(p123['p1_y']) * scale_y).long()
        p2_x = (torch.from_numpy(p123['p2_x']) * scale_x).long()
        p2_y = (torch.from_numpy(p123['p2_y']) * scale_y).long()
        p3_x = (torch.from_numpy(p123['p3_x']) * scale_x).long()
        p3_y = (torch.from_numpy(p123['p3_y']) * scale_y).long()
        pw1 = pw[:, p1_y, p1_x, :]
        pw2 = pw[:, p2_y, p2_x, :]
        pw3 = pw[:, p3_y, p3_x, :]

        # [B, N, 3(x,y,z), 3(p1,p2,p3)]
        pw_groups = torch.cat([pw1[:, :, :, np.newaxis], pw2[:, :, :, np.newaxis], pw3[:, :, :, np.newaxis]], 3)
        return pw_groups

    def filter_mask(self, p123, gt_xyz, delta_cos=0.867,
                    delta_diff_x=0.005,
                    delta_diff_y=0.005,
                    delta_diff_z=0.005):
        pw = self.form_pw_groups(p123, gt_xyz)
        pw12 = pw[:, :, :, 1] - pw[:, :, :, 0]
        pw13 = pw[:, :, :, 2] - pw[:, :, :, 0]
        pw23 = pw[:, :, :, 2] - pw[:, :, :, 1]
        ###ignore linear
        pw_diff = torch.cat([pw12[:, :, :, np.newaxis], pw13[:, :, :, np.newaxis], pw23[:, :, :, np.newaxis]],
                            3)  # [b, n, 3, 3]
        m_batchsize, groups, coords, index = pw_diff.shape
        proj_query = pw_diff.view(m_batchsize * groups, -1, index).permute(0, 2,
                                                                           1)  # (B* X CX(3)) [bn, 3(p123), 3(xyz)]
        proj_key = pw_diff.view(m_batchsize * groups, -1, index)  # B X  (3)*C [bn, 3(xyz), 3(p123)]
        q_norm = proj_query.norm(2, dim=2)
        nm = torch.bmm(q_norm.view(m_batchsize * groups, index, 1), q_norm.view(m_batchsize * groups, 1, index))  # []
        energy = torch.bmm(proj_query, proj_key)  # transpose check [bn, 3(p123), 3(p123)]
        norm_energy = energy / (nm + 1e-8)
        norm_energy = norm_energy.view(m_batchsize * groups, -1)
        mask_cos = torch.sum((norm_energy > delta_cos) + (norm_energy < -delta_cos), 1) > 3  # igonre
        mask_cos = mask_cos.view(m_batchsize, groups)
        ##ignore padding and invilid depth
        mask_pad = torch.sum(pw[:, :, 2, :] > self.delta_z, 2) == 3

        ###ignore near
        mask_x = torch.sum(torch.abs(pw_diff[:, :, 0, :]) < delta_diff_x, 2) > 0
        mask_y = torch.sum(torch.abs(pw_diff[:, :, 1, :]) < delta_diff_y, 2) > 0
        mask_z = torch.sum(torch.abs(pw_diff[:, :, 2, :]) < delta_diff_z, 2) > 0

        mask_ignore = (mask_x & mask_y & mask_z) | mask_cos
        mask_near = ~mask_ignore
        mask = mask_pad & mask_near

        return mask, pw

    def select_points_groups(self, input):
        pw_groups_left_not_ignore = []
        pw_groups_right_not_ignore = []
        pw_left_all = [] 
        pw_right_all = []  
        p123_all = []
        for i in range(len(input)):
            B, C, H, W = input[i].shape
            pw_left = self.transfer_xyz(input[i][:, 0, :, :]) 
            pw_right = self.transfer_xyz(input[i][:, 1, :, :])  
            pw_left_all.append(pw_left) 
            pw_right_all.append(pw_right) 
            p123 = self.select_index()
            p123_all.append(p123)

        for i in range(len(p123_all)):
            p123 = p123_all[i]
            pw_left = pw_left_all[i]
            pw_right = pw_right_all[i]
            mask, pw_groups_left = self.filter_mask(p123, pw_left, delta_cos=0.867, delta_diff_x=0.005, delta_diff_y=0.005,
                                                delta_diff_z=0.005)
            pw_groups_right = self.form_pw_groups(p123, pw_right)
            pw_groups_right[pw_groups_right[:, :, 2, :] == 0] = 0.0001
            mask_broadcast = mask.repeat(1, 9).reshape(B, 3, 3, -1).permute(0, 3, 1, 2)
            pw_groups_right_not_ignore.append(pw_groups_right[mask_broadcast].reshape(1, -1, 3, 3))
            pw_groups_left_not_ignore.append(pw_groups_left[mask_broadcast].reshape(1, -1, 3, 3))
        return pw_groups_left_not_ignore, pw_groups_right_not_ignore

    def forward(self, input, target, select=True):
        """
        Args:
            input [disp1, disp2, disp3, disp4]
            target [left, right]

        Return:
            (float): The loss
        """
        left, right = target
        left_pyramid = self.scale_pyramid(left, self.n)
        right_pyramid = self.scale_pyramid(right, self.n)

        # Prepare disparities
        disp_left_est = [d[:, 0, :, :].unsqueeze(1) for d in input]
        disp_right_est = [d[:, 1, :, :].unsqueeze(1) for d in input]

        self.disp_left_est = disp_left_est
        self.disp_right_est = disp_right_est
        # Generate images
        left_est = [self.generate_image_left(right_pyramid[i],
                    disp_left_est[i]) for i in range(self.n)]
        right_est = [self.generate_image_right(left_pyramid[i],
                     disp_right_est[i]) for i in range(self.n)]
        self.left_est = left_est
        self.right_est = right_est

        # L-R Consistency
        right_left_disp = [self.generate_image_left(disp_right_est[i],
                           disp_left_est[i]) for i in range(self.n)]
        left_right_disp = [self.generate_image_right(disp_left_est[i],
                           disp_right_est[i]) for i in range(self.n)]

        # Disparities smoothness
        disp_left_smoothness = self.disp_smoothness(disp_left_est,
                                                    left_pyramid)
        disp_right_smoothness = self.disp_smoothness(disp_right_est,
                                                     right_pyramid)

        # L1
        l1_left = [torch.mean(torch.abs(left_est[i] - left_pyramid[i]))
                   for i in range(self.n)]
        l1_right = [torch.mean(torch.abs(right_est[i]
                    - right_pyramid[i])) for i in range(self.n)]

        # SSIM
        ssim_left = [torch.mean(self.SSIM(left_est[i],
                     left_pyramid[i])) for i in range(self.n)]
        ssim_right = [torch.mean(self.SSIM(right_est[i],
                      right_pyramid[i])) for i in range(self.n)]

        image_loss_left = [self.SSIM_w * ssim_left[i]
                           + (1 - self.SSIM_w) * l1_left[i]
                           for i in range(self.n)]
        image_loss_right = [self.SSIM_w * ssim_right[i]
                            + (1 - self.SSIM_w) * l1_right[i]
                            for i in range(self.n)]
        image_loss = sum(image_loss_left + image_loss_right)

        # L-R Consistency
        lr_left_loss = [torch.mean(torch.abs(right_left_disp[i]
                        - disp_left_est[i])) for i in range(self.n)]
        lr_right_loss = [torch.mean(torch.abs(left_right_disp[i]
                         - disp_right_est[i])) for i in range(self.n)]
        lr_loss = sum(lr_left_loss + lr_right_loss)

        # Disparities smoothness
        disp_left_loss = [torch.mean(torch.abs(
                          disp_left_smoothness[i])) / 2 ** i
                          for i in range(self.n)]
        disp_right_loss = [torch.mean(torch.abs(
                           disp_right_smoothness[i])) / 2 ** i
                           for i in range(self.n)]
        disp_gradient_loss = sum(disp_left_loss + disp_right_loss)

        # 3D geometric constraint loss

        gt_points_list, dt_points_list = self.select_points_groups(input)

        loss_list = []
        for idx in range(len(gt_points_list)):
            gt_points = gt_points_list[idx]
            dt_points = dt_points_list[idx]
            gt_p12 = gt_points[:, :, :, 1] - gt_points[:, :, :, 0]
            gt_p13 = gt_points[:, :, :, 2] - gt_points[:, :, :, 0]
            dt_p12 = dt_points[:, :, :, 1] - dt_points[:, :, :, 0]
            dt_p13 = dt_points[:, :, :, 2] - dt_points[:, :, :, 0]
            gt_normal = torch.cross(gt_p12, gt_p13, dim=2)
            dt_normal = torch.cross(dt_p12, dt_p13, dim=2)
            dt_norm = torch.norm(dt_normal, 2, dim=2, keepdim=True)
            gt_norm = torch.norm(gt_normal, 2, dim=2, keepdim=True)
            dt_mask = dt_norm == 0.0
            gt_mask = gt_norm == 0.0
            dt_mask = dt_mask.to(torch.float32)
            gt_mask = gt_mask.to(torch.float32)
            dt_mask *= 0.01
            gt_mask *= 0.01
            gt_norm = gt_norm + gt_mask
            dt_norm = dt_norm + dt_mask
            gt_normal = gt_normal / gt_norm
            dt_normal = dt_normal / dt_norm
            loss_3d = torch.abs(gt_normal - dt_normal)
            loss_3d = torch.sum(torch.sum(loss_3d, dim=2), dim=0)
            if select:
                loss_3d, indices = torch.sort(loss_3d, dim=0, descending=False)
            loss = loss_3d[int(loss_3d.size(0) * 0.25):]
            loss = torch.mean(loss)

        loss_list.append(loss)
        final_3d_loss = torch.mean(torch.stack(loss_list))  

   
        cd_loss = []
        for gt_points, dt_points in zip(gt_points_list, dt_points_list):
            loss = self.compute_chamfer_distance(gt_points, dt_points)
        cd_loss.append(loss)
        final_cd_loss = torch.mean(torch.stack(cd_loss))

        loss = image_loss + self.disp_gradient_w * disp_gradient_loss\
               + self.lr_w * lr_loss + final_3d_loss + final_cd_loss
        self.image_loss = image_loss
        self.disp_gradient_loss = disp_gradient_loss
        self.lr_loss = lr_loss
        self.final_3d_loss = final_3d_loss
        self.final_cd_loss = final_cd_loss
        return loss
