import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import cv2
import matplotlib.image as pim
from PIL import Image

# ../data/output/sa_mondepth//sa100_4pp.npy
# ../data/kitti/test/image_02/gt_npy/001.npy
filename = '00000000.npy'#"../data/output/disparities.npy"

npy = np.load(filename)

print(npy)
#mask = npy >0
depth = npy/100

# for i in range(npy.shape[0]):
#     a = npy[i, :, :]
#     disp_resized_np = a.squeeze()
#     vmax = np.percentile(disp_resized_np, 95)
#     normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
#     mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
#     colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
#     colormapped_im1 = cv2.cvtColor(colormapped_im, cv2.COLOR_BGR2RGB)
#     width = 640
#     height = 480
#     colormapped_im1 = cv2.resize(colormapped_im1, (width, height))
#     cv2.imwrite("../data/depth_img/monodepth1/"+str(i)+".png", colormapped_im1)
# # ../data/depth_img/sa_monodepth/sa100

#####为了显示单幅真实深度图 导图单个.npy
# mask = depth > 0
# depth = depth[mask]

disp_resized_np = depth.squeeze()
vmax = np.percentile(disp_resized_np, 95)
normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
colormapped_im1 = cv2.cvtColor(colormapped_im, cv2.COLOR_BGR2RGB)
cv2.imwrite("./depth_image/"+".png", colormapped_im1)

