'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
import os
import cv2
import torch
import numpy as np
from parse_model import ParseNet
import torch.nn.functional as F


class FaceParse:

	def __init__(self, base_dir='./', model='ParseNet-latest', device='cuda'):
		self.mfile = os.path.join(base_dir, 'facedetection', model + '.pth')
		self.size = 512
		self.device = device
		self.MASK_COLORMAP = torch.tensor([0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 255, 255, 0], device=self.device)
		self.load_model()
		print("FaceParse initialized")

	def load_model(self):
		self.faceparse = ParseNet(self.size, self.size, 32, 64, 19, norm_type='bn', relu_type='LeakyReLU', ch_range=[32, 256])
		self.faceparse.load_state_dict(torch.load(self.mfile, map_location=self.device))
		self.faceparse.to(self.device)
		self.faceparse.eval()

	def process(self, im):
		im = cv2.resize(im, (self.size, self.size))
		imt = self.img2tensor(im)
		with torch.no_grad():
			pred_mask, _ = self.faceparse(imt)
		mask = self.tensor2mask(pred_mask)
		return mask

	def process_tensor(self, imt):
		imt = F.interpolate(imt.flip(1) * 2 - 1, (self.size, self.size))
		with torch.no_grad():
			pred_mask, _ = self.faceparse(imt)
		mask = pred_mask.argmax(dim=1)
		mask = self.MASK_COLORMAP[mask].unsqueeze(0)
		return mask

	def img2tensor(self, img):
		img = img[..., ::-1]  # Convert BGR to RGB
		img = (img / 255.0) * 2 - 1  # Scale image to [-1, 1]
		img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
		return img_tensor.float()

	def tensor2mask(self, tensor):
		if len(tensor.shape) < 4:
			tensor = tensor.unsqueeze(0)
		if tensor.shape[1] > 1:
			tensor = tensor.argmax(dim=1)
		tensor = tensor.squeeze(1).cpu().numpy()
		color_maps = [self.MASK_COLORMAP[t].cpu().numpy().astype(np.uint8) for t in tensor]
		return color_maps
