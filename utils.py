import torch
import torch.nn as nn
import numpy as np
import os
import cv2 as cv
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.utils as vutils
from torchsummary import summary
import glob
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import re, PIL, math
from PIL import Image
import random, time
from torch.nn import init
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import tqdm

class SODLoaderAugment(Dataset):
	def __init__(self, mode='train', im_size = 256):
		if mode == 'train':
			self.inp_path = './AugmentedDUTS384/Image'
			self.out_path = './AugmentedDUTS384/Mask'
			self.contour_path = './AugmentedDUTS384/Contour'
		elif mode == 'test':
			self.inp_path = './DUTS-TE/DUTS-TE/DUTS-TE-Image'
			self.out_path = './DUTS-TE/DUTS-TE/DUTS-TE-Mask'
		else:
			print("mode should be either 'train' or 'test'.")
			sys.exit(0)

		self.mode = mode
		self.im_size = im_size

		self.inp_files = sorted(glob.glob(self.inp_path + '/*'))
		self.out_files = sorted(glob.glob(self.out_path + '/*'))
		if self.mode != 'test':
			self.contour_files = sorted(glob.glob(self.contour_path + '/*'))

	def __getitem__(self, idx):
		inp_img = cv.imread(self.inp_files[idx])
		inp_img = cv.cvtColor(inp_img, cv.COLOR_BGR2RGB)
		inp_img = inp_img.astype('float32')

		mask_img = cv.imread(self.out_files[idx], 0)
		mask_img = mask_img.astype('float32')
		mask_img /= np.max(mask_img)

		if self.mode != 'test':
			contour_img = cv.imread(self.contour_files[idx], 0)
			contour_img = contour_img.astype('float32')
			contour_img /= np.max(contour_img)

		if self.im_size is not None:
			if self.mode == 'train':
				inp_img = cv.resize(inp_img, (self.im_size, self.im_size), interpolation = cv.INTER_AREA)
				mask_img = cv.resize(mask_img, (self.im_size, self.im_size), interpolation = cv.INTER_AREA)
				contour_img = cv.resize(contour_img, (self.im_size, self.im_size), interpolation = cv.INTER_AREA)


		inp_img /= np.max(inp_img)
		inp_img = np.transpose(inp_img, axes=(2, 0, 1))
		inp_img = torch.from_numpy(inp_img).float()

		mask_img = np.expand_dims(mask_img, axis=0)

		if self.mode != 'test':

			contour_img = np.expand_dims(contour_img, axis=0)

			return inp_img, torch.from_numpy(mask_img).float(), torch.from_numpy(contour_img).float()
		
		else:
			return inp_img, torch.from_numpy(mask_img).float()

	def __len__(self):
		return len(self.inp_files)

class COCOLoaderAugment(Dataset):
	def __init__(self, im_size=256):
		self.inp_path = './AugmentedCOCO/Image'
		self.out_path = './AugmentedCOCO/Mask'
		self.contour_path = './AugmentedCOCO/Contour'

		self.im_size = im_size

		self.inp_files = sorted(glob.glob(self.inp_path + '/*'))
		self.out_files = sorted(glob.glob(self.out_path + '/*'))
		self.contour_files = sorted(glob.glob(self.contour_path + '/*'))

	def __getitem__(self, idx):
		inp_img = cv.imread(self.inp_files[idx])
		inp_img = cv.cvtColor(inp_img, cv.COLOR_BGR2RGB)
		inp_img = inp_img.astype('float32')

		mask_img = cv.imread(self.out_files[idx], 0)
		mask_img = mask_img.astype('float32')
		if np.max(mask_img) != 255.0:
			pass
		else:
			mask_img /= np.max(mask_img)

		contour_img = cv.imread(self.contour_files[idx], 0)
		contour_img = contour_img.astype('float32')
		if np.max(contour_img) != 255.0:
			pass
		else:
			contour_img /= np.max(contour_img)

		if self.im_size is not None:
			if self.mode == 'train':
				inp_img = cv.resize(inp_img, (self.im_size, self.im_size), interpolation = cv.INTER_AREA)
				mask_img = cv.resize(mask_img, (self.im_size, self.im_size), interpolation = cv.INTER_AREA)
				contour_img = cv.resize(contour_img, (self.im_size, self.im_size), interpolation = cv.INTER_AREA)

		inp_img /= np.max(inp_img)
		inp_img = np.transpose(inp_img, axes=(2, 0, 1))
		inp_img = torch.from_numpy(inp_img).float()

		mask_img = np.expand_dims(mask_img, axis=0)

		contour_img = np.expand_dims(contour_img, axis=0)

		return inp_img, torch.from_numpy(mask_img).float(), torch.from_numpy(contour_img).float()

	def __len__(self):
		return len(self.inp_files)

class OpenImagesLoader(Dataset):
	
	def __init__(self, im_size=256):
		self.inp_path = './train_data_oiv7/Image'
		self.out_path = './train_data_oiv7/Mask'
		self.contour_path = './train_data_oiv7/Contour'

		self.im_size = im_size

		self.inp_files = sorted(glob.glob(self.inp_path + '/*'))
		self.out_files = sorted(glob.glob(self.out_path + '/*'))
		self.contour_files = sorted(glob.glob(self.contour_path + '/*'))

	def __getitem__(self, idx):
		inp_img = cv.imread(self.inp_files[idx])
		inp_img = cv.cvtColor(inp_img, cv.COLOR_BGR2RGB)
		inp_img = inp_img.astype('float32')

		mask_img = cv.imread(self.out_files[idx], 0)
		mask_img = mask_img.astype('float32')
		if np.max(mask_img) != 255.0:
			pass
		else:
			mask_img /= np.max(mask_img)

		contour_img = cv.imread(self.contour_files[idx], 0)
		contour_img = contour_img.astype('float32')
		if np.max(contour_img) != 255.0:
			pass
		else:
			contour_img /= np.max(contour_img)

		if self.im_size is not None:
			inp_img = cv.resize(inp_img, (self.im_size, self.im_size), interpolation = cv.INTER_AREA)
			mask_img = cv.resize(mask_img, (self.im_size, self.im_size), interpolation = cv.INTER_AREA)
			contour_img = cv.resize(contour_img, (self.im_size, self.im_size), interpolation = cv.INTER_AREA)

		inp_img /= np.max(inp_img)
		inp_img = np.transpose(inp_img, axes=(2, 0, 1))
		inp_img = torch.from_numpy(inp_img).float()

		mask_img = np.expand_dims(mask_img, axis=0)

		contour_img = np.expand_dims(contour_img, axis=0)

		return inp_img, torch.from_numpy(mask_img).float(), torch.from_numpy(contour_img).float()

	def __len__(self):
		return len(self.inp_files)

def MAE(tensors, gts):
	b, _, h, w = tensors.size()
	res = 0.0
	for i in range(len(tensors)):
		res += (torch.sum(torch.abs(tensors[i, :, :, :] - gts[i, :, :, :]))/(h * w))

	return res/b

def get_mae(model, dataloader, cuda, im_size):
	model.eval()
	iou_l = 0.0
	mae = 0.0
	j = nn.Sigmoid()
	count = 0
	pred_list = []
	gt_list = []
	transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((im_size, im_size)), transforms.ToTensor()])
	for x, z in tqdm.tqdm(dataloader):
		count += 1
		_, _, h, w = x.size()
		x = transform(x.squeeze(0)).unsqueeze(0)
		prediction = model(x.to(cuda))[-1]
		pred = j(prediction[-1])
		up = nn.Upsample(size=(h, w), mode='bilinear',align_corners=False)
		pred = up(pred)
		for i in range(len(pred)):
			pred_list.append((pred[i, :, :, :] / torch.max(pred[i, :, :, :])).detach().squeeze().cpu().numpy())
			gt_list.append(z[i, :, :, :].detach().squeeze().cpu().numpy())
		mae += MAE(pred.cpu(), z.cpu())
	print("MAE = %.4f" %(mae / count), flush = True)
	return pred_list, gt_list, mae/count

def compute_pre_rec(gt,mask,mybins=np.arange(0,256)):

	if(len(gt.shape)<2 or len(mask.shape)<2):
		print("ERROR: gt or mask is not matrix!")
		exit()
	if(len(gt.shape)>2): # convert to one channel
		gt = gt[:,:,0]
	if(len(mask.shape)>2): # convert to one channel
		mask = mask[:,:,0]
	if(gt.shape!=mask.shape):
		print("ERROR: The shapes of gt and mask are different!")
		exit()

	gtNum = gt[gt>128].size # pixel number of ground truth foreground regions
	pp = mask[gt>128] # mask predicted pixel values in the ground truth foreground region
	nn = mask[gt<=128] # mask predicted pixel values in the ground truth bacground region

	pp_hist,pp_edges = np.histogram(pp,bins=mybins) #count pixel numbers with values in each interval [0,1),[1,2),...,[mybins[i],mybins[i+1]),...,[254,255)
	nn_hist,nn_edges = np.histogram(nn,bins=mybins)

	pp_hist_flip = np.flipud(pp_hist) # reverse the histogram to the following order: (255,254],...,(mybins[i+1],mybins[i]],...,(2,1],(1,0]
	nn_hist_flip = np.flipud(nn_hist)

	pp_hist_flip_cum = np.cumsum(pp_hist_flip) # accumulate the pixel number in intervals: (255,254],(255,253],...,(255,mybins[i]],...,(255,0]
	nn_hist_flip_cum = np.cumsum(nn_hist_flip)

	precision = pp_hist_flip_cum/(pp_hist_flip_cum + nn_hist_flip_cum+1e-8) #TP/(TP+FP)
	recall = pp_hist_flip_cum/(gtNum+1e-8) #TP/(TP+FN)

	precision[np.isnan(precision)]= 0.0
	recall[np.isnan(recall)] = 0.0

	return np.reshape(precision,(len(precision))),np.reshape(recall,(len(recall)))


def compute_PRE_REC_FM_of_methods(gt_name_list,pred_name_list,rs_dir_lists=1,beta=0.3):
#input 'gt_name_list': ground truth name list
#input 'rs_dir_lists': to-be-evaluated mask directories (not the file names, just folder names)
#output precision 'PRE': numpy array with shape of (num_rs_dir, 256)
#       recall    'REC': numpy array with shape of (num_rs_dir, 256)
#       F-measure (beta) 'FM': numpy array with shape of (num_rs_dir, 256)

	mybins = np.arange(0,256) # different thresholds to achieve binarized masks for pre, rec, Fm measures

	num_gt = len(gt_name_list) # number of ground truth files
	num_rs_dir = 1 # number of method folders
	if(num_gt==0):
		#print("ERROR: The ground truth directory is empty!")
		exit()

	PRE = np.zeros((num_gt,num_rs_dir,len(mybins)-1)) # PRE: with shape of (num_gt, num_rs_dir, 256)
	REC = np.zeros((num_gt,num_rs_dir,len(mybins)-1)) # REC: the same shape with PRE
	# FM = np.zeros((num_gt,num_rs_dir,len(mybins)-1)) # Fm: the same shape with PRE
	gt2rs = np.zeros((num_gt,num_rs_dir)) # indicate if the mask of methods is correctly computed

	for i in range(0,num_gt):
		# print('>>Processed %d/%d'%(i+1,num_gt),end='\r')
		gt = gt_name_list[i]# read ground truth
		gt = gt*255.0 # convert gt to [0,255]
		#gt_name = gt_name_list[i].split('/')[-1] # get the file name of the ground truth "xxx.png"

		for j in range(0,num_rs_dir):
			pre, rec, f = np.zeros(len(mybins)), np.zeros(len(mybins)), np.zeros(len(mybins)) # pre, rec, f or one mask w.r.t different thresholds
			try:
				rs = pred_name_list[i] # read the corresponding mask from each method
				rs = rs*255.0 # convert rs to [0,255]
			except IOError:
				#print('ERROR: Couldn\'t find the following file:',rs_dir_lists[j]+gt_name)
				continue
			try:
				pre, rec = compute_pre_rec(gt,rs,mybins=np.arange(0,256))
			except IOError:
				#print('ERROR: Fails in compute_mae!')
				continue

			PRE[i,j,:] = pre
			REC[i,j,:] = rec
			gt2rs[i,j] = 1.0
	print('\n')
	gt2rs = np.sum(gt2rs,0) # num_rs_dir
	gt2rs = np.repeat(gt2rs[:, np.newaxis], 255, axis=1) #num_rs_dirx255

	PRE = np.sum(PRE,0)/(gt2rs+1e-8) # num_rs_dirx255, average PRE over the whole dataset at every threshold
	REC = np.sum(REC,0)/(gt2rs+1e-8) # num_rs_dirx255
	FM = (1+beta)*PRE*REC/(beta*PRE+REC+1e-8) # num_rs_dirx255

	return PRE, REC, FM, gt2rs

def validation(model, dataloader, cuda, im_size, use_depth = False):
	with torch.no_grad():
		if use_depth:
			pred_list, gt_list, mae = get_mae_depthv2(model, dataloader, cuda, im_size)
		else:
			pred_list, gt_list, mae = get_mae(model, dataloader, cuda, im_size)
		PRE, REC, FM, gt2rs_fm = compute_PRE_REC_FM_of_methods(gt_list,pred_list,1, beta=0.3)
		for i in range(0,FM.shape[0]):
			print(">>", "My Method",":", "num_rs/num_gt-> %d/%d,"%(int(gt2rs_fm[i][0]),len(gt_list)), "maxF->%.3f, "%(np.max(FM,1)[i]), "meanF->%.3f, "%(np.mean(FM,1)[i]))
	return mae
