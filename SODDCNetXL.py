import torch.nn as nn
import torch
import os
import torchvision
import torch.nn.functional as F
import random
from torch.nn import init
import math
from torch.autograd import Variable
from math import exp
import numpy as np
from itertools import combinations

class DoubleConv(nn.Module):
	"""(convolution => [BN] => ReLU) * 2"""

	def __init__(self, in_channels, out_channels, kernel_size = 3, mid_channels=None):
		super().__init__()
		if not mid_channels:
			mid_channels = out_channels
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, mid_channels, kernel_size= kernel_size, padding = kernel_size // 2, dilation = 1),
			nn.BatchNorm2d(mid_channels),
			nn.GELU(),
			nn.Conv2d(mid_channels, out_channels, kernel_size= kernel_size, padding=kernel_size//2, dilation=1),
			nn.BatchNorm2d(out_channels),
			nn.GELU()
		)

	def forward(self, x):
		return self.double_conv(x)

class Down(nn.Module):
	"""Downscaling with maxpool then double conv"""

	def __init__(self, in_channels, out_channels, mid_channels = None):
		super().__init__()
		self.maxpool_conv = nn.Sequential(
		nn.MaxPool2d(2),
		DoubleConv(in_channels, out_channels)
		)

	def forward(self, x):
		return self.maxpool_conv(x)

class DynConvolution(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding=1, groups=1, factor=128, downsample=False):
		super().__init__()
		self.kernel_size = kernel_size
		self.padding = padding
		self.dilation = dilation
		self.factor = factor
		self.out_channels = out_channels
		self.downsample = downsample
		if downsample:
			self.compress = nn.Sequential(
				nn.Conv2d(in_channels, factor, kernel_size, stride=2, padding=kernel_size//2),
				nn.BatchNorm2d(factor),
				nn.GELU()

			)
			self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		else:
			self.compress = nn.Sequential(
				nn.Conv2d(in_channels, factor, kernel_size, stride=1, padding=kernel_size//2),
				nn.BatchNorm2d(factor),
				nn.GELU()

			)
		self.norm = nn.GroupNorm(32, out_channels)
		self.act = nn.GELU()
		self.register_parameter('conv_weight', 
			nn.Parameter(torch.randn(out_channels, factor, kernel_size, kernel_size)))
		n = kernel_size * kernel_size * factor
		init.normal_(self.conv_weight.data, 0.0, math.sqrt(2.0 / n))

	def forward(self, x, dyn_weight):
		
		compress_x = self.compress(x)  # shape: (b, factor, h, w)
		b, c, h, w = compress_x.shape
		unfolded_x = F.unfold(
			compress_x, 
			kernel_size=self.kernel_size,
			padding=self.padding, 
			dilation=self.dilation
		)  # shape: (b, factor*k*k, h*w)
		dyn_weight = dyn_weight.view(b, 1, self.kernel_size*self.kernel_size, h*w)
		new_weights = self.conv_weight.view(self.out_channels, -1)
		new_weights = new_weights.unsqueeze(0).expand(b, -1, -1)

		mod_x = unfolded_x.view(b, c, -1, h*w) * dyn_weight
		dil_feat = torch.bmm(new_weights, mod_x.view(b, -1, h*w)).view(b, self.out_channels, h, w)

		# 9) Normalization + activation
		dil_feat = self.norm(dil_feat)
		dil_feat = self.act(dil_feat)

		if self.downsample:
			return self.upsample(dil_feat)

		return dil_feat

class Up(nn.Module):
	"""Upscaling then double conv"""

	def __init__(self, in_channels, out_channels, kernel_size = 3, bilinear=True):
		super().__init__()
		if bilinear:
			self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
			self.conv = DoubleConv(in_channels, out_channels, kernel_size, in_channels // 2)
		else:
			self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
			self.conv = DoubleConv(in_channels, out_channels, kernel_size)


	def forward(self, x1, x2):
		x1 = self.up(x1)
		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]

		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
		diffY // 2, diffY - diffY // 2])
		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)

		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
		diffY // 2, diffY - diffY // 2])
		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)

class OutConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(OutConv, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

	def forward(self, x):
		return self.conv(x)

class DoubleConvMod(nn.Module):
	"""(convolution => [GN] => ReLU) * 1"""

	def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, mid_channels=None):
		super().__init__()
		if not mid_channels:
			mid_channels = out_channels
		if mid_channels % 32 != 0:
			groups_n = 16
		else:
			groups_n = 32
		self.double_conv = nn.Sequential(
		nn.Conv2d(in_channels, mid_channels, kernel_size= kernel_size, stride=stride, padding=padding, dilation=dilation, groups = groups),
		nn.GroupNorm(groups_n, mid_channels),
		nn.GELU()
		)

	def forward(self, x):
		return self.double_conv(x)

class DoubleConvMody(nn.Module):
	"""(convolution => [BN] => ReLU) * 1"""

	def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, mid_channels=None):
		super().__init__()
		if not mid_channels:
			mid_channels = out_channels
		self.double_conv = nn.Sequential(
		nn.Conv2d(in_channels, mid_channels, kernel_size= kernel_size, stride=stride,padding=padding, dilation=dilation, groups = groups),
		nn.BatchNorm2d(mid_channels),
		nn.GELU(),
		)

	def forward(self, x):
		return self.double_conv(x)

class DoubleConvLS(nn.Module):
	"""(convolution => [GN] => ReLU) * 1"""

	def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, mid_channels=None):
		super().__init__()
		if not mid_channels:
			mid_channels = out_channels
		if out_channels < 32:
			if out_channels < 16:
				groups_n = 8
			else:
				groups_n = 16
		else:
			groups_n = 32
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, mid_channels, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, groups = groups),
			nn.GroupNorm(groups_n, mid_channels),
			nn.GELU()
		)

	def forward(self, x):
		return self.conv(x)

class Self_Attn(nn.Module):
	def __init__(self,in_dim,factor=1,padding=1,dilation=1,sr_ratio=1):
		super(Self_Attn,self).__init__()
		self.factor = factor
		self.chanel_in = in_dim
		if sr_ratio > 1:
			self.key_conv = DoubleConvMod(in_dim ,in_dim//factor , kernel_size= sr_ratio, stride = sr_ratio, padding = 0, dilation=1)
			self.value_conv = DoubleConvMod(in_dim ,in_dim , kernel_size= sr_ratio, stride = sr_ratio, padding = 0, dilation=1)
		else:
			self.key_conv = DoubleConvMod(in_dim ,in_dim//factor , kernel_size= 1, padding = 0, dilation=1)
			self.value_conv = DoubleConvMod(in_dim ,in_dim , kernel_size= 1, padding = 0, dilation=1)

		self.query_conv = DoubleConvMod(in_dim , in_dim//factor , kernel_size= 1, padding = 0, dilation=1)

		self.softmax  = nn.Softmax(dim=-1)
		self.C = in_dim
		self.sr_ratio = sr_ratio

	def forward(self, x):
		m_batchsize,C,width ,height = x.size()
		proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X C X (N)
		proj_key =  self.key_conv(x).view(m_batchsize,-1,width // self.sr_ratio * height//self.sr_ratio) # B X C x (*W*H)
		energy =  torch.bmm(proj_query,proj_key)
		attention = self.softmax(energy / ((self.chanel_in // self.factor) ** 0.5)) # BX (N) X (N)
		proj_value = self.value_conv(x).view(m_batchsize,-1,width//self.sr_ratio*height//self.sr_ratio) # B X C X N

		out = torch.bmm(proj_value,attention.permute(0,2,1) )
		out = out.view(m_batchsize,self.C,width,height)

		return out

class DLRU(nn.Module):
	def __init__(
		self, ns, in_channels, out_channels,
		mid_channels = 32, factor = 4, factorw = 1, 
		feat_size = None, dilation_rates = None, conv_sizes = None, levels = 1,
		conv_levels = 1
	):
		super(DLRU, self).__init__()

		self.dyn_convs = nn.ModuleDict({})
		self.layers_pre = nn.ModuleDict({})
		self.layers_post = nn.ModuleDict({})
		self.change_attention_channels = nn.ModuleDict({})

		self.ns = ns
		self.out_channels = out_channels
		self.conv_sizes = conv_sizes

		spatial_res = feat_size // factor
		factorw = 1

		print("Attention Map Spatial Resolution is (%d x %d)" %(spatial_res, spatial_res), flush = True)

		self.rescon = DoubleConv(in_channels, mid_channels // factorw)
		if factor > 1:
			self.attn_pre = nn.Sequential(
				nn.AvgPool2d(factor, factor),
				DoubleConv(in_channels, mid_channels // factorw)
			)

		else:
			self.attn_pre = DoubleConv(in_channels, mid_channels // factorw)

		self.attn = nn.ModuleList(
			[Self_Attn(mid_channels // factorw) for _ in range(levels)]
		)
		if self.ns >= 1:
			for d_rate, k_size in zip(dilation_rates, conv_sizes):
				kernel_size = k_size + (k_size - 1) * (d_rate - 1)

				self.dyn_convs[str(k_size)] = nn.ModuleList([
					DynConvolution(in_channels=mid_channels, out_channels=mid_channels, kernel_size=k_size, padding=kernel_size // 2, 
					dilation=d_rate, downsample=True if factor > 1 else False) for _ in range(conv_levels)
				])

				self.layers_pre[str(k_size)] = nn.ModuleList([
					DoubleConvLS(mid_channels // factorw, mid_channels, 3, 1, 1, 1) for _ in range(conv_levels)
				])

				self.layers_post[str(k_size)] = nn.ModuleList([
					DoubleConvLS(mid_channels, mid_channels // factorw, 3, 1, 1, 1) for _ in range(conv_levels)
				])

				self.change_attention_channels[str(k_size)] = DoubleConvMody(mid_channels // factorw, k_size ** 2, 1, 1, 0)

			self.conv = Down((ns + 1) * (mid_channels // factorw), self.out_channels)

	def forward(self, x):
		ns = self.ns
		attn_weight_c = self.attn_pre(x)

		for i in range(len(self.attn)):
			if i == 0:
				attn_weight = self.attn[i](attn_weight_c)
			else:
				attn_weight = self.attn[i](attn_weight)

		x = self.rescon(x)

		dyn_conv_features = []
		out_dil_features = []

		for k_size in self.conv_sizes:

			dyn_weights = self.change_attention_channels[str(k_size)](attn_weight)
			dyn_weights = dyn_weights.unsqueeze(1).unsqueeze(2).view(x.size()[0], 1, k_size, k_size, *attn_weight.size()[2:])

			for ix in range(len(self.dyn_convs[str(k_size)])):

				if ix == 0:
					inp_feat_pre = self.layers_pre[str(k_size)][ix](x)
				else:
					inp_feat_pre = self.layers_pre[str(k_size)][ix](inp_feat_post)

				dil_feat = self.dyn_convs[str(k_size)][ix](inp_feat_pre, dyn_weights)
				inp_feat_post = self.layers_post[str(k_size)][ix](dil_feat) + x

			dyn_conv_features.append(inp_feat_post)
			out_dil_features.append(inp_feat_post)

		dyn_conv_features.append(x)

		return self.conv(torch.cat(dyn_conv_features, dim = 1)), out_dil_features

class SODDCNetXL(nn.Module):

	def __init__(self, 
		n_channels, 
		n_classes, 
		bilinear = True, 
		use_contour = False,
		deep_supervision = False, 
		factorw = 1,
		factorwo = 1,
		img_res = 320,
		dilation_rates = [[1, 2, 3, 4, 5], [1, 2, 3, 4]],
		conv_sizes = [[5, 7, 9, 11, 13], [5, 7, 9, 11]],
		levels = [4, 8],
		conv_levels = [4, 8]
	):

		super(SODDCNetXL, self).__init__()
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.bilinear = bilinear
		self.inc = nn.Sequential(
			DoubleConv(n_channels, 32 * factorwo),
			Down(32 * factorwo, 32 * factorwo),
			Down(32 * factorwo, 32 * factorwo)
		)

		print("Dilation Rates - ", dilation_rates, flush = True)

		print("Convolutional Kernel Sizes - ", conv_sizes, flush = True)

		self.down1 = DLRU(
			(len(dilation_rates[0])), 32 * factorwo, 64 * factorwo, 
			32 * (factorw // 2), 2, factorw = (factorw // 2), feat_size = img_res // 2, dilation_rates = dilation_rates[0], 
			conv_sizes = conv_sizes[0], levels = levels[0], conv_levels = conv_levels[0]
		)
		factor = 2 if bilinear else 1
		self.down2 = DLRU(
			(len(dilation_rates[1])), 64 * factorwo, (128 * factorwo) // factor, 
			32 * factorw, 1, factorw = factorw, feat_size = img_res // 4, dilation_rates = dilation_rates[1], 
			conv_sizes = conv_sizes[1], levels = levels[1], conv_levels = conv_levels[1]
		)

		self.up3 = Up(128 * factorwo, (64 * factorwo) // factor, bilinear = bilinear)
		self.up4 = Up(64 * factorwo, 32 * factorwo, bilinear = bilinear)

		self.outc = OutConv(32 * factorwo, n_classes); self.outs = OutConv(32 * factorwo, n_classes)


		self.contour = use_contour
		self.deep_supervision = deep_supervision

		if self.deep_supervision:
			self.outsdd = OutConv(64 * factorwo, n_classes); self.outsrdd = OutConv((128 * factorwo) // factor, n_classes)

			self.out_sal = nn.ModuleList([])

			for _ in range(len(dilation_rates[0])):
				self.out_sal.append(OutConv(32 * (factorw // 2), n_classes))

			for _ in range(len(dilation_rates[1])):
				self.out_sal.append(OutConv(32 * factorw, n_classes))

		if self.contour:

			self.enccontour1 = OutConv(64 * factorwo, n_classes)
			self.enccontour2 = OutConv((128 * factorwo) // factor, n_classes)

			self.outcontour1 = OutConv(32 * factorwo, n_classes)
			self.outcontour2 = OutConv(32 * factorwo, n_classes)

			self.out_con = nn.ModuleList([])

			for _ in range(len(dilation_rates[0])):
				self.out_con.append(OutConv(32 * (factorw // 2), n_classes))

			for _ in range(len(dilation_rates[1])):
				self.out_con.append(OutConv(32 * factorw, n_classes))

		self.up2b = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
		self.up3b = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
		self.up4b = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

	def forward(self, x):

		saliency = []
		contours = []

		x1 = self.inc(x)

		ds1, dil_feats1 = self.down1(x1)
		x2 = ds1

		ds2, dil_feats2 = self.down2(x2)

		x3 = ds2

		if self.deep_supervision:

			ll_smallest = self.up3b(self.outsdd(ds1))
			saliency.append(ll_smallest)

			ll_small = self.up2b(self.outsrdd(ds2))
			saliency.append(ll_small)

			for ix in range(len(dil_feats1)):
				saliency.append(self.up4b(self.out_sal[ix](dil_feats1[ix])))

			for ix in range(len(dil_feats2)):
				saliency.append(self.up3b(self.out_sal[len(dil_feats1) + ix](dil_feats2[ix])))

		if self.contour:

			ll_smallest_contour = self.up3b(self.enccontour1(ds1))
			contours.append(ll_smallest_contour)

			ll_small_contour = self.up2b(self.enccontour2(ds2))
			contours.append(ll_small_contour)

			for ix in range(len(dil_feats1)):
				contours.append(self.up4b(self.out_con[ix](dil_feats1[ix])))

			for ix in range(len(dil_feats2)):
				contours.append(self.up3b(self.out_con[len(dil_feats1) + ix](dil_feats2[ix])))

			x = self.up3(x3, x2); logits_small = self.up3b(self.outs(x)); contour_small = self.up3b(self.outcontour2(x))
			
			saliency.append(logits_small)
			contours.append(contour_small)
			
			x = self.up4(x, x1)

			logits = self.up4b(self.outc(x)) + logits_small
			contour_s = self.up4b(self.outcontour1(x)) + contour_small
			
			saliency.append(logits)
			contours.append(contour_s)

			return contours, saliency

		else:

			x = self.up3(self.combd2(x3, self.up3x(x3)), x2); logits_small = self.up3b(self.outs(x))

			saliency.append(logits_small)
			
			x = self.up4(self.combd1(x, self.up4x(x)), x1)
			logits = self.up4b(self.outc(x)) + logits_small

			saliency.append(logits)
			return saliency
