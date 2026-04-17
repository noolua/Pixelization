import torch
import torch.nn as nn
from torch.nn import init
import functools
import numpy as np
import scipy.stats as st
from .c2pGen import *


class Identity(nn.Module):
  def forward(self, x):
    return x


def get_norm_layer(norm_type='instance'):
  if norm_type == 'batch':
    norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
  elif norm_type == 'instance':
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
  elif norm_type == 'none':
    def norm_layer(x): return Identity()
  else:
    raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
  return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
  def init_func(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
      if init_type == 'normal':
        init.normal_(m.weight.data, 0.0, init_gain)
      elif init_type == 'xavier':
        init.xavier_normal_(m.weight.data, gain=init_gain)
      elif init_type == 'kaiming':
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
      elif init_type == 'orthogonal':
        init.orthogonal_(m.weight.data, gain=init_gain)
      else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
      if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
      init.normal_(m.weight.data, 1.0, init_gain)
      init.constant_(m.bias.data, 0.0)

  print('initialize network with %s' % init_type)
  net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
  if len(gpu_ids) > 0:
    if torch.cuda.is_available():
      net.to(gpu_ids[0])
      net = torch.nn.DataParallel(net, gpu_ids)
    elif torch.backends.mps.is_available():
      net.to('mps')
    else:
      net.to('cpu')

  init_weights(net, init_type, init_gain=init_gain)
  return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
  net = None
  norm_layer = get_norm_layer(norm_type=norm)

  if netG == 'c2pGen':
    net = C2PGen(input_nc, output_nc, ngf, 2, 4, 256, 256, activ='relu', pad_type='reflect')
    print('c2pgen resblock is 8')
  elif netG == 'antialias':
    net = AliasNet(input_nc, output_nc, ngf, 2, 3, activ='relu', pad_type='reflect')
  else:
    raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
  return init_net(net, init_type, init_gain, gpu_ids)
