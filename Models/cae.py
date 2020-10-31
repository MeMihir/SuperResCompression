import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import os
from PIL import Image
import torch
from Models.utils.bitstring import *
import lzma

class SignFunction(Function):
    def __init__(self):
        super(SignFunction,self).__init__()
    @staticmethod
    def forward(ctx,input, is_training=True):
        if is_training:
            prob = input.new(input.size()).uniform_()
            x = input.clone()
            x[(1 - input) / 2 <= prob] = 1
            x[(1 - input) / 2 > prob] = -1
            return x
        else:
            return input.sign()
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
        
class Sign(nn.Module):
    def __init__(self):
        super(Sign, self).__init__()
    def forward(self,x):
        return SignFunction.apply(x, self.training)


class Binarizer(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Binarizer,self).__init__()
        self.sign = Sign()
        self.conv1 = nn.Conv2d(in_channels, out_channels,kernel_size=1,bias=False)
    def forward(self,x):
        x = self.conv1(x)
        x =  F.tanh(x)
        return self.sign(x)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.enc = nn.Sequential(nn.Conv2d(3,32,8,stride=4,padding=2),
                                nn.ReLU(),
                                nn.BatchNorm2d(32),
                                nn.Conv2d(32,64,2,stride=2),
                                nn.ReLU(),
                                nn.BatchNorm2d(64),
    #                             nn.Conv2d(32,128,6,stride=4,padding=1),
    #                             nn.ReLU(),
    #                             nn.BatchNorm2d(128),
    #                             nn.Conv2d(128,128,3,stride=1,padding=1),
    #                             nn.Sigmoid()
                                )
        self.dec = nn.Sequential(nn.ConvTranspose2d(128,32,8,stride=4, padding=2),
                                nn.BatchNorm2d(32),
                                nn.ReLU(),
                                nn.ConvTranspose2d(32,3,2,2),
                                nn.BatchNorm2d(3),
                                nn.ReLU(),
    #                             nn.ConvTranspose2d(64,32,2,2),
    #                             nn.BatchNorm2d(32),
    #                             nn.Conv2d(32,32,3,stride=1,padding=1),
    #                             nn.BatchNorm2d(32),
    #                             nn.ConvTranspose2d(32,3,2,2),
    #                             nn.Sigmoid()
                                )
        self.binarizer = Binarizer(64,128)
    def forward(self,x):
    
        x = self.enc(x)
        x = self.binarizer(x)
    #     print(x.shape)
        x = self.dec(x)
    #     x = (x+1)*255
    #     x.round_()
        return x

class Encoder():
    def __init__(self, path):
        self.model = Autoencoder().float()
        self.model.eval()
        checkpoint = self.load_checkpoint(path)
        self.model.load_state_dict(checkpoint['model_state'])

    def load_checkpoint(self, path):
      checkpoint = torch.load(path, map_location=torch.device('cpu'))
      return checkpoint

    def compress(self,path):
        img = Image.open(path)
        width, height = img.size
        dw = 32 - (width%32)
        dh = 32 - (height%32)
        img = TF.pad(img,(dw,dh,0,0))
        x =  TF.to_tensor(img)
        x = x.unsqueeze(0)
        x = self.model.enc(x)
        x = self.model.binarizer(x)
        
        y = x.cpu().detach().numpy()
        y[y<0] = 0
        y[y>0] = 1
        return y,dw,dh
    def encode_and_save(self, in_path, out_path):
        y,dw,dh = self.compress(in_path)
        comp_dw = BitArray(uint=dw,length=8)
        comp_dh = BitArray(uint=dh,length=8)
        comp_S2 = BitArray(uint=y.shape[2],length = 16)
        comp_S3 = BitArray(uint = y.shape[3],length=16)

        y = y.ravel()
        comp_y = BitArray(y)
        # print(comp_y.bin[:200])
        with lzma.open(out_path , 'wb', preset=9) as fp:
            fp.write(comp_dw.tobytes())
            fp.write(comp_dh.tobytes())
            fp.write(comp_S2.tobytes())
            fp.write(comp_S3.tobytes())
            fp.write(comp_y.tobytes())
        # with ZipFile('%s.zip'%out_path,'w') as zip:
        #     zip.write(out_path)

class Decoder():
    def __init__(self,path):
        self.model = Autoencoder().float()
        # self.model.eval()
        checkpoint = self.load_checkpoint(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

    def load_checkpoint(self, path):
      checkpoint = torch.load(path, map_location=torch.device('cpu'))
      return checkpoint

    def decompress(self, in_path, out_path):
        dw = dh = y = S2 = S3 = None
        

        with lzma.open(in_path, 'rb') as fp:
            dw = int.from_bytes(fp.read(1), byteorder='big', signed=False)
            dh = int.from_bytes(fp.read(1), byteorder='big', signed=False)
            S2 = int.from_bytes(fp.read(2), byteorder='big', signed=False)
            S3 = int.from_bytes(fp.read(2), byteorder='big', signed=False)

            y = np.empty((1,128,S2,S3)).ravel()
            temp = None;
            j = 0

            print('reading matrix')
            byte = fp.read(1)
            while byte != b"":
                temp = BitArray(byte).bin
                # print(temp)
                for i in range(len(temp)):
                    y[j] = int(temp[i])
                    j += 1
                byte =  fp.read(1)
        

        y[y<0.5] = -1
        y = torch.from_numpy(y.reshape(1,128,S2,S3)).float()

        output = self.model.dec(y)
        img = TF.to_pil_image(output.squeeze(0))

        width, height = img.size
        img = img.crop((dw,dh,width,height));
        img.save(out_path, "PNG")
        return y

class CAE():
  def __init__(self, img_dir, comp_dir, decomp_dir):
    self.img_dir = img_dir
    self.comp_dir = comp_dir
    self.decomp_dir = decomp_dir

  def compress(self):
    f = os.listdir(self.img_dir)

    encoder = Encoder('Models/trained/cae.tar')
    for i in f:
      print('converting %s...'%i)
      encoder.encode_and_save(os.path.join(self.img_dir, i), os.path.join(self.comp_dir, '%scomp.xfr'%i[:-4]))
  
  def decompress(self):
    f = os.listdir(self.comp_dir)
    decoder = Decoder('Models/trained/cae.tar')
    for i in f:
      print('converting %s...'%i)
      decoder.decompress(os.path.join(self.comp_dir, i), os.path.join(self.decomp_dir, '%s.png'%i[:-4]))
  


  