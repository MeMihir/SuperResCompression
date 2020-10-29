import torch
import torchvision
from torch import nn , optim
from torchvision import datasets, transforms
from torch.autograd import Variable


from PIL import Image

import math
irange = range

imsize = 256
loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.24703223,  0.24348513 , 0.26158784))
])


CHANNELS = 3
HEIGHT = 256
WIDTH = 256
EPOCHS = 20
LOG_INTERVAL = 500

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x
   

class End_to_end(nn.Module):
  def __init__(self):
    super(End_to_end, self).__init__()
    
    # Encoder
    self.conv1 = nn.Conv2d(CHANNELS, 64, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0)
    self.bn1 = nn.BatchNorm2d(64, affine=False)
    self.conv3 = nn.Conv2d(64, CHANNELS, kernel_size=3, stride=1, padding=1)
    
    # Decoder
    self.interpolate = Interpolate(size=HEIGHT, mode='bilinear')
    self.deconv1 = nn.Conv2d(CHANNELS, 64, 3, stride=1, padding=1)
    self.deconv2 = []
    for _ in range(18):
        self.deconv2.append(nn.Conv2d(64, 64 ,3))
        self.deconv2.append(nn.BatchNorm2d(64))
        self.deconv2.append(nn.ReLU6())
    self.deconv2 = nn.Sequential(*self.deconv2)
    self.deconv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(64, affine=False)
    self.deconv3 = nn.Conv2d(64, 3, 3, stride=1, padding=1)
    
    self.deconv_n = nn.Conv2d(64, 64, 3, stride=1, padding=1)
    self.bn_n = nn.BatchNorm2d(64, affine=False)

    
    self.deconv3 = nn.ConvTranspose2d(64, CHANNELS, 3, stride=1, padding=1)
    
    
    self.relu = nn.ReLU()
  
  def encode(self, x):
    out = self.relu(self.conv1(x))
    out = self.relu(self.conv2(out))
    out = self.bn1(out)
    return self.conv3(out)
    
  
  def reparameterize(self, mu, logvar):
    pass
  
  def decode(self, z):
    upscaled_image = self.interpolate(z)
    out = self.relu(self.deconv1(upscaled_image))
    out = self.deconv2(out)
    out = self.deconv3(out)
    # for _ in range(10):
    #   out = self.relu(self.deconv_n(out))
    #   out = self.bn_n(out)
    # out = self.deconv3(out)
    final = upscaled_image + out
    return final,out,upscaled_image

    
  def forward(self, x):
    com_img = self.encode(x)
    final,out,upscaled_image = self.decode(com_img)
    return final, out, upscaled_image, com_img, x

model = End_to_end()
model1 = torch.load('./net.pth', map_location=torch.device('cpu'))
model.load_state_dict(model1, strict=False)
model.eval()

# ! wget https://github.com/AireshBhat/end_to_end_compression_cnn/raw/2cf00a4ce2996293e26bc9d52031369a6510915c/images/Peppers.jpg

test_loss = 0
for i, (data, _) in enumerate(test_loader):
      data = Variable(data, volatile=True)
      imageName = 'Peppers'
      image = image_loader('./'+ imageName + '.jpg')

      # image = make_grid(image)
      final, residual_img, upscaled_image, com_img, orig_im = model(image)
      # if torch.cuda.is_available():
      #     test_loss += loss_function(final, residual_img, upscaled_image, com_img, orig_im).image.cuda()
      # else:
      #     test_loss += loss_function(final, residual_img, upscaled_image, com_img, orig_im).image
          

      n = min(image.size(0), 6)
      print(n);
      comparison = torch.cat([image[:n],
          final[:n].cpu()])
      comparison = comparison.cpu()
#             print(comparison.data)
      save_image(com_img[:n],
                  './' + imageName + 'Compressed_' + '.jpeg', nrow=n)
      save_image(residual_img[:n],
                  './' + imageName + 'Residual_' + '.jpeg', nrow=n)
      save_image(upscaled_image[:n],
                  './' + imageName + 'Upscaled_' + '.jpeg', nrow=n)
      save_image(final[:n],
                  './' + imageName + 'Final_' + '.jpeg', nrow=n)
      print("saving the image "+str(n))

test_loss /= len(test_loader.dataset)
print('====> Test set loss: {:.4f}'.format(test_loss))