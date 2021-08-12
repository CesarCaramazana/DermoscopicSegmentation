import torch
import torchvision
import torch.nn as nn
from torchvision import models


class unet_inception(torch.nn.Module):
  def __init__(self):
    super().__init__()
    out_channels = 6

    #Downloads the pre-trained weights of an Inception v3 model
    self.inception = models.inception_v3(pretrained=True)

    #ENCODER
    self.l1 = nn.Sequential(*list(self.inception.children())[0:3])
    self.l2 = nn.Sequential(*list(self.inception.children())[3:6])
    self.l3 = nn.Sequential(*list(self.inception.children())[6:10])

    #ASPP layers
    self.a1 = nn.Sequential( #Dilation 2
        nn.Conv2d(in_channels= 192, out_channels=256, kernel_size=3, stride=1, dilation=2, padding=2),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(),
        nn.Conv2d(in_channels= 256, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(),
        nn.Conv2d(in_channels= 256, out_channels=256, kernel_size=3, stride=2, padding=0),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(),
    )
    self.a2 = nn.Sequential( #Dilation 2
        nn.Conv2d(in_channels= 192, out_channels=256, kernel_size=3, stride=1, dilation=3, padding=3),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(),
        nn.Conv2d(in_channels= 256, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(),
        nn.Conv2d(in_channels= 256, out_channels=256, kernel_size=3, stride=2, padding=0),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(),

    )
    self.a3 = nn.Sequential( #Dilation 2
        nn.Conv2d(in_channels= 192, out_channels=256, kernel_size=3, stride=1, dilation=4, padding=4),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(),
        nn.Conv2d(in_channels= 256, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(),
        nn.Conv2d(in_channels= 256, out_channels=256, kernel_size=3, stride=2, padding=0),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(),

    )

    #DECODER
    self.l5 = nn.Sequential(
        nn.ConvTranspose2d(in_channels=1056, out_channels=800, kernel_size=3, stride=2, padding=0, output_padding=1),
        nn.BatchNorm2d(800, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(),
        nn.Conv2d(in_channels= 800, out_channels=512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(),
    )
    self.l6 = nn.Sequential(
        nn.ConvTranspose2d(in_channels=704, out_channels=704, kernel_size=7, stride=2, padding=0, output_padding=0),
        nn.BatchNorm2d(704, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(),
        nn.Dropout2d(p=0.5),
        nn.Conv2d(in_channels= 704, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(),

    )
    self.l7 = nn.Sequential(
        nn.ConvTranspose2d(in_channels=320, out_channels=256, kernel_size=7, stride=2, padding=0, output_padding=1),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(),
        nn.Conv2d(in_channels= 256, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(out_channels),
    )


  def forward(self, input):

    #Encoder--
    l1 = self.l1(input)
    l2 = self.l2(l1)
    l3 = self.l3(l2)
    #Apply Atrous Spatial Pyramid Pooling
    a1 = self.a1(l2)
    a2 = self.a2(l2)
    a3 = self.a3(l2)

    a = torch.cat((a1, a2, a3), dim=1) #Generate a single volume.
    x = torch.cat((a, l3), dim=1) #Add layer 3

    #Decoder--
    x = self.l5(x)
    x = torch.cat((x, l2), dim=1) #Skip connection 1
    x = self.l6(x)
    x = torch.cat((x, l1), dim=1) #Skip connection 2
    x = self.l7(x)

    return x
