import torch
import torch.nn as nn
import torch.nn.functional as F

######
# modified to have shorter temporal, larger spatial kernel
######


class ResidualConv3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=(1,2,2)):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_ch, out_ch,
            kernel_size=(3,3,3),
            stride=stride,
            padding=1
        )

        self.norm1 = nn.InstanceNorm3d(out_ch)

        self.conv2 = nn.Conv3d(
            out_ch, out_ch,
            kernel_size=(1,5,5),
            padding=(0,2,2)
        )

        self.norm2 = nn.InstanceNorm3d(out_ch)

        self.skip = nn.Conv3d(
            in_ch, out_ch,
            kernel_size=1,
            stride=stride
        )

    def forward(self,x):

        identity = self.skip(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        return F.relu(out + identity)

class UpBlock(nn.Module):

    def __init__(self,in_ch,out_ch):
        super().__init__()

        self.up = nn.Sequential(
            nn.Upsample(
                scale_factor=(1, 2, 2),
                mode="trilinear",
                align_corners=False
            ),
            nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=1
            )
        )

        self.conv = ResidualConv3D(
            in_ch,
            out_ch,
            stride=1
        )

    def forward(self,x,skip):

        x = self.up(x)

        # 혹시 1 pixel mismatch 방지
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x,size=skip.shape[-3:],mode="trilinear",align_corners=False)

        x = torch.cat([x,skip],dim=1)

        x = self.conv(x)

        return x

class SpatialSmoothHead(nn.Module):
    def __init__(self,ch):
        super().__init__()
        self.smooth = nn.Sequential(
            nn.Conv3d(
                ch,
                ch,
                kernel_size=(1,5,5),
                padding=(0,2,2),
                groups=ch
            ),
            nn.Conv3d(
                ch,
                ch,
                kernel_size=(1,3,3),
                padding=(0,1,1),
            )
        )
    def forward(self,x):
        return self.smooth(x)

class PDN(nn.Module):

    def __init__(self,in_ch=1,filters=[16,32,64,128]):
        super().__init__()

        # input
        self.input_conv = nn.Conv3d(
            in_ch,
            filters[0],
            kernel_size=(3,3,3),
            padding=1
        )

        # encoder
        self.enc1 = ResidualConv3D(filters[0],filters[1])
        self.enc2 = ResidualConv3D(filters[1],filters[2])

        # bridge
        self.bridge = ResidualConv3D(filters[2],filters[3])

        # decoder
        self.up1 = UpBlock(filters[3],filters[2])
        self.up2 = UpBlock(filters[2],filters[1])
        self.up3 = UpBlock(filters[1],filters[0])

        # spatial smoothing head
        self.smooth = SpatialSmoothHead(filters[0])

        # output
        self.out = nn.Conv3d(filters[0],1,kernel_size=1)

    def forward(self,x):

        inp = x

        x1 = self.input_conv(x)

        x2 = self.enc1(x1)
        x3 = self.enc2(x2)

        x4 = self.bridge(x3)

        x = self.up1(x4,x3)
        x = self.up2(x,x2)
        x = self.up3(x,x1)

        x = self.smooth(x)

        out = self.out(x)

        return inp + out