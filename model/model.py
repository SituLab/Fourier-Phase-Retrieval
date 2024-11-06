from turtle import forward
from numpy import zeros
import torch.nn as nn
import torch
import math
import torchvision.transforms as transforms

class realFFT(nn.Module):
    def __init__(self, window_size = 2048):
        super(realFFT, self).__init__()
        self.pi = math.pi
        dxs = 10e-6
        dys = 10e-6
        #采样点数
        M = 2000
        N = M
        Dx = M*dxs
        Dy = N*dys
        z = 5e6
        lambda1 = 632.8e-9
        k = 2*self.pi/lambda1
        # 采样间隔 采样范围
        dxu = lambda1*z/M/dxs
        dyu = lambda1*z/N/dys
        Dxu = lambda1*z/dxs
        Dyu = lambda1*z/dys
        self.fullsize = 2048
        p = torch.linspace(-Dxu/2 , Dxu/2 - dxu, M)
        q = torch.linspace(-Dyu/2 , Dyu/2 - dyu, N)
        pp, qq = torch.meshgrid(p, q, indexing='ij')
        f2 = torch.exp(torch.tensor((1j*k*z)))/1j/lambda1/z
        f1 = torch.exp(1j*k*(pp**2+qq**2)/2/z)
        self.fra = f1*f2
        # self.fra = torch.unsqueeze(self.fra, 0)
        self.window_size = window_size
        self.window = torch.bartlett_window(self.window_size)
        self.window = self.window.unsqueeze(0)
        self.window2 = torch.multiply(self.window, torch.transpose(self.window, 0, 1)).cuda()

    def forward(self, I, P, out_size, window = True):
        """
        :param A P Z L:
        :return: I,P
        :size: batch,channels,M,N
        """
        M = int(I.shape[-2])
        N = int(I.shape[-1])
        # M = int(P.shape[-1])
        P = P.type(torch.complex128).cuda()
        I = I.type(torch.complex128).cuda()
        # P = 2 * self.pi * (P - 0.5)   # (0,1)-(0,2pi)
        P = P * self.pi                 # (0,1)-(0, pi)
        pad = transforms.CenterCrop(self.fullsize)
        I = pad(I)
        P = pad(P)
        self.window2 = pad(self.window2)
        if window:
            U_in = (torch.cos(P) + 1j * torch.sin(P)) * torch.sqrt(I) * self.window2
        else:
            U_in = (torch.cos(P) + 1j * torch.sin(P)) * torch.sqrt(I)
        U_out = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(U_in)))
        # U_out = U_out * self.fra.cuda()
        crop = transforms.CenterCrop(out_size)
        U_out = crop(U_out)
        I1 = U_out.abs() * U_out.abs()
        phi1 = torch.angle(U_out)
        # I1 = torch.log(I1 + 1)
        # I1.clamp_(min = 1e-5)
        outI1 = I1/(torch.max(torch.max(I1)).float())# (0,1)
        # resize = transforms.Resize([M,N])
        # outI = resize(outI1) 
        # outP = phi1.float()/(2 * self.pi) + 0.5 #(-pi,pi)
        # outP = phi1.float()

        return outI1, phi1

class realiFFT(nn.Module):
    def __init__(self):
        super(realiFFT, self).__init__()
        self.pi = math.pi
        dxs = 10e-6
        dys = 10e-6
        #采样点数
        M = 2000
        N = M
        Dx = M*dxs
        Dy = N*dys
        z = 5e6
        lambda1 = 632.8e-9
        k = 2*self.pi/lambda1
        # 采样间隔 采样范围
        dxu = lambda1*z/M/dxs
        dyu = lambda1*z/N/dys
        Dxu = lambda1*z/dxs
        Dyu = lambda1*z/dys

        p = torch.linspace(-Dxu/2 , Dxu/2 - dxu, M)
        q = torch.linspace(-Dyu/2 , Dyu/2 - dyu, N)
        pp, qq = torch.meshgrid(p, q)
        f2 = torch.exp(torch.tensor((1j*k*z)))/1j/lambda1/z
        f1 = torch.exp(1j*k*(pp**2+qq**2)/2/z)
        self.fra = f1*f2
        # self.fra = torch.unsqueeze(self.fra, 0)
        self.fullsize = 2048


    def forward(self, I, P, out_size):
        """
        :param A P Z L:
        :return: 
        """
        M = int(I.shape[-2])
        N = int(I.shape[-1])
        # M = int(P.shape[-1])
        P = P.type(torch.complex128).cuda()
        I = I.type(torch.complex128).cuda()
        # P = 2 * self.pi * (P - 0.5)   # (0,1)-(0,2pi)
        P = P * self.pi                 # (0,1)-(0, pi)
        pad = transforms.CenterCrop(self.fullsize)
        I = pad(I)
        P = pad(P)

        U_in = (torch.cos(P) + 1j * torch.sin(P)) * torch.sqrt(I)               
        U_out = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(U_in)))
        # U_out = U_out * self.fra.cuda()
        crop = transforms.CenterCrop(out_size)
        U_out = crop(U_out)
        I1 = U_out.abs() * U_out.abs()
        phi1 = torch.angle(U_out)
        # I1 = torch.log(I1 + 1)
        # I1.clamp_(min = 1e-5)
        outI1 = I1/(torch.max(torch.max(I1)).float())# (0,1)
        # resize = transforms.Resize([M,N])
        # outI = resize(outI1) 
        # outP = phi1.float()/(2 * self.pi) + 0.5 #(-pi,pi)
        # outP = phi1.float()

        return outI1, phi1

class myFFT(nn.Module):
    def __init__(self):
        super(myFFT, self).__init__()
        self.pi = math.pi
        dxs = 10e-6
        dys = 10e-6
        #采样点数
        M = 2000
        N = M
        Dx = M*dxs
        Dy = N*dys
        z = 5e6
        lambda1 = 632.8e-9
        k = 2*self.pi/lambda1
        # 采样间隔 采样范围
        dxu = lambda1*z/M/dxs
        dyu = lambda1*z/N/dys
        Dxu = lambda1*z/dxs
        Dyu = lambda1*z/dys

        p = torch.linspace(-Dxu/2 , Dxu/2 - dxu, M)
        q = torch.linspace(-Dyu/2 , Dyu/2 - dyu, N)
        pp, qq = torch.meshgrid(p, q)
        f2 = torch.exp(torch.tensor((1j*k*z)))/1j/lambda1/z
        f1 = torch.exp(1j*k*(pp**2+qq**2)/2/z)
        self.fra = f1*f2
        # self.fra = torch.unsqueeze(self.fra, 0)
    def forward(self, I, P):
        """
        :param A P Z L:
        :return: 
        """
        M = int(P.shape[-1])
        P = P.type(torch.complex128).cuda()
        I = I.type(torch.complex128).cuda()
        # P = 2 * self.pi * (P - 0.5)   # (0,1)-(0,2pi)
        P = P * self.pi                 # (0,1)-(0, pi)
        U_in = (torch.cos(P) + 1j * torch.sin(P)) * torch.sqrt(I)
                
        U_out = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(U_in)))
        # U_out = U_out * self.fra.cuda()

        I1 = U_out.abs() * U_out.abs()
        phi1 = torch.angle(U_out)
        I1 = torch.log(I1 + 1)
        outI = I1/torch.max(torch.max(I1)).float()# (0,1)

        # outP = phi1.float()/(2 * self.pi) + 0.5 #(-pi,pi)
        outP = phi1.float()

        return outI, phi1

class ampFFT(nn.Module):
    def __init__(self):
        super(ampFFT, self).__init__()
        self.pi = math.pi
        dxs = 10e-6
        dys = 10e-6
        #采样点数
        M = 2000
        N = M
        Dx = M*dxs
        Dy = N*dys
        z = 5e6
        lambda1 = 632.8e-9
        k = 2*self.pi/lambda1
        # 采样间隔 采样范围
        dxu = lambda1*z/M/dxs
        dyu = lambda1*z/N/dys
        Dxu = lambda1*z/dxs
        Dyu = lambda1*z/dys

        p = torch.linspace(-Dxu/2 , Dxu/2 - dxu, M)
        q = torch.linspace(-Dyu/2 , Dyu/2 - dyu, N)
        pp, qq = torch.meshgrid(p, q)
        f2 = torch.exp(torch.tensor((1j*k*z)))/1j/lambda1/z
        f1 = torch.exp(1j*k*(pp**2+qq**2)/2/z)
        self.fra = f1*f2
        # self.fra = torch.unsqueeze(self.fra, 0)
    def forward(self, I):
        """
        :param A P Z L:
        :return: 
        """
        # M = int(P.shape[-1])
        # P = P.type(torch.complex128).cuda()
        I = I.type(torch.complex128).cuda()
        # P = 2 * self.pi * (P - 0.5)   # (0,1)-(0,2pi)
        # P = P * self.pi                 # (0,1)-(0, pi)
        U_in = torch.sqrt(I)
                
        U_out = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(U_in)))
        # U_out = U_out * self.fra.cuda()

        I1 = U_out.abs() * U_out.abs()
        phi1 = torch.angle(U_out)
        outI = I1/torch.max(torch.max(I1)).float()# (0,1)
        I1 = torch.log(I1 + 1)
        # outI = I1/torch.max(torch.max(I1)).float()# (0,1)

        # outP = phi1.float()/(2 * self.pi) + 0.5 #(-pi,pi)
        outP = phi1.float()

        return outI, outP

class myIFFT(nn.Module):
        def __init__(self):
            super(myIFFT, self).__init__()
            self.pi = math.pi


        def forward(self, I, P):
            """
            :param A P Z L:
            :return: 
            """
            M = int(P.shape[-1])
            P = P.type(torch.complex128).cuda()
            I = I.type(torch.complex128).cuda()
            # P = 2 * self.pi * (P - 0.5)# (0,1)-(0,2pi)
            # P = P * self.pi 
            U_in = (torch.cos(P) + 1j * torch.sin(P)) * torch.sqrt(I)
            
            # x = torch.linspace(-M / 2, M / 2 - 1, M).cuda()
            # Fx, Fy = torch.meshgrid(x, x)
            # r = 0.5 * M
            
            # U_out = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(U_in)))
            # U_out = torch.where(Fx*Fx + Fy*Fy < r*r, U_out, torch.zeros_like(U_out))
            U_out = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(U_in),norm='forward'))

            I1 = U_out.abs() * U_out.abs()
            phi1 = torch.angle(U_out)
            # I1 = torch.log(I1 + 1)
            outI = I1/torch.max(torch.max(I1)).float()# (0,1)

            # outP = phi1.float()/(2 * self.pi) + 0.5 #(-pi,pi)
            outP = phi1.float()

            return outI, outP/ self.pi


if __name__ == '__main__':
    import torchvision.transforms as transforms
    test_data = torch.rand(2, 128, 128).cuda()
    d = realFFT()
    # test_pad=torch.nn.functional.pad(test_data,[436, 436, 436, 436])
    I1, P1 = d(test_data, test_data, 260)
    transform = transforms.CenterCrop(16)
    image_crop = transform(I1)
    # I0, P0 = d(I1, P1)
    print(I1.shape)
    # print(P1)
    