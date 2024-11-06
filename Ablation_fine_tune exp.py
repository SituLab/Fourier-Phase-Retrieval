import os
import sys
import time
from collections.abc import Iterable

import cv2
import hiddenlayer as hl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from numpy.core.numeric import ones_like
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch import nn
from torch.autograd import Variable

import torchvision.transforms.functional
# from ffdnet.models import FFDNet
# from ffdnet.test_ffdnet_ipol import ffdnet_vdenoiser
from loss import PSNR, SSIM
from model.model import realFFT, realiFFT
# from SwinIR.SwinIR.net import SwinIR
from model.unet import UNet
sys.path.append('..')

def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze
            
def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)

def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)

def set_freeze_by_idxs(model, idxs, freeze=True):
    if not isinstance(idxs, Iterable):
        idxs = [idxs]
    num_child = len(list(model.children()))
    idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, idxs))
    for idx, child in enumerate(model.children()):
        if idx not in idxs:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze
            
def freeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, True)

def unfreeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, False)

def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

def readimg2net(name, out_size=300):
    diffraction_I = cv2.imread(name, -1)
    diffraction_I = diffraction_I.astype(np.float32)
    diffraction_I = diffraction_I / np.max(diffraction_I)
    (nh, nw) = np.unravel_index(diffraction_I.argmax(), diffraction_I.shape)
    diffraction_I = diffraction_I[nh - out_size//2:nh + out_size//2, nw - out_size//2:nw + out_size//2]
    diffraction_I = torch.from_numpy(diffraction_I)
    diffraction_I = diffraction_I.unsqueeze(0).unsqueeze(0)
    return diffraction_I

def readimg2amp(name, dim=(128,128),out_size=300):
    transform = transforms.Resize(dim)
    amp = Image.open(name)
    amp = np.array(amp.convert('L'))
    amp = amp.astype(np.float32)
    amp = amp / np.max(amp)
    amp = cv2.resize(amp, dim)
    amp = torch.from_numpy(amp).cuda()
    H = realFFT()
    diffraction_I, _ = H(amp, torch.ones_like(amp), out_size)
    diffraction_I = diffraction_I.unsqueeze(0).unsqueeze(0)
    amp = amp.unsqueeze(0).unsqueeze(0)
    return diffraction_I, amp

def LF_NNPsupport(g1, percent):
    ##g1 is net output: sigmoid
    gs, indices=torch.sort(g1.reshape(-1))
    m=g1.size(-2)
    n=g1.size(-1)
    thre=gs[int(np.round(m*n*(1-percent)))]
    S = (g1 >= thre)
    Num=torch.sum(S!=0)
    AVRG=torch.sum(S*g1)/Num
    g2=g1-0.4*g1*(g1>(4*AVRG))  # too large is bad
    return S,g1

def addrun(valid_folder, dp=-2):
    while os.path.exists(valid_folder):
        valid_folder_dirs = valid_folder.split(os.path.sep)
        last_digit_start = None
        for i in range(len(valid_folder_dirs[dp]) - 1, -1, -1):
            if valid_folder_dirs[dp][i].isdigit():
                last_digit_start = i
            else:
                break
        if last_digit_start is not None:
        # 提取最后几个数字的部分
            last_digits = valid_folder_dirs[dp][last_digit_start:]
            # 将提取的数字转换为整数
            num = int(last_digits)
            # 将整数加1
            num += 1
            # 构建新的字符串
            valid_folder_dirs[dp] = valid_folder_dirs[dp][:last_digit_start] + str(num)
            valid_folder = os.path.join(*valid_folder_dirs)
        else:
            print("字符串中没有数字o.0?")
    return valid_folder

def findbestposition(O, I_pred, mse) :
    # 定义搜索窗口大小和步长
    window_size = 10  # 搜索窗口大小
    stride = 1  # 步长
    # 初始化最小差异和对应的平移量
    min_diff = float('inf')
    best_translation = (0, 0)
    best_translated_I_pred = I_pred
    # 遍历搜索窗口
    for flip in [-1 -2 -1]:
        for i in range(-window_size, window_size, stride):
            for j in range(-window_size, window_size, stride):
                # 翻转
                translated_I_pred = torch.flip(I_pred, dims=[flip])
                # 平移图像
                translated_I_pred = transforms.functional.affine(translated_I_pred, translate = (i, j), angle=0, scale=1, shear=0) # type: ignore

                # 计算均方误差
                diff = mse(O, translated_I_pred)

                # 如果找到更小的差异，更新最小差异和平移量
                if diff.item() < min_diff:
                    min_diff = diff.item()
                    best_translation = (i, j)
                    best_translated_I_pred = translated_I_pred

    

    return best_translated_I_pred, min_diff  


def main(object= '4' , net =  'swinir', pretrain = True, ffd = True,
          lr = 0.001, epochs = 1000, lr_reduce = False, fft_size=300, dim=(128,128), NNP=0.2,
          model_num = '1', pretrainsave=False, translate=True, window= True,
          optimizermode='adam', NNPTag = False, freq=10):
    # 1 prepare data
    # 得到输入数据
    # 仿真模式、实验模式 
    dirnote = object +'_'+ net +'_'+ model_num +'_lr='+ str(lr)
    
    # 1.parameters
    namenote = object +'_NNP='+str(NNP)+'_NNPTag'+str(NNPTag)
    if pretrain:
        namenote = namenote + '_pretrain_'
    if ffd:
        namenote = namenote + '_ffd_'
    if window:
        namenote = namenote + '_window_'
    


    # 2.objects 
    O = None
    if object == '-':
        I = readimg2net('realdata/3.28_-_1_000010.tif')
    elif object == '4':
        I = readimg2net('realdata/3.28_4_1_000010.tif')
    elif object == '5':
        I = readimg2net('realdata/3.28_5_1_000010.tif')
    elif object == '6':
        I = readimg2net('realdata/3.28_6_1_000010.tif')
    elif object == 'complex67':
        I = readimg2net('realdata/3.28_c67_1_000010.tif')
    elif object == 'III':
        I, O =  readimg2amp('ampdata/seed2.png')
    elif object == 'pepper':
        I, O =  readimg2amp('data/pepper256.bmp')
    elif object == 'celeba_B1':
        I, O =  readimg2amp('data/test_celeba_29524_p.png')
    elif object == 'mnist_A1':
        I, O =  readimg2amp('data/train_MNIST_128/1.png')
    elif object == 'mnist_B1':
        I, O =  readimg2amp('data/test_MNIST_128/49751.png')
    elif object == 'emnist_A1':
        I, O =  readimg2amp('data/train_EMNISTbinary/1.bmp')
    elif object == 'fashion_mnist_B1':
        I, O =  readimg2amp('fashion_MNIST_data/6real.png')
    elif object == 'fashion_mnist_B2':
        I, O =  readimg2amp('fashion_MNIST_data/0real.png')
    elif object == 'fashion_mnist_B3':
        I, O =  readimg2amp('fashion_MNIST_data/36real.png')
    else:
        I = readimg2net('realdata/3.28_4_1_000010.tif')

    I = I.cuda()


    print("start train")
    # 2 model
    H = realFFT()
    IH = realiFFT()
    if model_num == '1':
        model_name1 = './unet/checkpoint/checkpoint_lr:0.010000_epochs:150_v2.pth.tar'  #最新的网络 loss = 0.004830, valid = 0.011914
        model_name2 = './swinir/mnist/checkpoint/checkpoint_lr:0.001000_epochs:40_valid:0.041601.pth.tar' #最新的网络
    elif model_num == '2':
        model_name1 = './models/checkpoint/checkpoint_Unet_MNIST_v2.pth.tar'  #0.013
        model_name2 = './swinir/mnist/checkpoint/checkpoint_lr:0.001000_epochs:40_valid:0.038670.pth.tar'  #次新的网络 稳定版 loss = 0.007621, valid = 0.010876
    elif model_num == '3':
        model_name1 = './models/checkpoint/checkpoint_Unet_MNIST_v3.pth.tar' #0.009
        model_name2 = './fashion-models/swinir/checkpoint_swinlrV0.0108.2.pth.tar.pth.tar'
    elif model_num == '5':
        model_name1 = './models/checkpoint/checkpoint_Unet_MNIST_v5window.pth.tar' #0.012
        model_name2 = './fashion-models/swinir/checkpoint_lr:0.001000_epochs:60_v4v.pth.tar'
    else:
        model_name1 = './unet/checkpoint/checkpoint_lr:0.010000_epochs:150_v2.pth.tar'  #最新的网络 loss = 0.004830, valid = 0.011914
        model_name2 = './fashion-models/swinir/checkpoint_lr:0.001000_epochs:61_v4v.pth.tar' #最新的网络



    if net == 'unet' :
        net = UNet(in_channels=1).cuda()
        if pretrain:
            # net.load_state_dict(torch.load('./models/checkpoint/checkpoint_Unet_MNIST_v2.pth.tar'))   
            net.load_state_dict(torch.load(model_name1))
            freeze_by_names(net, ('d1', 'd2', 'd3', 'd4'))
    elif net == 'swinir':
        net = SwinIR(img_size=128, patch_size=3, in_chans=1,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=1, img_range=1., upsampler='', resi_connection='1conv').cuda()

        if pretrain:   
            net.load_state_dict(torch.load(model_name2))
    else:
        net = UNet(in_channels=1).cuda()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net = net.to(device)   

    if ffd == True:
        ffdnet = FFDNet(num_input_channels=1).cuda()
        model_fn = './ffdnet/models/net_gray.pth'
        state_dict = torch.load(model_fn)
        ffdnet = torch.nn.parallel.DataParallel(ffdnet, device_ids=[0]).cuda()
        ffdnet.load_state_dict(state_dict)
    else:
        ffdnet = None

    mse = nn.MSELoss().cuda()
    SSIM0 = SSIM().cuda()
    PSNR0 = PSNR().cuda()
    # 3 construct loss and optimizer
    if optimizermode == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.8, 0.999))
    elif optimizermode == 'adamw':
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.8, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           mode='min', 
                                                           factor=0.2, 
                                                           patience=10,
                                                           cooldown=50,
                                                           verbose=True,
                                                           threshold=0.001)
    best_vmse = float('inf')
    best_epoch = float('inf')
    # 记录训练过程的指标 A History object to store metrics
    history = hl.History()
    # 使用canvas进行可视化 A Canvas object to draw the metrics
    canvas = hl.Canvas()
    # 目录
    mkdir('./ablation/'+dirnote+'/loss/')
    losstxt = ('./ablation/'+dirnote+'/loss/'+namenote+'.txt')
    valid_folder = './ablation/'+dirnote+'/out/'+namenote+'run_0/'
    result_folder = './ablation/'+dirnote+'/result/'
    mkdir(result_folder)
    valid_folder = addrun(valid_folder)
    mkdir(valid_folder)
    torchname = 'best_'+namenote+'.pth'


    # I = I.to(device=device, dtype=torch.float32)
    transform = transforms.Resize(dim)

    I_in = transform(I.cuda())
    I_in = I_in.to(device=device, dtype=torch.float32)
    net.train()

    
    for epoch in range(epochs):
        # iter mode
        optimizer.zero_grad()

        I_pred = net(I_in.cuda())

        if not NNP == 0:
            S, g1 = LF_NNPsupport(I_pred, NNP)
            if NNPTag:
                I1_test, P1 = H(g1, torch.ones_like(g1), fft_size)  
                g1_tag, _ = IH(I1_test, P1, 128)
                I_pred=g1*(g1_tag>=0)*S
            else:
                I_pred=g1*S

        # if ffd:
        #     I_pred = ffdnet_vdenoiser(I_pred.cuda(), 5/255, ffdnet)
        # else:
        #     I_pred = ffdnet_vdenoiser(I_pred.cuda(), 0/255, ffdnet)


        I1, P1 = H(I_pred.cuda(), torch.ones_like(I_pred).cuda(), fft_size, window = window)
        loss = mse(I1.cuda(), I.double())
        if O is not None:
            if translate:
                translated_I_pred, vmse = findbestposition(O.cuda(), I_pred.cuda(), mse)
            else:
                translated_I_pred = I_pred.cuda()
                vmse = mse(O.cuda(), translated_I_pred.cuda())
            psnr = PSNR0(O.cuda(), translated_I_pred)
            ssim = SSIM0(O.cuda(), translated_I_pred)
            

            if vmse < best_vmse:
                best_vmse = vmse # type: ignore
                best_SSIM = ssim
                best_PSNR = psnr
                best_epoch = epoch
                I_pred_best = I_pred
                translated_I_pred_best = translated_I_pred
                resultname = namenote +'VMSE=%.4g,SSIM=%.4g,PSNR=%.4g,epoch=%d'%(best_vmse,best_SSIM,best_PSNR,best_epoch)+'.png'
                # 保存模型参数 结果
                torch.save(net.state_dict(), valid_folder + torchname)

            with open(losstxt,"a") as f:
                print('iteration:epoch=>%d, loss=%.3g, VMSE=%.4g, SSIM=%.4g, PSNR=%.4g, best_epoch=%d, best_VMSE=%.3g, best_SSIM=%.4g, best_PSNR=%.4g' 
                    % (epoch + 1, loss.item(), vmse, ssim, psnr, best_epoch, best_vmse, best_SSIM, best_PSNR), file=f)# type: ignore  
            if pretrainsave and epoch == 0:
                torchvision.utils.save_image(I_pred, result_folder+namenote+'VMSE=%.4g,SSIM=%.4g,PSNR=%.4g,epoch%d.png'%(vmse,ssim.item(),psnr.item(),epoch))
             
        else:
            if loss < best_vmse:
                best_vmse = loss
                best_epoch = epoch
                I_pred_best = I_pred
                resultname = namenote +'epoch=%d'%(best_epoch)+ '.png'
                # 保存模型参数 结果
                torch.save(net.state_dict(), valid_folder + torchname)

            with open(losstxt,"a") as f:
                print('iteration:epoch=>%d, loss=%.3g, best_epoch=%d, best_VMSE=%.3g' 
                    % (epoch + 1, loss.item(), best_epoch, best_vmse), file=f) # redirect 重定向
            if pretrainsave and epoch == 0:
                torchvision.utils.save_image(I_pred, result_folder+namenote+'epoch%d.png'%(epoch))
        # 更新参数
        loss.backward()
        optimizer.step()
        if lr_reduce:
            scheduler.step(best_vmse)
        # f = open('./evaluation/loss/'+note+'.txt','a')

        if epoch % 10 == 0:
            history.log(epoch + 1,
                        train_loss=best_vmse,
                        # real_loss=valid,
                        # psnr=psnr,
                        # ssim=ssim,                        
                        out=I_pred.detach().cpu().squeeze().squeeze().numpy(),
                        Fourier=I1.detach().cpu().squeeze().squeeze().numpy())
                        #experiment=O.detach().cpu().squeeze().squeeze().numpy())# type: ignore
                        # truth=P_real.detach().cpu().squeeze().squeeze().numpy())
            # Print progress status
            history.progress()
            # Less occasionally, save a snapshot of the graphs
            # Plot the two metrics in one graph
            # start = time.time()  
            with canvas:
                # canvas.draw_plot(history["psnr"], ylabel='psnr')
                # canvas.draw_plot(history["ssim"], ylabel='ssim')
                canvas.draw_plot(history["train_loss"], ylabel='loss')
                canvas.draw_image(history["out"], limit=5)
                canvas.draw_image(history["Fourier"], limit=5)
                # canvas.draw_image(history["experiment"], limit=5)
                # canvas.draw_image(history["truth"], limit=1, cmap='gray', p = 224, titletime = 'GT')       
                       
        
        if epoch % freq == 0 :#and epoch > 800:            
            torchvision.utils.save_image(I_pred,valid_folder+str(epoch)+'.png')
        if epoch % 5 == 0 :    
            if O is not None:
                print('iteration:epoch=>%d, loss=%.3g, vmse=%.4g, SSIM=%.4g, PSNR=%.4g, best_epoch=%d, best_VMSE=%.3g, best_SSIM=%.4g, best_PSNR=%.4g' 
                        % (epoch + 1, loss.item(), vmse, ssim, psnr, best_epoch, best_vmse, best_SSIM, best_PSNR))# type: ignore
            
    torchvision.utils.save_image(I_pred_best, result_folder+resultname)  # type: ignore 
    torchvision.utils.save_image(I_pred_best, valid_folder+resultname)  # type: ignore
    # torchvision.utils.save_image(translated_I_pred_best, result_folder+'translated_'+resultname)  # type: ignore  
    # torchvision.utils.save_image(translated_I_pred_best, valid_folder+'translated_'+resultname)  # type: ignore  




if __name__ == '__main__':

    # for i in range (0,10,1):
    #     main(object= 'III' , net = 'unet', model_num = '2',optimizermode='adam',
    #         pretrain = False, ffd = True, lr = 0.01, lr_reduce = False, epochs=200,
    #         NNP=0.5, NNPTag = True, freq=10, pretrainsave=True) 
        
    # for i in range (0,10,1):
    #     main(object= 'III' , net = 'unet', model_num = '1',optimizermode='adam',
    #         pretrain = True, ffd = True, lr = 0.1, lr_reduce = False, epochs=200,
    #         NNP=0.5, NNPTag = True, freq=10, pretrainsave=True) 
    # for i in range (0,5,1):    
    #     main(object= 'fashion_mnist_B1' , net = 'unet', model_num = '1',optimizermode='adam',
    #             pretrain = True, ffd = True, lr = 0.1, lr_reduce = False, epochs=100,
    #             NNP=0.5, NNPTag = False, freq=10, pretrainsave=True,translate=False) 
    #     main(object= 'fashion_mnist_B1' , net = 'unet', model_num = '1',optimizermode='adam',
    #             pretrain = True, ffd = True, lr = 0.01, lr_reduce = False, epochs=200,
    #             NNP=0.5, NNPTag = False, freq=10, pretrainsave=True,translate=False)
    #     main(object= 'fashion_mnist_B1' , net = 'unet', model_num = '1',optimizermode='adam',
    #             pretrain = True, ffd = True, lr = 0.001, lr_reduce = False, epochs=500,
    #             NNP=0.5, NNPTag = False, freq=10, pretrainsave=True,translate=False) 
    #     main(object= 'fashion_mnist_B1' , net = 'unet', model_num = '1',optimizermode='adam',
    #             pretrain = True, ffd = True, lr = 0.0001, lr_reduce = False, epochs=800,
    #             NNP=0.5, NNPTag = False, freq=10, pretrainsave=True,translate=False) 

    main(object= '6' , net = 'unet', model_num = '5', optimizermode='adam',
         pretrain = True, ffd = False, lr = 0.0001, lr_reduce = False, epochs=200,
         NNP=0, NNPTag = False, freq=10, pretrainsave=True, translate=True, window= True)
    print('Done!')
