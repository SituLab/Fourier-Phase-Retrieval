import json
import os

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from model.model import myFFT
from model.model import realFFT
import matplotlib.pyplot as plt

def convert_image(img, source, target):
    """
    Convert an image from a source format to a target format.

    :param img: image
    :param source: source format, one of 'pil' (PIL image), '[0, 1]' or '[-1, 1]' (pixel value ranges)
    :param target: target format, one of 'pil' (PIL image), '[0, 255]', '[0, 1]', '[-1, 1]' (pixel value ranges),
                   'imagenet-norm' (pixel values standardized by imagenet mean and std.),
                   'y-channel' (luminance channel Y in the YCbCr color format, used to calculate PSNR and SSIM)
    :return: converted image
    """
    assert source in {
        'pil', '[0, 1]', '[-1, 1]'}, "Cannot convert from source format %s!" % source
    assert target in {'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm',
                      'y-channel'}, "Cannot convert to target format %s!" % target

    # Convert from source to [0, 1]
    if source == 'pil':
        img = FT.to_tensor(img)

    elif source == '[0, 1]':
        pass  # already in [0, 1]

    elif source == '[-1, 1]':
        img = (img + 1.) / 2.

    return img


class ImageTransforms(object):
    """
    Image transformation pipeline.
    """

    def __init__(self, split, crop_size, scaling_factor, diff_I_img_type, phase_img_type):
        """
        :param split: one of 'train' or 'test'
        :param crop_size: crop size of HR images
        :param scaling_factor: LR images will be downsampled from the HR images by this factor
        :param lr_img_type: the target format for the LR image; see convert_image() above for available formats
        :param hr_img_type: the target format for the HR image; see convert_image() above for available formats
        """
        self.split = split.lower()
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.lr_img_type = diff_I_img_type
        self.hr_img_type = phase_img_type

        assert self.split in {'train', 'test'}

    def __call__(self, img):
        """
        :param img: a PIL source image from which the HR image will be cropped, and then downsampled to create the LR image
        :return: LR and HR images in the specified format
        """

        # Crop
        if self.split == 'train':
            # Take a random fixed-size crop of the image, which will serve as the high-resolution (HR) image
            # left = random.randint(1, img.width - self.crop_size)
            # top = random.randint(1, img.height - self.crop_size)
            # right = left + self.crop_size
            # bottom = top + self.crop_size
            # hr_img = img.crop((left, top, right, bottom))
            crop_it = transforms.Resize(self.crop_size)
            phase_img = crop_it(img)
        else:
            # Take the largest possible center-crop of it such that its dimensions are perfectly divisible by the scaling factor
            x_remainder = img.width % self.scaling_factor
            y_remainder = img.height % self.scaling_factor
            left = x_remainder // 2
            top = y_remainder // 2
            right = left + (img.width - x_remainder)
            bottom = top + (img.height - y_remainder)
            phase_img = img.crop((left, top, right, bottom))

        # Downsize this crop to obtain a low-resolution version of it
        # lr_img = hr_img.resize((int(hr_img.width / self.scaling_factor), int(hr_img.height / self.scaling_factor)),
        #                        Image.BICUBIC)
        # Sanity check
        # assert hr_img.width == lr_img.width * self.scaling_factor and hr_img.height == lr_img.height * self.scaling_factor

        #############################################################
        # imposing forward propagation model to obtain {lr_img = H(hr_im) + noise} for end-to-end learning
        # lr_img = hr_im for Auto-encoder
        # transforms.Grayscale()

        #############################################################

        # Convert the LR and HR image to the required type

        phase_img = convert_image(
            phase_img, source='pil', target=self.hr_img_type)

        return phase_img


class FFT_Dataset(Dataset):
    """
    A PyTorch Dataset to be used by a PyTorch DataLoader.
    """

    def __init__(self, data_folder, crop_size, split='train', scaling_factor=1, 
                 diff_I_img_type='[0, 1]', phase_img_type='[0, 1]',
                 train_data_name=None, test_data_name=None):
        """
        :param data_folder: # folder with JSON data files
        :param split: one of 'train' or 'test'
        :param crop_size: crop size of target HR images
        :param scaling_factor: the input LR images will be downsampled from the target HR images by this factor; the scaling done in the super-resolution
        :param lr_img_type: the format for the LR image supplied to the model; see convert_image() in utils.py for available formats
        :param hr_img_type: the format for the HR image supplied to the model; see convert_image() in utils.py for available formats
        :param test_data_name: if this is the 'test' split, which test dataset? (for example, "Set14")
        """

        self.data_folder = data_folder
        self.split = split.lower()
        self.crop_size = int(crop_size)
        self.scaling_factor = int(scaling_factor)
        self.diff_I_img_type = diff_I_img_type
        self.phase_img_type = phase_img_type
        self.train_data_name = train_data_name
        self.test_data_name = test_data_name

        assert self.split in {'train', 'test'}
        if self.split == 'test' and self.test_data_name is None:
            raise ValueError("Please provide the name of the test dataset!")
        assert diff_I_img_type in {'[0, 255]',
                                   '[0, 1]', '[-1, 1]', 'imagenet-norm'}
        assert phase_img_type in {'[0, 255]',
                                  '[0, 1]', '[-1, 1]', 'imagenet-norm'}

        # Read list of image-paths
        if self.split == 'train':
            with open(os.path.join(data_folder + self.train_data_name + '_images.json'), 'r') as j:
                self.images = json.load(j)
        else:
            with open(os.path.join(data_folder + self.test_data_name + '_images.json'), 'r') as j:
                self.images = json.load(j)

        # Select the correct set of transforms
        self.transform = ImageTransforms(split=self.split,
                                         crop_size=self.crop_size,
                                         scaling_factor=self.scaling_factor,
                                         diff_I_img_type=self.diff_I_img_type,
                                         phase_img_type=self.phase_img_type)
        self.support = torch.ones(128, 128)
        for i in range(128):
            for j in range(128):
                if (i - 64)**2 + (j - 64)**2 > 30**2:
                    self.support[i, j] = 0

    def __getitem__(self, i):
        """
        This method is required to be defined for use in the PyTorch DataLoader.

        :param i: index to retrieve
        :return: the 'i'th pair LR and HR images to be fed into the model
        """
        # Read image
        img = Image.open(self.images[i], mode='r')
        img = img.convert('L')
        amp_img = self.transform(img)
        H = myFFT()
        # phase_img = phase_img * self.support
        diff_I_img, __ = H(amp_img, torch.ones_like(amp_img))

        return diff_I_img, amp_img

    def __len__(self):
        """
        This method is required to be defined for use in the PyTorch DataLoader.

        :return: size of this data (in number of images)
        """
        return len(self.images)

class resolution_dataset(Dataset):
    """
    A PyTorch Dataset to be used by a PyTorch DataLoader.
    """

    def __init__(self, data_folder, crop_size, fourier_size):
        """
        :param data_folder: # folder with JSON data files
        :param split: one of 'train' or 'test'
        :param crop_size: crop size of target HR images
        :param scaling_factor: the input LR images will be downsampled from the target HR images by this factor; the scaling done in the super-resolution
        :param lr_img_type: the format for the LR image supplied to the model; see convert_image() in utils.py for available formats
        :param hr_img_type: the format for the HR image supplied to the model; see convert_image() in utils.py for available formats
        :param test_data_name: if this is the 'test' split, which test dataset? (for example, "Set14")
        """
        super(resolution_dataset).__init__()
        self.data_folder = data_folder
        self.crop_size = int(crop_size)
        self.fourier_size = fourier_size

        self.img_path_list = os.listdir(self.data_folder)

        self.crop = transforms.CenterCrop(self.crop_size)
        self.transform = transforms.Resize(self.crop_size)

    def __getitem__(self, i):
        """
        This method is required to be defined for use in the PyTorch DataLoader.

        :param i: index to retrieve
        :return: the 'i'th pair LR and HR images to be fed into the model
        """
        # Read image
        img_name = self.img_path_list[i]  # 只获取了文件名
        img_item_path = os.path.join(self.data_folder, img_name) # 每个图片的位置
        img = Image.open(img_item_path, mode='r')
        img = img.convert('L')
        amp_img = FT.to_tensor(img)
        amp_img = self.crop(amp_img)

        H = realFFT()
        # phase_img = phase_img * self.support
        #noise = torch.randn_like(amp_img) * ((0.001*torch.var(amp_img))**0.5)
        diff_I_img, __ = H(amp_img , torch.ones_like(amp_img), out_size=self.fourier_size)
        # if nan in diff_I_img:
        #     diff_I_img = amp_img
        if torch.isnan(diff_I_img).sum() != 0:
            print(self.img_path_list[i])
            os.remove(img_item_path)
            print("Delete File: " + self.img_path_list[i])  
    

        return diff_I_img, amp_img

    def __len__(self):
        """
        This method is required to be defined for use in the PyTorch DataLoader.

        :return: size of this data (in number of images)
        """
        return len(self.img_path_list)

class mnist_dataset(Dataset):
    """
    A PyTorch Dataset to be used by a PyTorch DataLoader.
    """

    def __init__(self, data_folder, crop_size, fourier_size=800):
        """
        :param data_folder: # folder with JSON data files
        :param split: one of 'train' or 'test'
        :param crop_size: crop size of target HR images
        :param scaling_factor: the input LR images will be downsampled from the target HR images by this factor; the scaling done in the super-resolution
        :param lr_img_type: the format for the LR image supplied to the model; see convert_image() in utils.py for available formats
        :param hr_img_type: the format for the HR image supplied to the model; see convert_image() in utils.py for available formats
        :param test_data_name: if this is the 'test' split, which test dataset? (for example, "Set14")
        """
        super(mnist_dataset).__init__()
        self.data_folder = data_folder
        self.crop_size = int(crop_size)
        self.fourier_size = fourier_size

        self.img_path_list = os.listdir(self.data_folder)

        self.crop = transforms.CenterCrop(self.crop_size)
        self.transform = transforms.Resize(self.crop_size)


    def __getitem__(self, i):
        """
        This method is required to be defined for use in the PyTorch DataLoader.

        :param i: index to retrieve
        :return: the 'i'th pair LR and HR images to be fed into the model
        """
        # Read image
        img_name = self.img_path_list[i]  # 只获取了文件名
        img_item_path = os.path.join(self.data_folder, img_name) # 每个图片的位置
        img = Image.open(img_item_path, mode='r')
        img = img.convert('L')
        amp_img = FT.to_tensor(img)
        amp_img = self.crop(amp_img)

        H = realFFT()
        # phase_img = phase_img * self.support
        #noise = torch.randn_like(amp_img) * ((0.001*torch.var(amp_img))**0.5)
        diff_I_img, __ = H(amp_img , torch.ones_like(amp_img), out_size=self.fourier_size)
        # if nan in diff_I_img:
        #     diff_I_img = amp_img
        if torch.isnan(diff_I_img).sum() != 0:
            print(self.img_path_list[i])
            os.remove(img_item_path)
            print("Delete File: " + self.img_path_list[i])        
        diff_I_img = self.transform(diff_I_img)          
        return diff_I_img, amp_img

    def __len__(self):
        """
        This method is required to be defined for use in the PyTorch DataLoader.

        :return: size of this data (in number of images)
        """
        return len(self.img_path_list)

class emnist_dataset(Dataset):
    """
    A PyTorch Dataset to be used by a PyTorch DataLoader.
    """

    def __init__(self, data_folder, crop_size, fourier_size=300):
        """
        :param data_folder: # folder with JSON data files
        :param split: one of 'train' or 'test'
        :param crop_size: crop size of target HR images
        :param scaling_factor: the input LR images will be downsampled from the target HR images by this factor; the scaling done in the super-resolution
        :param lr_img_type: the format for the LR image supplied to the model; see convert_image() in utils.py for available formats
        :param hr_img_type: the format for the HR image supplied to the model; see convert_image() in utils.py for available formats
        :param test_data_name: if this is the 'test' split, which test dataset? (for example, "Set14")
        """
        super(emnist_dataset).__init__()
        self.data_folder = data_folder
        self.crop_size = int(crop_size)
        self.fourier_size = fourier_size

        self.img_path_list = os.listdir(self.data_folder)

        self.crop = transforms.CenterCrop(self.crop_size)
        self.transform = transforms.Resize(self.crop_size)


    def __getitem__(self, i):
        """
        This method is required to be defined for use in the PyTorch DataLoader.

        :param i: index to retrieve
        :return: the 'i'th pair LR and HR images to be fed into the model
        """
        # Read image
        img_name = self.img_path_list[i]  # 只获取了文件名
        img_item_path = os.path.join(self.data_folder, img_name) # 每个图片的位置
        img = Image.open(img_item_path, mode='r')
        img = img.convert('L')
        amp_img = FT.to_tensor(img)
        amp_img = self.transform(amp_img)

        H = realFFT()
        # phase_img = phase_img * self.support
        #noise = torch.randn_like(amp_img) * ((0.001*torch.var(amp_img))**0.5)
        diff_I_img, __ = H(amp_img , torch.ones_like(amp_img), out_size=self.fourier_size)
        # if nan in diff_I_img:
        #     diff_I_img = amp_img
        if torch.isnan(diff_I_img).sum() != 0:
            print(self.img_path_list[i])
            os.remove(img_item_path)
            print("Delete File: " + self.img_path_list[i])        
        diff_I_img = self.transform(diff_I_img)          
        return diff_I_img, amp_img

    def __len__(self):
        """
        This method is required to be defined for use in the PyTorch DataLoader.

        :return: size of this data (in number of images)
        """
        return len(self.img_path_list)


if __name__ == '__main__':
    data_folder = './data/train_EMNISTbinary/'  # folder with JSON data files
    dataset = 'MNIST'
    model_name = 'Unet'
    train_data_name = 'train_' + dataset + '_128'
    test_data_name = 'test_' + dataset + '_128'
    crop_size = 128          # crop size of target HR images
    scaling_factor = 1       # the scaling factor for the generator; the input LR images will be downsampled from the target HR images by this factor

    train_dataset = emnist_dataset(data_folder, crop_size=128)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle= True, num_workers=0,
                                                pin_memory=False, drop_last=False)
    P1 = Image.open('./data/'+train_data_name+'/1.png')
    print(len(train_loader))
    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        # d,a = train_dataset[i]
        # criterion = torch.nn.MSELoss()
        # crop_it = transforms.Resize(crop_size)
        # d = crop_it(d)
        # loss = criterion(d.cuda(),a.cuda())
        # print(loss)
        # img_path_list = os.listdir(data_folder)
        # print(img_path_list[i])
        # d = d.detach().cpu().numpy()
        # plt.subplot(121)
        # plt.imshow(a[ 0, :, :])
        # plt.subplot(122)
        # plt.imshow(d[ 0, :, :])
        # plt.show()
        pass
        print('File Number: {0}/{1}'.format(i,len(train_loader)))
