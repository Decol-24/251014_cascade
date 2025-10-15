import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines, pfm_imread


class SceneFlowDatset(Dataset):
    def __init__(self, datapath, list_filename, training, crop_width, crop_height, test_crop_width, test_crop_height):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(self.datapath)
        self.training = training
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.test_crop_width = test_crop_width
        self.test_crop_height = test_crop_height

    def load_path(self, datapath):
        all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = dataloader_SceneFlow(datapath)
        return all_left_img, all_right_img, all_left_disp

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data, scale = pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))

        if self.training:
            w, h = left_img.size
            crop_w, crop_h = self.crop_width, self.crop_height

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity}
        else:
            w, h = left_img.size
            crop_w, crop_h = self.test_crop_width, self.test_crop_height

            left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
            right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
            disparity = disparity[h - crop_h:h, w - crop_w: w]

            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "top_pad": 0,
                    "right_pad": 0}

# class myImageFloder_SceneFlow(data.Dataset):
#     def __init__(self, left, right, left_disparity, training):

#         self.left = left
#         self.right = right
#         self.disp_L = left_disparity
#         self.training = training
#         if self.training:
#             self.transforms = inception_color_preproccess()
#         else:
#             self.transforms = base_norm()

#     def __getitem__(self, index):

#         left = self.left[index]
#         right = self.right[index]
#         _disp_L = self.disp_L[index]

#         left_img = Image.open(left).convert('RGB')
#         right_img = Image.open(right).convert('RGB')
#         disp_L, scaleL = self.disparity_loader(_disp_L)
#         disp_L = torch.tensor(disp_L.copy(), dtype=torch.float32)
        
#         # imgL   [wigth:960,height:540] type:Image
#         # dataL  [wigth:960,height:540]

#         if self.training:
#             #随机区域裁剪
#             left_img = self.transforms(left_img)
#             right_img = self.transforms(right_img)

#             left_img,right_img,disp_L = crop_img(left_img,right_img,disp_L,th=256,tw=512)

#             return left_img, right_img, disp_L

#         else:
#             left_img = self.transforms(left_img)
#             right_img = self.transforms(right_img)
#             left_img,right_img,disp_L = crop_img(left_img,right_img,disp_L,th=256,tw=512)

#             return left_img, right_img, disp_L

#     def __len__(self):
#         return len(self.left)
    
#     def disparity_loader(self,path):
#         if '.png' in path:
#             return Image.open(path)
#         else:
#             return readPFM(path)
        
def is_image_file(filename):
    IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
        
def dataloader_SceneFlow(filepath,select=[0,1,2]):
    all_left_img = []
    all_right_img = []
    all_left_disp = []
    test_left_img = []
    test_right_img = []
    test_left_disp = []

    classes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
    image = [img for img in classes if img.find('frames_finalpass') > -1] #找到所有包含原始图片的文件夹
    disp = [dsp for dsp in classes if dsp.find('disparity') > -1]  #找到所有包含视差文件的文件夹

    # monkaa_part
    if 0 in select:
        monkaa_path = filepath + [x for x in image if 'monkaa' in x][0] #找到monkaa原始图片的文件夹
        monkaa_disp = filepath + [x for x in disp if 'monkaa' in x][0] #找到monkaa视差文件的文件夹
        monkaa_path = monkaa_path + '/frames_finalpass/'
        monkaa_disp = monkaa_disp + '/disparity/'

        monkaa_dir = os.listdir(monkaa_path)
        for dd in monkaa_dir:
            for im in os.listdir(monkaa_path + '/' + dd + '/left/'):
                if is_image_file(monkaa_path + '/' + dd + '/left/' + im):
                    all_left_img.append(monkaa_path + '/' + dd + '/left/' + im) #对每个列出文件判断，如果是图片文件，则添加到列表中
                    all_left_disp.append(monkaa_disp + '/' + dd + '/left/' + im.split(".")[0] + '.pfm')  #对每个列出文件判断，如果是视差文件，则添加到此列表中

            for im in os.listdir(monkaa_path + '/' + dd + '/right/'):
                if is_image_file(monkaa_path + '/' + dd + '/right/' + im):
                    all_right_img.append(monkaa_path + '/' + dd + '/right/' + im) #对右侧图像做同处理。右侧图片不含视差文件

    if 1 in select:
        flying_path = filepath + [x for x in image if x == 'frames_finalpass'][0] #找到包含飞行图像的文件夹，这个文件夹名字就是frames_finalpass
        flying_disp = filepath + [x for x in disp if x == 'disparity'][0] #找到包含飞行图像的视差文件的文件夹
        flying_dir = flying_path + '/TRAIN/' #飞行图像的训练集目录
        subdir = ['A', 'B', 'C'] #飞行图像的子目录

        for ss in subdir:
            flying = os.listdir(flying_dir + ss)

            for ff in flying:
                imm_l = os.listdir(flying_dir + ss + '/' + ff + '/left/')
                for im in imm_l:
                    if is_image_file(flying_dir + ss + '/' + ff + '/left/' + im):
                        all_left_img.append(flying_dir + ss + '/' + ff + '/left/' + im) #添加左图

                    all_left_disp.append(flying_disp + '/TRAIN/' + ss + '/' + ff + '/left/' + im.split(".")[0] + '.pfm') #添加左图对应的视差文件，在另一个文件夹中

                    if is_image_file(flying_dir + ss + '/' + ff + '/right/' + im): #添加右图
                        all_right_img.append(flying_dir + ss + '/' + ff + '/right/' + im)

        flying_dir = flying_path + '/TEST/' #飞行图像的测试集目录

        subdir = ['A', 'B', 'C']

        for ss in subdir:
            flying = os.listdir(flying_dir + ss)

            for ff in flying:
                imm_l = os.listdir(flying_dir + ss + '/' + ff + '/left/')
                for im in imm_l:
                    if is_image_file(flying_dir + ss + '/' + ff + '/left/' + im):
                        test_left_img.append(flying_dir + ss + '/' + ff + '/left/' + im)

                    test_left_disp.append(flying_disp + '/TEST/' + ss + '/' + ff + '/left/' + im.split(".")[0] + '.pfm')

                    if is_image_file(flying_dir + ss + '/' + ff + '/right/' + im):
                        test_right_img.append(flying_dir + ss + '/' + ff + '/right/' + im)

    if 2 in select:
        driving_dir = filepath + [x for x in image if 'driving' in x][0]
        driving_disp = filepath + [x for x in disp if 'driving' in x][0]

        driving_dir = driving_dir + '/frames_finalpass/'
        driving_disp = driving_disp + '/disparity/'

        subdir1 = ['35mm_focallength', '15mm_focallength'] #一级子目录
        subdir2 = ['scene_backwards', 'scene_forwards'] #二级子目录
        subdir3 = ['fast', 'slow'] #三级子目录

        for i in subdir1:
            for j in subdir2:
                for k in subdir3:
                    imm_l = os.listdir(driving_dir + i + '/' + j + '/' + k + '/left/')
                    for im in imm_l:
                        if is_image_file(driving_dir + i + '/' + j + '/' + k + '/left/' + im):
                            all_left_img.append(driving_dir + i + '/' + j + '/' + k + '/left/' + im)
                        all_left_disp.append(
                            driving_disp + '/' + i + '/' + j + '/' + k + '/left/' + im.split(".")[0] + '.pfm')

                        if is_image_file(driving_dir + i + '/' + j + '/' + k + '/right/' + im):
                            all_right_img.append(driving_dir + i + '/' + j + '/' + k + '/right/' + im)
        #基本上和上面一样的逻辑，添加左图、右图和视差文件

    return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp