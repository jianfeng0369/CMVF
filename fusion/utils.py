import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def calc_mean_std(feat, eps=1e-5):

    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

import os
from os import listdir, mkdir, sep
import random
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
import cv2
from torchvision import transforms

def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append( file)
        elif name.endswith('.jpg'):
            images.append(file)
        elif name.endswith('.jpeg'):
            images.append(file)
        name1 = name.split('.')
        names.append(name1[0])
    return images


def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


def tensor_save_rgbimage(tensor, filename, cuda=True):
    if cuda:

        img = tensor.cpu().clamp(0, 255).data[0].numpy()
    else:

        img = tensor.clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def tensor_save_bgrimage(tensor, filename, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename, cuda)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def matSqrt(x):
    U,D,V = torch.svd(x)
    return U * (D.pow(0.5).diag()) * V.t()



def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]

    random.shuffle(original_imgs_path)
    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]
    batches = int(len(original_imgs_path) // BATCH_SIZE)
    return original_imgs_path, batches


def get_image(path, height=256, width=256, mode='L'):
    if mode == 'L':
        image = cv2.imread(path, 0)
    elif mode == 'RGB':
        image = cv2.cvtColor(path, cv2.COLOR_BGR2RGB)

    if height is not None and width is not None:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return image


def get_train_images_auto(paths, height=256, width=256, mode='RGB'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images

def get_test_images(paths, height=None, width=None, mode='L'):
    ImageToTensor = transforms.Compose([transforms.ToTensor()])
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        w, h = image.shape[0], image.shape[1]
        w_s = 256 - w % 256
        h_s = 256 - h % 256
        image = cv2.copyMakeBorder(image, 0, w_s, 0, h_s, cv2.BORDER_CONSTANT,
                                     value=128)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = ImageToTensor(image).float().numpy()*255
    images.append(image)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images

def patch_test(paths, height=None, width=None, mode='L'):
    ImageToTensor = transforms.Compose([transforms.ToTensor()])
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        w, h = image.shape[0], image.shape[1]
        w_s = 256 - w % 256
        h_s = 256 - h % 256
        image = cv2.copyMakeBorder(image, 0, w_s, 0, h_s, cv2.BORDER_CONSTANT, value=128)
        nw = (w // 256 + 1)
        nh = (h // 256 + 1)
        crop = []
        if mode == 'L':
            for j in range(nh):
                for i in range(nw):
                    crop.append(image[i*256:(i+1)*256, j*256:(j+1)*256])
            crop = np.stack(crop, axis=0)

        else:
            image = ImageToTensor(image).float().numpy() * 255
    images.append(crop)
    images = np.stack(images, axis=1)
    images = torch.from_numpy(images).float()
    return images

def save_patch_images(path, data, out):

    if data.shape[1] == 1:
        data = data.reshape([data.shape[0], data.shape[2], data.shape[3]])
    w, h = out.shape[0], out.shape[1]
    nw = (w // 256 + 1)
    nh = (h // 256 + 1)
    result = np.zeros((nw*256, nh*256))
    num = 0
    for j in range(nh):
        for i in range(nw):
            result[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256] = data[num]
            num += 1

    ori = result[0:w, 0:h]
    cv2.imwrite(path, ori)


def save_images(path, data, out):
    w, h = out.shape[0], out.shape[1]
    if data.shape[1] == 1:
        data = data.reshape([data.shape[2], data.shape[3]])
    ori = data[0:w, 0:h]
    cv2.imwrite(path, ori)
