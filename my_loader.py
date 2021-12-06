
import numpy as np
import torch.utils.data as data

from PIL import Image
import os
import os.path
import pickle

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, path


    def __len__(self):
        return len(self.imgs)


class mix_img_query_loader(data.Dataset):
    def __init__(self, transform=None, target_transform=None, loader=default_loader):
        images = []
        targets = []

        datainfo = pickle.load(open('./images/cifar10_mix_images_499000.pkl', 'rb')) ### Mix=1000 and Comb=1000*999/2=499500; 500 is truncated for rotation processing
#        datainfo = pickle.load(open('./images/left/cifar10_left_unlabeled_489000.pkl', 'rb'))
        
        self.image_path_1 = datainfo['x_1_unlabeled_path']
        self.image_path_2 = datainfo['x_2_unlabeled_path']

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):

        ### Path in Pickle
        img_path_1 = self.image_path_1[index]
        img_1 = Image.open(img_path_1).convert('RGB')

        img_path_2 = self.image_path_2[index]
        img_2 = Image.open(img_path_2).convert('RGB')

        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        return img_1, img_2, img_path_1, img_path_2

    def __len__(self):
        return len(self.image_path_1)


class real_img_query_loader(data.Dataset):
    def __init__(self, transform=None, target_transform=None, loader=default_loader):
        images = []
        targets = []

        datainfo = pickle.load(open('./images/cifar10_real_images_1000.pkl', 'rb'))
#        datainfo = pickle.load(open('./images/cifar10_real_images_2000.pkl', 'rb'))

        self.image = datainfo['x_oris']

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):

        ### Array In Pickle
        sample = self.image[index].transpose(1,2,0)
        img = np.asarray(sample*256,dtype=np.uint8)
        img = Image.fromarray(img,'RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.image)



class active_query_loader(data.Dataset):
    def __init__(self, transform=None, target_transform=None, loader=default_loader):
        images = []
        targets = []


        datainfo = pickle.load(open('./images/query/cifar10_query_new_label_10000.pkl', 'rb'))
#        datainfo = pickle.load(open('./images/query/cifar10_query_new_label_20000.pkl', 'rb'))
#        datainfo = pickle.load(open('./images/query/cifar10_query_new_label_40000.pkl', 'rb'))
#        datainfo = pickle.load(open('./images/query/cifar10_query_new_label_80000.pkl', 'rb'))


#        self.image_path_1 = datainfo['x_1_labeled_path']
#        self.image_path_2 = datainfo['x_2_labeled_path']
#        self.w_1 = datainfo['mix_weights']

        self.mix_images = datainfo['mix_array']

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):

        ### Array In Pickle
        sample = self.mix_images[index].transpose(1,2,0)
        img_mix = np.asarray(sample*256,dtype=np.uint8)
        img_mix = Image.fromarray(img_mix,'RGB')

        if self.transform is not None:
            img_mix = self.transform(img_mix)

        return img_mix

    def __len__(self):
        return len(self.mix_images)
#        return len(self.w_1)




class active_learning_loader(data.Dataset):
    def __init__(self, transform=None, target_transform=None, loader=default_loader):
        images = []
        targets = []

        datainfo = pickle.load(open('./images/query/cifar10_query_label_1000.pkl', 'rb')) ### Stage 0
#        datainfo = pickle.load(open('./images/query/cifar10_query_label_11000.pkl', 'rb'))  ### Stage 1                                  
#        datainfo = pickle.load(open('./images/query/cifar10_query_label_21000.pkl', 'rb'))  ### Stage 2

        self.mix_images = datainfo['x_labeled_oris']
        self.logits = datainfo['y_labeled_logits']
        self.labels = datainfo['y_labeled_label']

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):

        ### Array In Pickle#
        sample = self.mix_images[index].transpose(1,2,0)
        img_mix = np.asarray(sample*256,dtype=np.uint8)
        img_mix = Image.fromarray(img_mix,'RGB')

        if self.transform is not None:
             img_mix = self.transform(img_mix)

        logit = self.logits[index]
        label = self.labels[index]

        return img_mix, logit, label

    def __len__(self):
        return len(self.mix_images)


