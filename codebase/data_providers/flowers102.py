import warnings
import os
import math
import numpy as np

import PIL

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from ofa.imagenet_codebase.data_providers.base_provider import DataProvider, MyRandomResizedCrop, MyDistributedSampler


class Flowers102DataProvider(DataProvider):
    
    def __init__(self, save_path=None, train_batch_size=32, test_batch_size=512, valid_size=None, n_worker=32,
                 resize_scale=0.08, distort_color=None, image_size=224,
                 num_replicas=None, rank=None):
        
        # warnings.filterwarnings('ignore')
        self._save_path = save_path
        
        self.image_size = image_size  # int or list of int
        self.distort_color = distort_color
        self.resize_scale = resize_scale

        self._valid_transform_dict = {}
        if not isinstance(self.image_size, int):
            assert isinstance(self.image_size, list)
            from ofa.imagenet_codebase.data_providers.my_data_loader import MyDataLoader
            self.image_size.sort()  # e.g., 160 -> 224
            MyRandomResizedCrop.IMAGE_SIZE_LIST = self.image_size.copy()
            MyRandomResizedCrop.ACTIVE_SIZE = max(self.image_size)

            for img_size in self.image_size:
                self._valid_transform_dict[img_size] = self.build_valid_transform(img_size)
            self.active_img_size = max(self.image_size)
            valid_transforms = self._valid_transform_dict[self.active_img_size]
            train_loader_class = MyDataLoader  # randomly sample image size for each batch of training image
        else:
            self.active_img_size = self.image_size
            valid_transforms = self.build_valid_transform()
            train_loader_class = torch.utils.data.DataLoader

        train_transforms = self.build_train_transform()
        train_dataset = self.train_dataset(train_transforms)

        weights = self.make_weights_for_balanced_classes(
            train_dataset.imgs, self.n_classes)
        weights = torch.DoubleTensor(weights)
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        if valid_size is not None:
            raise NotImplementedError("validation dataset not yet implemented")
            # valid_dataset = self.valid_dataset(valid_transforms)
            
            # self.train = train_loader_class(
            #     train_dataset, batch_size=train_batch_size, sampler=train_sampler,
            #     num_workers=n_worker, pin_memory=True)
            # self.valid = torch.utils.data.DataLoader(
            #     valid_dataset, batch_size=test_batch_size,
            #     num_workers=n_worker, pin_memory=True)
        else:
            self.train = train_loader_class(
                train_dataset, batch_size=train_batch_size, sampler=train_sampler,
                num_workers=n_worker, pin_memory=True,
            )
            self.valid = None
        
        test_dataset = self.test_dataset(valid_transforms)
        self.test = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=n_worker, pin_memory=True,
        )
        
        if self.valid is None:
            self.valid = self.test
    
    @staticmethod
    def name():
        return 'flowers102'
    
    @property
    def data_shape(self):
        return 3, self.active_img_size, self.active_img_size  # C, H, W
    
    @property
    def n_classes(self):
        return 102
    
    @property
    def save_path(self):
        if self._save_path is None:
            # self._save_path = '/mnt/datastore/Oxford102Flowers'  # home server
            self._save_path = '/mnt/datastore/Flowers102'  # home server

            if not os.path.exists(self._save_path):
                # self._save_path = '/mnt/datastore/Oxford102Flowers'  # home server
                self._save_path = '/mnt/datastore/Flowers102'  # home server
        return self._save_path
    
    @property
    def data_url(self):
        raise ValueError('unable to download %s' % self.name())
    
    def train_dataset(self, _transforms):
        dataset = datasets.ImageFolder(self.train_path, _transforms)
        return dataset

    # def valid_dataset(self, _transforms):
    #     dataset = datasets.ImageFolder(self.valid_path, _transforms)
    #     return dataset

    def test_dataset(self, _transforms):
        dataset = datasets.ImageFolder(self.test_path, _transforms)
        return dataset
    
    @property
    def train_path(self):
        return os.path.join(self.save_path, 'train')
    
    # @property
    # def valid_path(self):
    #     return os.path.join(self.save_path, 'train')

    @property
    def test_path(self):
        return os.path.join(self.save_path, 'test')
    
    @property
    def normalize(self):
        return transforms.Normalize(
            mean=[0.5178361839861569, 0.4106749456881299, 0.32864167836880803],
            std=[0.2972239085211309, 0.24976049135203868, 0.28533308036347665])

    @staticmethod
    def make_weights_for_balanced_classes(images, nclasses):
        count = [0] * nclasses

        # Counts per label
        for item in images:
            count[item[1]] += 1

        weight_per_class = [0.] * nclasses

        # Total number of images.
        N = float(sum(count))

        # super-sample the smaller classes.
        for i in range(nclasses):
            weight_per_class[i] = N / float(count[i])

        weight = [0] * len(images)

        # Calculate a weight per image.
        for idx, val in enumerate(images):
            weight[idx] = weight_per_class[val[1]]

        return weight

    def build_train_transform(self, image_size=None, print_log=True):
        if image_size is None:
            image_size = self.image_size
        if print_log:
            print('Color jitter: %s, resize_scale: %s, img_size: %s' %
                  (self.distort_color, self.resize_scale, image_size))

        if self.distort_color == 'torch':
            color_transform = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        elif self.distort_color == 'tf':
            color_transform = transforms.ColorJitter(brightness=32. / 255., saturation=0.5)
        else:
            color_transform = None
        
        if isinstance(image_size, list):
            resize_transform_class = MyRandomResizedCrop
            print('Use MyRandomResizedCrop: %s, \t %s' % MyRandomResizedCrop.get_candidate_image_size(),
                  'sync=%s, continuous=%s' % (MyRandomResizedCrop.SYNC_DISTRIBUTED, MyRandomResizedCrop.CONTINUOUS))
        else:
            resize_transform_class = transforms.RandomResizedCrop

        train_transforms = [
            transforms.RandomAffine(
                45, translate=(0.4, 0.4), scale=(0.75, 1.5), shear=None, resample=PIL.Image.BILINEAR, fillcolor=0),
            resize_transform_class(image_size, scale=(self.resize_scale, 1.0)),
            # transforms.RandomHorizontalFlip(),
        ]
        if color_transform is not None:
            train_transforms.append(color_transform)
        train_transforms += [
            transforms.ToTensor(),
            self.normalize,
        ]

        train_transforms = transforms.Compose(train_transforms)
        return train_transforms

    def build_valid_transform(self, image_size=None):
        if image_size is None:
            image_size = self.active_img_size
        return transforms.Compose([
            transforms.Resize(int(math.ceil(image_size / 0.875))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            self.normalize,
        ])

    def assign_active_img_size(self, new_img_size):
        self.active_img_size = new_img_size
        if self.active_img_size not in self._valid_transform_dict:
            self._valid_transform_dict[self.active_img_size] = self.build_valid_transform()
        # change the transform of the valid and test set
        self.valid.dataset.transform = self._valid_transform_dict[self.active_img_size]
        self.test.dataset.transform = self._valid_transform_dict[self.active_img_size]
    
    def build_sub_train_loader(self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None):
        # used for resetting running statistics
        if self.__dict__.get('sub_train_%d' % self.active_img_size, None) is None:
            if num_worker is None:
                num_worker = self.train.num_workers
            
            n_samples = len(self.train.dataset.samples)
            g = torch.Generator()
            g.manual_seed(DataProvider.SUB_SEED)
            rand_indexes = torch.randperm(n_samples, generator=g).tolist()
            
            new_train_dataset = self.train_dataset(
                self.build_train_transform(image_size=self.active_img_size, print_log=False))
            chosen_indexes = rand_indexes[:n_images]
            if num_replicas is not None:
                sub_sampler = MyDistributedSampler(new_train_dataset, num_replicas, rank, np.array(chosen_indexes))
            else:
                sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(chosen_indexes)
            sub_data_loader = torch.utils.data.DataLoader(
                new_train_dataset, batch_size=batch_size, sampler=sub_sampler,
                num_workers=num_worker, pin_memory=True,
            )
            self.__dict__['sub_train_%d' % self.active_img_size] = []
            for images, labels in sub_data_loader:
                self.__dict__['sub_train_%d' % self.active_img_size].append((images, labels))
        return self.__dict__['sub_train_%d' % self.active_img_size]
