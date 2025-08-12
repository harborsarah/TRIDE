import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
from torch.utils.data.distributed import DistributedSampler

f = 1266.417203046554
cx = 816.2670197447984
cy = 491.50706579294757

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])

def read_paths(filepath):
    '''
    Reads a newline delimited file containing paths

    Arg(s):
        filepath : str
            path to file to be read
    Return:
        list[str] : list of paths
    '''

    path_list = []
    with open(filepath) as f:
        while True:
            path = f.readline().rstrip('\n')

            # If there was nothing to read
            if path == '':
                break

            path_list.append(path)

    return path_list
    

class TextDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.main_path = args.main_path
            self.image_path = args.train_image_path
            self.radar_path = args.train_radar_path
            self.text_feature_general_path = args.train_text_feature_general_path
            self.text_feature_left_path = args.train_text_feature_left_path
            self.text_feature_mid_left_path = args.train_text_feature_mid_left_path
            self.text_feature_mid_right_path = args.train_text_feature_mid_right_path
            self.text_feature_right_path = args.train_text_feature_right_path

            self.weather_condition_path = args.train_weather_condition_path
            self.ground_truth_path = args.train_ground_truth_path
            self.lidar_path = args.train_lidar_path

            text_feature_general_paths = read_paths(self.text_feature_general_path)
            text_feature_left_paths = read_paths(self.text_feature_left_path)
            text_feature_mid_left_paths = read_paths(self.text_feature_mid_left_path)
            text_feature_mid_right_paths = read_paths(self.text_feature_mid_right_path)
            text_feature_right_paths = read_paths(self.text_feature_right_path)

            weather_conditions = read_paths(self.weather_condition_path)
            image_paths = read_paths(self.image_path)
            ground_truth_paths = read_paths(self.ground_truth_path)
            lidar_paths = read_paths(self.lidar_path)
            radar_paths = read_paths(self.radar_path)

            self.dataset = DataLoadPreprocess(args, mode, image_paths=image_paths, weather_conditions=weather_conditions, main_path=self.main_path, \
                                                text_feature_general_paths=text_feature_general_paths, text_feature_left_paths=text_feature_left_paths,\
                                                text_feature_mid_left_paths=text_feature_mid_left_paths, text_feature_mid_right_paths=text_feature_mid_right_paths,\
                                                text_feature_right_paths=text_feature_right_paths, ground_truth_paths=ground_truth_paths, lidar_paths=lidar_paths, \
                                                radar_paths=radar_paths, transform=preprocessing_transforms(mode))

            self.sampler = DistributedSampler(self.dataset, 
                                              num_replicas=getattr(args, 'world_size', 1),
                                              rank=getattr(args, 'rank', 0),
                                              shuffle=True) if getattr(args, 'distributed', False) else None

            self.data = DataLoader(
                self.dataset, args.batch_size,
                shuffle=(self.sampler is None),
                sampler=self.sampler,
                num_workers=args.num_threads,
                pin_memory=True,
            )
        else:
            self.main_path = args.main_path
            self.image_path = args.validation_image_path
            self.radar_path = args.validation_radar_path

            self.text_feature_general_path = args.validation_text_feature_general_path
            self.text_feature_left_path = args.validation_text_feature_left_path
            self.text_feature_mid_left_path = args.validation_text_feature_mid_left_path
            self.text_feature_mid_right_path = args.validation_text_feature_mid_right_path
            self.text_feature_right_path = args.validation_text_feature_right_path

            self.weather_condition_path = args.validation_weather_condition_path
            self.ground_truth_path = args.validation_ground_truth_path

            # text_embs_paths = read_paths(self.text_emb_path)
            text_feature_general_paths = read_paths(self.text_feature_general_path)
            text_feature_left_paths = read_paths(self.text_feature_left_path)
            text_feature_mid_left_paths = read_paths(self.text_feature_mid_left_path)
            text_feature_mid_right_paths = read_paths(self.text_feature_mid_right_path)
            text_feature_right_paths = read_paths(self.text_feature_right_path)

            weather_conditions = read_paths(self.weather_condition_path)
            image_paths = read_paths(self.image_path)
            ground_truth_paths = read_paths(self.ground_truth_path)
            radar_paths = read_paths(self.radar_path)

            self.dataset = DataLoadPreprocess(args, mode, image_paths=image_paths, weather_conditions=weather_conditions, main_path=self.main_path, \
                                                        text_feature_general_paths=text_feature_general_paths, text_feature_left_paths=text_feature_left_paths,\
                                                        text_feature_mid_left_paths=text_feature_mid_left_paths, text_feature_mid_right_paths=text_feature_mid_right_paths,\
                                                        text_feature_right_paths=text_feature_right_paths, ground_truth_paths=ground_truth_paths, \
                                                        radar_paths=radar_paths, transform=preprocessing_transforms(mode))

            self.sampler = None
            self.data = DataLoader(
                self.dataset, 1,
                shuffle=False,
                num_workers=1,
                pin_memory=True,
            )
    
    def set_epoch(self, epoch):
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)

class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, image_paths, weather_conditions, main_path, \
                text_feature_general_paths, text_feature_left_paths, text_feature_mid_left_paths,\
                text_feature_mid_right_paths, text_feature_right_paths, \
                ground_truth_paths=None, lidar_paths=None, radar_paths=None, transform=None):
        self.args = args
        self.mode = mode
        self.image_paths = image_paths
        self.text_feature_general_paths = text_feature_general_paths
        self.text_feature_left_paths = text_feature_left_paths
        self.text_feature_mid_left_paths = text_feature_mid_left_paths
        self.text_feature_mid_right_paths = text_feature_mid_right_paths
        self.text_feature_right_paths = text_feature_right_paths
        self.weather_conditions = weather_conditions
        self.main_path = main_path
        self.ground_truth_paths = ground_truth_paths
        self.lidar_paths = lidar_paths
        self.radar_paths = radar_paths
        self.transform = transform

    def __getitem__(self, idx):
        if self.mode == 'train':
            K = np.array([
                [f, 0, cx, 0],
                [0, f, cy, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)
            image_path = self.image_paths[idx]
            depth_path = self.main_path + self.ground_truth_paths[idx]
            lidar_path = self.main_path + self.lidar_paths[idx]

            image = Image.open(image_path)
            image = np.asarray(image, dtype=np.float32) / 255.0
            width = image.shape[1]
            height = image.shape[0]
            radar_path = self.main_path + self.radar_paths[idx]
            radar_points_2d = np.load(radar_path)
            radar_channels = np.zeros((height, width, radar_points_2d.shape[-1]-3), dtype=np.float32)
            for i in range(radar_points_2d.shape[0]):
                x = int(radar_points_2d[i, 0])
                y = int(radar_points_2d[i, 1])
                # last feature is alignment, not useful in this project
                
                # generate radar channels
                if radar_channels[y, x, 0] == 0:
                    radar_channels[y, x] = radar_points_2d[i, 2:-1]
                elif radar_channels[y, x, 0] > radar_points_2d[i, 2]:
                    radar_channels[y, x] = radar_points_2d[i, 2:-1]
                elif radar_channels[y, x, -1] == 0 and radar_points_2d[i, -1] != 0:
                    radar_channels[y, x] = radar_points_2d[i, 2:-1]
            
            text_length = np.zeros((5,))

            # Now, for each paragraph, encoded sentence by sentence
            # text_feature_general:
            text_feature_general = torch.load(self.main_path + self.text_feature_general_paths[idx]).float()
            text_length[0] = text_feature_general.shape[0]
            text_feature_general_pad = torch.zeros((16, text_feature_general.shape[1]))
            text_feature_general_pad[:text_feature_general.shape[0]] = text_feature_general

            # text_feature_left:
            text_feature_left = torch.load(self.main_path + self.text_feature_left_paths[idx]).float()
            text_length[1] = text_feature_left.shape[0]
            text_feature_left_pad = torch.zeros((9, text_feature_left.shape[1]))
            text_feature_left_pad[:text_feature_left.shape[0]] = text_feature_left

            # text_feature_mid_left:
            text_feature_mid_left = torch.load(self.main_path + self.text_feature_mid_left_paths[idx]).float()
            text_length[2] = text_feature_mid_left.shape[0]
            text_feature_mid_left_pad = torch.zeros((9, text_feature_mid_left.shape[1]))
            text_feature_mid_left_pad[:text_feature_mid_left.shape[0]] = text_feature_mid_left

            # text_feature_mid_right:
            text_feature_mid_right = torch.load(self.main_path + self.text_feature_mid_right_paths[idx]).float()
            text_length[3] = text_feature_mid_right.shape[0]
            text_feature_mid_right_pad = torch.zeros((9, text_feature_mid_right.shape[1]))
            text_feature_mid_right_pad[:text_feature_mid_right.shape[0]] = text_feature_mid_right

            # text_feature_right:
            text_feature_right = torch.load(self.main_path + self.text_feature_right_paths[idx]).float()
            text_length[4] = text_feature_right.shape[0]
            text_feature_right_pad = torch.zeros((9, text_feature_right.shape[1]))
            text_feature_right_pad[:text_feature_right.shape[0]] = text_feature_right

            text_mask = np.ones((height,width))
            text_mask[:, int(width/4):int(width/2)] = 2
            text_mask[:, int(width/2):int(3*width/4)] = 3
            text_mask[:, int(3*width/4):] = 4
            text_mask = np.expand_dims(text_mask, axis=2)

            weather_condition = self.weather_conditions[idx]
            label = self.to_label(weather_condition)

            depth_gt = Image.open(depth_path)
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            depth_gt = depth_gt / 256.0

            lidar = Image.open(lidar_path)
            lidar = np.asarray(lidar, dtype=np.float32)
            lidar = np.expand_dims(lidar, axis=2)
            lidar = lidar / 256.0

            # if use NewCRFDepth maps as dense map supervision
            # change the pixel values which single scan depth has to the ground truth
            idx = np.where(lidar!=0)
            depth = lidar[idx[0], idx[1]]
            x = idx[1]
            y = idx[0]
            depth_gt[y, x] = depth

            image, depth_gt, lidar, text_mask, radar_channels, K = self.random_crop(image, depth_gt, lidar, text_mask, self.args.input_height, self.args.input_width, radar_channels, K)

            image, depth_gt, lidar, text_mask, radar_channels, K = self.train_preprocess(image, depth_gt, lidar, text_mask, radar_channels, K)

            # for map back to 3D point, concatenate the radar channels with the text mask, so that it is easy to track which point belongs to which region
            radar_points2d_crop = self.channel_back_to_points(np.concatenate((radar_channels, text_mask), -1))
            radar_points3d_crop = self.point2d_to_3d(radar_points2d_crop, K)
            radar_points3d = np.zeros((125, radar_points3d_crop.shape[1]))
            radar_points3d[:radar_points3d_crop.shape[0]] = radar_points3d_crop
            radar_points3d = torch.from_numpy(radar_points3d).float()


            sample = {'text_feature_general': text_feature_general_pad, 'text_feature_left': text_feature_left_pad, 'text_feature_mid_left': text_feature_mid_left_pad, \
                      'text_feature_mid_right': text_feature_mid_right_pad, 'text_feature_right': text_feature_right_pad, 'text_length': text_length, \
                      'label': label, 'image':image, 'depth': depth_gt, 'lidar': lidar, 'text_mask':text_mask, 'radar_channels':radar_channels, \
                      'radar_points': radar_points3d}


        else:
            K = np.array([
                [f, 0, cx, 0],
                [0, f, cy, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)


            text_length = np.zeros((5,))
            # text_feature_general:
            text_feature_general = torch.load(self.main_path + self.text_feature_general_paths[idx]).float()
            text_length[0] = text_feature_general.shape[0]

            # text_feature_left:
            text_feature_left = torch.load(self.main_path + self.text_feature_left_paths[idx]).float()
            text_length[1] = text_feature_left.shape[0]

            # text_feature_mid_left:
            text_feature_mid_left = torch.load(self.main_path + self.text_feature_mid_left_paths[idx]).float()
            text_length[2] = text_feature_mid_left.shape[0]

            # text_feature_mid_right:
            text_feature_mid_right = torch.load(self.main_path + self.text_feature_mid_right_paths[idx]).float()
            text_length[3] = text_feature_mid_right.shape[0]

            # text_feature_right:
            text_feature_right = torch.load(self.main_path + self.text_feature_right_paths[idx]).float()
            text_length[4] = text_feature_right.shape[0]

            weather_condition = self.weather_conditions[idx]
            
            image_path = self.image_paths[idx]
            image = Image.open(image_path)
            image = np.asarray(image, dtype=np.float32) / 255.0
            width = image.shape[1]
            height = image.shape[0]

            text_mask = np.ones((height,width))
            text_mask[:, int(width/4):int(width/2)] = 2
            text_mask[:, int(width/2):int(3*width/4)] = 3
            text_mask[:, int(3*width/4):] = 4
            text_mask = np.expand_dims(text_mask, axis=2)

            depth_path =  self.main_path + self.ground_truth_paths[idx]
            depth_gt = Image.open(depth_path)
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            depth_gt = depth_gt / 256.0

            radar_path = self.main_path + self.radar_paths[idx]
            radar_points_2d = np.load(radar_path)
            radar_channels = np.zeros((height, width, radar_points_2d.shape[-1]-3), dtype=np.float32)
            for i in range(radar_points_2d.shape[0]):
                x = int(radar_points_2d[i, 0])
                y = int(radar_points_2d[i, 1])
                # last feature is alignment, not useful in this project
                
                # generate radar channels
                if radar_channels[y, x, 0] == 0:
                    radar_channels[y, x] = radar_points_2d[i, 2:-1]
                elif radar_channels[y, x, 0] > radar_points_2d[i, 2]:
                    radar_channels[y, x] = radar_points_2d[i, 2:-1]
                elif radar_channels[y, x, -1] == 0 and radar_points_2d[i, -1] != 0:
                    radar_channels[y, x] = radar_points_2d[i, 2:-1]
            radar_channels = radar_channels[4:, ...]

            K[1, 2] = K[1, 2] - 4

            image = image[4:, ...] # (896, 1600, 3)
            depth_gt = depth_gt[4:, ...]
            text_mask = text_mask[4:, ...]

            label = self.to_label(weather_condition)
            radar_points2d = self.channel_back_to_points(np.concatenate((radar_channels, text_mask), -1))
            radar_points3d = self.point2d_to_3d(radar_points2d, K)
            radar_points3d = torch.from_numpy(radar_points3d).float()

            sample = {'text_feature_general': text_feature_general, 'text_feature_left': text_feature_left, 'text_feature_mid_left': text_feature_mid_left, \
                      'text_feature_mid_right': text_feature_mid_right, 'text_feature_right': text_feature_right, 'text_length': text_length, \
                      'label': label, 'image':image, 'depth': depth_gt, 'text_mask':text_mask, 'radar_channels':radar_channels, \
                      'radar_points': radar_points3d}


        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def channel_back_to_points(self, radar_channels):
        y, x = np.where(radar_channels[..., 0] != 0)
        radar_points = np.concatenate([x[:, None], y[:, None], radar_channels[y, x]], axis=-1) # x, y, depth, rcs, vx, vy, text_mask

        return radar_points

    def point2d_to_3d(self, radar_points, K, normalize=False):
        viewpad_inv = np.linalg.inv(K)
        depth = radar_points[:,2:3]
        radar_point3d = np.concatenate((radar_points[:, 0:2], np.ones((depth.shape[0], 1)), 1.0/depth), axis=-1)
        radar_point3d = np.transpose(radar_point3d)    
        radar_point3d = depth.transpose().repeat(4, 0).reshape(4, -1) * np.dot(viewpad_inv, radar_point3d) # (4, N)
        if normalize:
            point3d = radar_point3d[:3, :].transpose()
            point3d, centroid, furthest_distance = self.normalize_point_cloud(point3d)
            radar_point3d = np.concatenate((point3d, radar_points[:, 3:]), axis=-1)
            return radar_point3d, centroid, furthest_distance

        else:
            radar_point3d = np.concatenate((radar_point3d[:3, :].transpose(), radar_points[:, 3:]), axis=-1)
            return radar_point3d
        
    def resize_depth(self, depth):
        # depth = np.array(depth)
        depth = depth.squeeze()
        re_depth = np.zeros((450,800))
        pts = np.where(depth!=0)
        re_depth[(pts[0][:]/2).astype(np.int32), (pts[1][:]/2).astype(np.int32)] = depth[pts[0][:], pts[1][:]]
        re_depth = np.expand_dims(re_depth[2:], 2)
        
        return re_depth

    def to_label(self, label):
        if label == 'sunny':
            return np.asarray([1, 0, 0]).astype(np.float32)
        elif label == 'night' or label == 'nightrain':
            return np.asarray([0, 1, 0]).astype(np.float32)
        elif label == 'rain':
            return np.asarray([0, 0, 1]).astype(np.float32)
        else:
            return -1

    def random_crop(self, img, depth, lidar, text_mask, height, width, radar=None, K=None):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)

        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        lidar = lidar[y:y + height, x:x + width, :]
        text_mask = text_mask[y:y + height, x:x + width, :]

        if radar is not None:
            radar = radar[y:y + height, x:x + width, :]
            K[0, 2] = K[0, 2]- x
            K[1, 2] = K[1, 2] - y
            return img, depth, lidar, text_mask, radar, K

        return img, depth, lidar, text_mask, None, None

    def train_preprocess(self, image, depth_gt, lidar, text_mask, radar=None, K=None):
        # Random flipping
        do_flip = random.random()
        w = image.shape[1]

        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()
            lidar = (lidar[:, ::-1, :]).copy()
            text_mask = (text_mask[:, ::-1, :]).copy()

            if radar is not None:
                radar = (radar[:, ::-1, :]).copy()
                K[0, 2] = w - K[0, 2] - 1

    
        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        if radar is not None:
            return image, depth_gt, lidar, text_mask, radar, K
        else:
            return image, depth_gt, lidar, text_mask, None, None
            
    
    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.weather_conditions)

class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __call__(self, sample):
        text_feature_general = sample['text_feature_general'].detach()
        text_feature_left = sample['text_feature_left'].detach()
        text_feature_mid_left = sample['text_feature_mid_left'].detach()
        text_feature_mid_right = sample['text_feature_mid_right'].detach()
        text_feature_right = sample['text_feature_right'].detach()

        text_length = torch.from_numpy(sample['text_length'])

        text_mask = self.to_tensor(sample['text_mask'])
        label = torch.from_numpy(sample['label'])
        image = sample['image']
        image = self.to_tensor(image)
        image = self.normalize(image)
        depth = sample['depth']
        depth_gt = self.to_tensor(depth)
        radar_channels = sample['radar_channels']
        radar_points = sample['radar_points']
        radar_channels = self.to_tensor(radar_channels)
        radar_points = radar_points.permute(1, 0) # (channels, #points)

        if self.mode == 'train':
            lidar = sample['lidar']
            lidar = self.to_tensor(lidar)
            return  {'text_feature_general': text_feature_general, 'text_feature_left': text_feature_left, 'text_feature_mid_left': text_feature_mid_left, \
                    'text_feature_mid_right': text_feature_mid_right, 'text_feature_right': text_feature_right, 'text_length': text_length, \
                    'label': label, 'image':image, 'depth': depth_gt, 'lidar': lidar, 'text_mask':text_mask, \
                    'radar_channels':radar_channels, 'radar_points':radar_points}

        return  {'text_feature_general': text_feature_general, 'text_feature_left': text_feature_left, 'text_feature_mid_left': text_feature_mid_left, \
                'text_feature_mid_right': text_feature_mid_right, 'text_feature_right': text_feature_right, 'text_length': text_length, \
                'label': label, 'image':image, 'depth': depth_gt, 'text_mask':text_mask, 'radar_channels':radar_channels, 'radar_points':radar_points}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
        
        if isinstance(pic, np.ndarray):
            if len(pic.shape) > 2:
                img = torch.from_numpy(pic.transpose((2, 0, 1)))
                return img
            else:
                arr = torch.from_numpy(pic)
                return arr

        
        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
