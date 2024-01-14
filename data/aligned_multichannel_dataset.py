import os
import re

import imageio
import numpy as np
from torch import tensor
from torchvision import transforms

from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset_by_pattern


def multichannel_loader(dir_name, image_id, filename_segments, channel_ids):
    channel_filepaths = []
    for channel_id in channel_ids:
        filename = str(filename_segments[0] + f"{image_id}" + \
                       filename_segments[1] + f"{channel_id}" + \
                       filename_segments[2])
        filepath = os.path.join(dir_name, filename)
        channel_filepaths.append(filepath)

    channels = [np.array(imageio.v2.imread(filepath)) for filepath in channel_filepaths]
    img = np.stack(channels, axis=-1).transpose(2, 0, 1).astype(np.float32)
    # TODO: is the transpose suitable here?

    return img, channel_filepaths


class AlignedMultichannelDataset(BaseDataset):
    """A dataset class for paired image dataset that are not concatenated to {A,B} images.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--max_A', type=float, default=1.0, help='A divided by this value')
        parser.add_argument('--max_B', type=float, default=1.0, help='B divided by this value')
        parser.add_argument('--filename_re_pattern', type=str, required=True, help='RE to match filenames and extract image IDs')
        parser.add_argument('--filename_segments_A', type=str, required=True, nargs=3, help='Segments of filename for A')
        parser.add_argument('--filename_segments_B', type=str, required=True, nargs=3, help='Segments of filename for B')
        parser.add_argument('--channel_ids_A', type=str, required=True, nargs='+', help='Channel IDs for A')
        parser.add_argument('--channel_ids_B', type=str, required=True, nargs='+', help='Channel IDs for B')
        parser.set_defaults(max_A=1.0, max_B=1.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        A_paths = sorted(make_dataset_by_pattern(self.dir_AB, self.opt.filename_re_pattern, opt.max_dataset_size))  # get image paths
        self.image_ids = [re.match(self.opt.filename_re_pattern, os.path.basename(path)).group(1) for path in A_paths]

        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image

        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        self.alignment_mode = 'both'  # TODO: make this an option

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths
        """
        # read a image given a random integer index
        image_id = self.image_ids[index]

        # Load multi-channel image
        if self.alignment_mode in ['A', 'both']:
            A, A_paths = multichannel_loader(self.dir_AB, image_id, self.opt.filename_segments_A, self.opt.channel_ids_A)
            A = tensor(A / self.opt.max_A)

        # Load heightmap image
        if self.alignment_mode in ['B', 'both']:
            B, B_paths = multichannel_loader(self.dir_AB, image_id, self.opt.filename_segments_B, self.opt.channel_ids_B)
            B = tensor(B / self.opt.max_B)

            # B = A[0:1, ...]  # DEBUGGING!!!

        # apply corresponding transforms to A and B
        # transform_params = get_params(self.opt, A.size()[-2:])
        # assert transform_params['crop_pos'] == (0, 0), "AlignedMultichannelDataset does not support cropping"
        # A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        # B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        # A = A_transform(A)
        # B = B_transform(B)
        crop_params = transforms.RandomCrop.get_params(A, output_size=(self.opt.crop_size, self.opt.crop_size))
        A = transforms.functional.crop(A, *crop_params)
        B = transforms.functional.crop(B, *crop_params)

        # TODO: handle the various alignment modes
        return {'A': A, 'B': B, 'A_paths': A_paths[0], 'B_paths': B_paths[0]}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_ids)
