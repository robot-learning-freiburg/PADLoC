import glob
from itertools import chain
from os import path
import torch
import numpy as np
import torch.utils.data as data
import umsgpack
from PIL import Image
from .laserscan_p import LaserScan

class ISSDataset_(data.Dataset):
    """Instance segmentation dataset

    This assumes the dataset to be formatted as defined in:
        https://github.com/mapillary/seamseg/wiki/Dataset-format

    Parameters
    ----------
    root_dir : str
        Path to the root directory of the dataset
    split_name : str
        Name of the split to load: this must correspond to one of the files in `root_dir/lst`
    transform : callable
        Transformer function applied to the loaded entries to prepare them for pytorch. This should be callable as
        `transform(img, msk, cat, cls)`, where:
            - `img` is a PIL.Image with `mode="RGB"`, containing the RGB data
            - `msk` is a list of PIL.Image with `mode="L"`, containing the instance segmentation masks
            - `cat` is a list containing the instance id to class id mapping
            - `cls` is an integer specifying a requested class for class-uniform sampling, or None

    """
    _IMG_DIR = "img"
    _MSK_DIR = "msk"
    _LST_DIR = "lst"
    _METADATA_FILE = "metadata.bin"

    def __init__(self, root_dir, split_name, transform):
        super(ISSDataset_, self).__init__()
        self.root_dir = root_dir
        self.split_name = split_name
        self.transform = transform

        # Folders
        self._img_dir = path.join(root_dir, ISSDataset_._IMG_DIR)
        self._msk_dir = path.join(root_dir, ISSDataset_._MSK_DIR)
        self._lst_dir = path.join(root_dir, ISSDataset_._LST_DIR)
        self.sensor_img_H = 64
        self.sensor_img_W = 2048
        self.sensor_fov_up = 3
        self.sensor_fov_down = -25
        self.max_points = 150000
        self.sensor_img_means = torch.tensor([12.12,10.88,0.23,-1.04,0.21],
                                         dtype=torch.float).view(-1,1,1)
        self.sensor_img_stds = torch.tensor([12.32,11.47,6.91,0.86,0.16],
                                        dtype=torch.float).view(-1,1,1)
        for d in self._img_dir, self._msk_dir, self._lst_dir:
            if not path.isdir(d):
                raise IOError("Dataset sub-folder {} does not exist".format(d))

        # Load meta-data and split
        self._meta, self._images = self._load_split()

    def _load_split(self):
        with open(path.join(self.root_dir, ISSDataset_._METADATA_FILE), "rb") as fid:
            metadata = umsgpack.unpack(fid, encoding="utf-8")

        with open(path.join(self._lst_dir, self.split_name + ".txt"), "r") as fid:
            lst = fid.readlines()
        lst = set(line.strip() for line in lst)

        meta = metadata["meta"]
        images = [img_desc for img_desc in metadata["images"] if img_desc["id"] in lst]

        return meta, images

    def _load_item(self, item):
        scan_desc = self._images[item]

        scan_file = path.join(self._img_dir, scan_desc["id"])
        if path.exists(scan_file + ".bin"):
            scan_file = scan_file + ".bin"
        else:
            raise IOError("Cannot find any image for id {} in {}".format(img_desc["id"], self._img_dir))
        scan = LaserScan(project=True,
                       H=self.sensor_img_H,
                       W=self.sensor_img_W,
                       fov_up=self.sensor_fov_up,
                       fov_down=self.sensor_fov_down)
        scan.open_scan(scan_file)
        unproj_n_points = scan.points.shape[0]
        unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
        unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
        unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
        unproj_remissions = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)

        # get points and labels
        proj_range = torch.from_numpy(scan.proj_range).clone()
        proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
        proj_remission = torch.from_numpy(scan.proj_remission).clone()
        proj_mask = torch.from_numpy(scan.proj_mask)
        proj_x = torch.full([self.max_points], -1, dtype=torch.long)
        proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
        #proj_x[:unproj_n_points] = (torch.from_numpy(scan.proj_x)/2048)*4096
        proj_y = torch.full([self.max_points], -1, dtype=torch.long)
        proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)
        #proj_y[:unproj_n_points] = (torch.from_numpy(scan.proj_y)/64)*256
        proj = torch.cat([proj_range.unsqueeze(0).clone(),
                      proj_xyz.clone().permute(2, 0, 1),
                      proj_remission.unsqueeze(0).clone()])
#        print (proj.shape)
        proj = (proj - self.sensor_img_means
            ) / self.sensor_img_stds
        proj = proj * proj_mask.float()
        # Load all masks
        msk_file = path.join(self._msk_dir, scan_desc["id"] + ".png")
        msk = [Image.open(msk_file)]
        i = 1
        while path.exists("{}.{}".format(msk_file, i)):
            msk.append(Image.open("{}.{}".format(msk_file, i)))
            i += 1
        del proj_xyz, proj_remission,unproj_xyz, unproj_remissions, scan
        cat = scan_desc["cat"]
        iscrowd = scan_desc["iscrowd"]
        track_id = scan_desc["iscrowd"]
        return proj, msk, cat, track_id, iscrowd, proj_mask, proj_x, proj_y, proj_range, unproj_range, unproj_n_points ,scan_desc["id"]

    @property
    def categories(self):
        """Category names"""
        return self._meta["categories"]

    @property
    def num_categories(self):
        """Number of categories"""
        return len(self.categories)

    @property
    def num_stuff(self):
        """Number of "stuff" categories"""
        return self._meta["num_stuff"]

    @property
    def num_thing(self):
        """Number of "thing" categories"""
        return self.num_categories - self.num_stuff

    @property
    def original_ids(self):
        """Original class id of each category"""
        return self._meta["original_ids"]

    @property
    def palette(self):
        """Default palette to be used when color-coding semantic labels"""
        return np.array(self._meta["palette"], dtype=np.uint8)

    @property
    def img_sizes(self):
        """Size of each image of the dataset"""
        return [img_desc["size"] for img_desc in self._images]

    @property
    def img_categories(self):
        """Categories present in each image of the dataset"""
        return [img_desc["cat"] for img_desc in self._images]

    def __len__(self):
        return len(self._images)

    def __getitem__(self, item):
        #print ('item', item)
        img, msk, cat, track_id, iscrowd, proj_msk, p_x, p_y, proj_range, unproj_range, n_points, idx = self._load_item(item)
        rec = self.transform(img, msk, cat, track_id, iscrowd, proj_msk)
        size = (msk[0].size[1], msk[0].size[0])
        #print ('id', idx)  
#        print (size)
#        img.close()
        for m in msk:
            m.close()
        rec["p_x"] = p_x
        rec["p_y"] = p_y
        rec["proj_range"] = proj_range
        rec["unproj_range"] = unproj_range
        rec["n_points"] = n_points
        
        rec["idx"] = idx
        rec["size"] = size
        return rec

    def get_raw_image(self, idx):
        """Load a single, unmodified image with given id from the dataset"""
        img_file = path.join(self._img_dir, idx)
        if path.exists(img_file + ".png"):
            img_file = img_file + ".png"
        elif path.exists(img_file + ".jpg"):
            img_file = img_file + ".jpg"
        else:
            raise IOError("Cannot find any image for id {} in {}".format(idx, self._img_dir))

        return Image.open(img_file)

    def get_image_desc(self, idx):
        """Look up an image descriptor given the id"""
        matching = [img_desc for img_desc in self._images if img_desc["id"] == idx]
        if len(matching) == 1:
            return matching[0]
        else:
            raise ValueError("No image found with id %s" % idx)


class ISSTestDataset_(data.Dataset):
    _EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]

    def __init__(self, in_dir, transform):
        super(ISSTestDataset_, self).__init__()
        self.in_dir = in_dir
        self.transform = transform

        # Find all images
        self._images = []
        for img_path in chain(
                *(glob.iglob(path.join(self.in_dir, '**', ext), recursive=True) for ext in ISSTestDataset._EXTENSIONS)):
            _, name_with_ext = path.split(img_path)
            idx, _ = path.splitext(name_with_ext)

            with Image.open(img_path) as img_raw:
                size = (img_raw.size[1], img_raw.size[0])

            self._images.append({
                "idx": idx,
                "path": img_path,
                "size": size,
            })

    @property
    def img_sizes(self):
        """Size of each image of the dataset"""
        return [img_desc["size"] for img_desc in self._images]

    def __len__(self):
        return len(self._images)

    def __getitem__(self, item):
        # Load image
        with Image.open(self._images[item]["path"]) as img_raw:
            size = (img_raw.size[1], img_raw.size[0])
            img = self.transform(img_raw.convert(mode="RGB"))

        return {
            "img": img,
            "idx": self._images[item]["idx"],
            "size": size,
            "abs_path": self._images[item]["path"],
            "rel_path": path.relpath(self._images[item]["path"], self.in_dir),
        }
