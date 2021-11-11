import paddle
import paddle.nn as nn
import paddle.vision.models.vgg as vgg



import os
import sys
import time
import shutil
import hashlib
import tarfile
import zipfile
import requests
import os.path as osp
from tqdm import tqdm

PPGAN_HOME = os.path.expanduser(os.path.join('~', '.cache', 'ppgan'))

DOWNLOAD_RETRY_LIMIT = 3


def is_url(path):
    """
    Whether path is URL.
    Args:
        path (string): URL string or not.
    """
    return path.startswith('http://') or path.startswith('https://')


def _map_path(url, root_dir):
    # parse path after download under root_dir
    fname = osp.split(url)[-1]
    fpath = fname
    return osp.join(root_dir, fpath)


def get_path_from_url(url, md5sum=None, check_exist=True):
    """ Download from given url to root_dir.
    if file or directory specified by url is exists under
    root_dir, return the path directly, otherwise download
    from url and decompress it, return the path.

    Args:
        url (str): download url
        md5sum (str): md5 sum of download package

    Returns:
        str: a local path to save downloaded models & weights & datasets.
    """

    from paddle.distributed import ParallelEnv

    assert is_url(url), "downloading from {} not a url".format(url)
    root_dir = PPGAN_HOME
    # parse path after download to decompress under root_dir
    fullpath = _map_path(url, root_dir)

    if not (osp.exists(fullpath) and check_exist and _md5check(fullpath, md5sum)):
        if ParallelEnv().local_rank == 0:
            fullpath = _download(url, root_dir, md5sum)
        else:
            while not os.path.exists(fullpath):
                time.sleep(1)

    if ParallelEnv().local_rank == 0:
        if tarfile.is_tarfile(fullpath) or zipfile.is_zipfile(fullpath):
            fullpath = _decompress(fullpath)

    return fullpath


def _download(url, path, md5sum=None):
    """
    Download from url, save to path.

    url (str): download url
    path (str): download to given path
    """
    if not osp.exists(path):
        os.makedirs(path)

    fname = osp.split(url)[-1]
    fullname = osp.join(path, fname)
    retry_cnt = 0

    while not (osp.exists(fullname) and _md5check(fullname, md5sum)):
        if retry_cnt < DOWNLOAD_RETRY_LIMIT:
            retry_cnt += 1
        else:
            raise RuntimeError("Download from {} failed. "
                               "Retry limit reached".format(url))
        req = requests.get(url, stream=True)
        if req.status_code != 200:
            raise RuntimeError("Downloading from {} failed with code "
                               "{}!".format(url, req.status_code))

        # For protecting download interupted, download to
        # tmp_fullname firstly, move tmp_fullname to fullname
        # after download finished
        tmp_fullname = fullname + "_tmp"
        total_size = req.headers.get('content-length')
        with open(tmp_fullname, 'wb') as f:
            if total_size:
                with tqdm(total=(int(total_size) + 1023) // 1024) as pbar:
                    for chunk in req.iter_content(chunk_size=1024):
                        f.write(chunk)
                        pbar.update(1)
            else:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        shutil.move(tmp_fullname, fullname)
    return fullname


def _md5check(fullname, md5sum=None):
    if md5sum is None:
        return True

    md5 = hashlib.md5()
    with open(fullname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    calc_md5sum = md5.hexdigest()

    if calc_md5sum != md5sum:
        return False
    return True


def _decompress(fname):
    """
    Decompress for zip and tar file
    """

    # For protecting decompressing interupted,
    # decompress to fpath_tmp directory firstly, if decompress
    # successed, move decompress files to fpath and delete
    # fpath_tmp and remove download compress file.

    if tarfile.is_tarfile(fname):
        uncompressed_path = _uncompress_file_tar(fname)
    elif zipfile.is_zipfile(fname):
        uncompressed_path = _uncompress_file_zip(fname)
    else:
        raise TypeError("Unsupport compress file type {}".format(fname))

    return uncompressed_path


def _uncompress_file_zip(filepath):
    files = zipfile.ZipFile(filepath, 'r')
    file_list = files.namelist()

    file_dir = os.path.dirname(filepath)

    if _is_a_single_file(file_list):
        rootpath = file_list[0]
        uncompressed_path = os.path.join(file_dir, rootpath)

        for item in file_list:
            files.extract(item, file_dir)

    elif _is_a_single_dir(file_list):
        rootpath = os.path.splitext(file_list[0])[0].strip(os.sep).split(
            os.sep)[-1]
        uncompressed_path = os.path.join(file_dir, rootpath)

        for item in file_list:
            files.extract(item, file_dir)

    else:
        rootpath = os.path.splitext(filepath)[0].split(os.sep)[-1]
        uncompressed_path = os.path.join(file_dir, rootpath)
        if not os.path.exists(uncompressed_path):
            os.makedirs(uncompressed_path)
        for item in file_list:
            files.extract(item, os.path.join(file_dir, rootpath))

    files.close()

    return uncompressed_path


def _uncompress_file_tar(filepath, mode="r:*"):
    files = tarfile.open(filepath, mode)
    file_list = files.getnames()

    file_dir = os.path.dirname(filepath)

    if _is_a_single_file(file_list):
        rootpath = file_list[0]
        uncompressed_path = os.path.join(file_dir, rootpath)
        for item in file_list:
            files.extract(item, file_dir)
    elif _is_a_single_dir(file_list):
        rootpath = os.path.splitext(file_list[0])[0].split(os.sep)[-1]
        uncompressed_path = os.path.join(file_dir, rootpath)
        for item in file_list:
            files.extract(item, file_dir)
    else:
        rootpath = os.path.splitext(filepath)[0].split(os.sep)[-1]
        uncompressed_path = os.path.join(file_dir, rootpath)
        if not os.path.exists(uncompressed_path):
            os.makedirs(uncompressed_path)

        for item in file_list:
            files.extract(item, os.path.join(file_dir, rootpath))

    files.close()

    return uncompressed_path


def _is_a_single_file(file_list):
    if len(file_list) == 1 and file_list[0].find('/') < -1:
        return True
    return False


def _is_a_single_dir(file_list):
    file_name = file_list[0].split('/')[0]
    for i in range(1, len(file_list)):
        if file_name != file_list[i].split('/')[0]:
            return False
    return True


class PerceptualVGG(nn.Layer):
    """VGG network used in calculating perceptual loss.

    In this implementation, we allow users to choose whether use normalization
    in the input feature and the type of vgg network. Note that the pretrained
    path must fit the vgg type.

    Args:
        layer_name_list (list[str]): According to the name in this list,
            forward function will return the corresponding features. This
            list contains the name each layer in `vgg.feature`. An example
            of this list is ['4', '10'].
        vgg_tyep (str): Set the type of vgg network. Default: 'vgg19'.
        use_input_norm (bool): If True, normalize the input image.
            Importantly, the input feature must in the range [0, 1].
            Default: True.
        pretrained_url (str): Path for pretrained weights. Default:
    """
    def __init__(
            self,
            layer_name_list,
            vgg_type='vgg19',
            use_input_norm=True,
            pretrained_url='https://paddlegan.bj.bcebos.com/models/vgg19.pdparams'
    ):
        super(PerceptualVGG, self).__init__()

        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm

        # get vgg model and load pretrained vgg weight
        _vgg = getattr(vgg, vgg_type)()

        if pretrained_url:
            weight_path = get_path_from_url(pretrained_url)
            state_dict = paddle.load(weight_path)
            _vgg.load_dict(state_dict)
            print('PerceptualVGG loaded pretrained weight.')

        num_layers = max(map(int, layer_name_list)) + 1
        assert len(_vgg.features) >= num_layers

        # only borrow layers that will be used from _vgg to avoid unused params
        self.vgg_layers = nn.Sequential(
            *list(_vgg.features.children())[:num_layers])

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer(
                'mean',
                paddle.to_tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]))
            # the std is for image with range [-1, 1]
            self.register_buffer(
                'std',
                paddle.to_tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1]))

        for v in self.vgg_layers.parameters():
            v.trainable = False

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = {}

        for name, module in self.vgg_layers.named_children():
            x = module(x)
            if name in self.layer_name_list:
                output[name] = x.clone()
        return output

class PerceptualLoss(nn.Layer):
    """Perceptual loss with commonly used style loss.

    Args:
        layers_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'4': 1., '9': 1., '18': 1.}, which means the
            5th, 10th and 18th feature layer will be extracted with weight 1.0
            in calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplified by the
            weight. Default: 1.0.
        style_weight (flaot): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplified by the weight.
            Default: 1.0.
        norm_img (bool): If True, the image will be normed to [0, 1]. Note that
            this is different from the `use_input_norm` which norm the input in
            in forward fucntion of vgg according to the statistics of dataset.
            Importantly, the input image must be in range [-1, 1].
        pretrained (str): Path for pretrained weights. Default:

    """
    def __init__(
            self,
            layer_weights,
            vgg_type='vgg19',
            use_input_norm=True,
            perceptual_weight=1.0,
            style_weight=1.0,
            norm_img=True,
            pretrained='https://paddlegan.bj.bcebos.com/models/vgg19.pdparams',
            criterion='l1'):
        super(PerceptualLoss, self).__init__()
        # when loss weight less than zero return None
        if perceptual_weight <= 0 and style_weight <= 0:
            return None

        self.norm_img = norm_img
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = PerceptualVGG(layer_name_list=list(layer_weights.keys()),
                                 vgg_type=vgg_type,
                                 use_input_norm=use_input_norm,
                                 pretrained_url=pretrained)

        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        else:
            raise NotImplementedError(
                f'{criterion} criterion has not been supported in'
                ' this version.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        if self.norm_img:
            x = (x + 1.) * 0.5
            gt = (gt + 1.) * 0.5
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate preceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                percep_loss += self.criterion(
                    x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                style_loss += self.criterion(self._gram_mat(
                    x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (paddle.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            paddle.Tensor: Gram matrix.
        """
        (n, c, h, w) = x.shape
        features = x.reshape([n, c, w * h])
        features_t = features.transpose([1, 2])
        gram = features.bmm(features_t) / (c * h * w)
        return gram
