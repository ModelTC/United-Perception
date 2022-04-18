import os
import cv2
from PIL import Image
import numpy as np
from up.utils.general.registry_factory import IMAGE_READER_REGISTRY
from up.utils.general.petrel_helper import PetrelHelper


__all__ = ['FileSystemCVReader', 'FileSystemPILReader']


def get_cur_image_dir(image_dir, idx):
    if isinstance(image_dir, list) or isinstance(image_dir, tuple):
        assert idx < len(image_dir)
        return image_dir[idx]
    return image_dir


class ImageReader(object):
    def __init__(self, image_dir, color_mode, memcached=None):
        super(ImageReader, self).__init__()
        if image_dir == '/' or image_dir == '//':
            image_dir = ''
        self.image_dir = image_dir
        self.color_mode = color_mode

    def image_directory(self):
        return self.image_dir

    def image_color(self):
        return self.color_mode

    def hash_filename(self, filename):
        import hashlib
        md5 = hashlib.md5()
        md5.update(filename.encode('utf-8'))
        hash_filename = md5.hexdigest()
        return hash_filename

    def read(self, filename):
        return self.fs_read(filename)

    def __call__(self, filename, image_dir_idx=0):
        if filename.startswith("//"):
            filename = filename[1:]
        image_dir = get_cur_image_dir(self.image_dir, image_dir_idx)
        filename = os.path.join(image_dir, filename)
        img = self.read(filename)
        return img


@IMAGE_READER_REGISTRY.register('fs_opencv')
class FileSystemCVReader(ImageReader):
    def __init__(self, image_dir, color_mode, memcached=None, to_float32=False):
        super(FileSystemCVReader, self).__init__(image_dir, color_mode, memcached)
        assert color_mode in ['RGB', 'BGR', 'GRAY'], '{} not supported'.format(color_mode)
        if color_mode == 'RGB':
            self.cvt_color = getattr(cv2, 'COLOR_BGR2{}'.format(color_mode))
        else:
            self.cvt_color = None
        self.to_float32 = to_float32
        self.memcached = memcached
        if memcached:
            self.initialized = False

    def fs_read(self, filename):
        assert os.path.exists(filename), filename
        if self.memcached:
            import mc
            self._init_memcached()
            value = mc.pyvector()
            assert len(filename) < 250, 'memcached rquires length of path < 250'
            self.mclient.Get(filename, value)
            value_buf = mc.ConvertBuffer(value)
            img_array = np.frombuffer(value_buf, np.uint8)
            if self.color_mode == 'GRAY':
                img = cv2.imdecode(img_array, 0)
            else:
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            if self.color_mode == 'GRAY':
                img = cv2.imread(filename, 0)
            else:
                img = cv2.imread(filename, cv2.IMREAD_COLOR)
        if self.color_mode == 'RGB':
            img = cv2.cvtColor(img, self.cvt_color)
        if self.to_float32:
            img = img.astype(np.float32)
        return img

    def _init_memcached(self):
        if not self.initialized:
            import mc
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def fake_image(self, *size):
        if len(size) == 0:
            if self.color_mode == 'GRAY':
                size = (512, 512, 1)
            else:
                size = (512, 512, 3)
        return np.zeros(size, dtype=np.uint8)


@IMAGE_READER_REGISTRY.register('fs_pillow')
class FileSystemPILReader(ImageReader):
    def __init__(self, image_dir, color_mode, memcached=None):
        super(FileSystemPILReader, self).__init__(image_dir, color_mode, memcached)
        assert color_mode == 'RGB', 'only RGB mode supported for pillow for now'

    def fake_image(self, *size):
        if len(size) == 0:
            size = (512, 512, 3)
        return Image.new(self.color_mode, size)

    def fs_read(self, filename):
        assert os.path.exists(filename), filename
        img = Image.open(filename).convert(self.color_mode)
        return img


@IMAGE_READER_REGISTRY.register('ceph_opencv')
class CephSystemCVReader(ImageReader):
    def __init__(self, image_dir, color_mode, memcached=True,
                 conf_path=PetrelHelper.default_conf_path, default_cluster=''):
        super(CephSystemCVReader, self).__init__(image_dir, color_mode)
        self.image_dir = image_dir
        self.color_mode = color_mode
        assert color_mode in ['RGB', 'BGR', 'GRAY'], '{} not supported'.format(color_mode)
        if color_mode != 'BGR':
            self.cvt_color = getattr(cv2, 'COLOR_BGR2{}'.format(color_mode))
        else:
            self.cvt_color = None
        self.conf_path = os.path.expanduser(conf_path)
        self.memcached = memcached
        self.initialized = False
        self.default_cluster = default_cluster

    def image_directory(self):
        return self.image_dir

    def image_color(self):
        return self.color_mode

    def ceph_join(self, root, filename):
        if 's3://' in filename:
            abs_filename = filename
        else:
            abs_filename = os.path.join(root, filename)

        if abs_filename.startswith('s3://') and self.default_cluster:
            abs_filename = self.default_cluster + ':' + abs_filename
        return abs_filename

    def bytes_to_img(self, value):
        img_array = np.frombuffer(value, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        assert img is not None
        return img

    def __call__(self, filename, image_dir_idx=0):
        image_dir = get_cur_image_dir(self.image_dir, image_dir_idx)
        filename = self.ceph_join(image_dir, filename)
        if not self.initialized:
            self._init_memcached()
        try:
            value = self.mclient.Get(filename)
            assert value is not None, filename
            img = self.bytes_to_img(value)
        except Exception as e:  # noqa
            value = self.mclient.Get(filename, update_cache=True)
            assert value is not None, filename
            img = self.bytes_to_img(value)
        if self.color_mode != 'BGR':
            img = cv2.cvtColor(img, self.cvt_color)
        return img

    def _init_memcached(self):
        if not self.initialized:
            from petrel_client.client import Client
            # from petrel_client.mc_client import py_memcache as mc
            self.mclient = Client(enable_mc=self.memcached, conf_path=self.conf_path)
            self.initialized = True


@IMAGE_READER_REGISTRY.register('osg')
class OSGReader(ImageReader):
    def __init__(self, osg_server_url, color_mode):
        from spring_sdk import OSG
        self.client = OSG(osg_server_url, secure=False)
        self.color_mode = color_mode
        assert color_mode in ['RGB', 'BGR', 'GRAY'], '{} not supported'.format(color_mode)
        if color_mode != 'BGR':
            self.cvt_color = getattr(cv2, 'COLOR_BGR2{}'.format(color_mode))
        else:
            self.cvt_color = None

    def __call__(self, bucket, key):
        """
        Arguments:
            index: (bucket, key) pair
        """
        img_str = self.client.get_object(bucket, key)
        img = cv2.imdecode(np.fromstring(img_str, np.uint8), 1)
        if self.color_mode != 'BGR':
            img = cv2.cvtColor(img, self.cvt_color)

        return img


def build_image_reader(cfg_reader):
    return IMAGE_READER_REGISTRY.build(cfg_reader)
