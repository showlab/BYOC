"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data

from exp2bs.data.base_dataset import BaseDataset


# from data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "exp2bs.data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)

        # HDFS single process
        # self.dataloader = torch.utils.data.DataLoader(
        #     self.dataset,
        #     batch_size=opt.batch_size,
        #     shuffle=not opt.serial_batches,
        #     # num_workers=int(opt.num_threads)
        #     num_workers=0
        # )

        hdfs_flag = False
        for key in vars(opt).keys():
            if 'hdfs' in str(key).lower():
                hdfs_flag = True
                break
        if hdfs_flag:
            # HDFS multi process
            if opt.serial_batches:
                HDFS_batch_sampler = KVSampler(dataset=self.dataset,batch_size=1,
                                               num_replicas=1, rank=0, shuffle=not opt.serial_batches)
            else:
                HDFS_batch_sampler = KVSampler(dataset=self.dataset, batch_size=opt.batch_size,
                                               num_replicas=1, rank=0, shuffle=not opt.serial_batches)
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=None,
                sampler=HDFS_batch_sampler,
                num_workers=int(opt.num_threads),
                worker_init_fn=worker_init_fn
            )
        else:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                shuffle=not opt.serial_batches,
                num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data


# The following code is wrote specifically for HDFS multi-processes loading data
def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset

    # for cycle_gan model
    # dataset.readerA = KVReader(dataset.pathA, 32)
    # dataset.readerB = KVReader(dataset.pathB, 32)

    # added for face2face
    if 'xigua' in dataset.name().lower() or 'swap' in dataset.name().lower():
        print('init two HDFS readers')
        dataset.readerA = KVReader(dataset.pathA, 32)
        dataset.readerB = KVReader(dataset.pathB, 32)
    else:
        dataset.reader = KVReader(dataset.hdfs_path, 32)


def chunk(iterable, chunk_size):
    ret = []
    for record in iterable:
        ret.append(record)
        if len(ret) == chunk_size:
            yield ret
            ret = []
    if ret:
        yield ret


class KVSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset, batch_size, num_replicas, rank, shuffle=True):
        super(KVSampler, self).__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.batch_size = batch_size

    def __iter__(self):
        # 这里返回一个batch_size的索引，提供给dataset中的reader
        iterable = super(KVSampler, self).__iter__()
        return chunk(iterable, self.batch_size)
