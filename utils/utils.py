import math
import time
from collections import OrderedDict
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import skimage.measure as measure
import torch.nn.init as init
import torchvision.transforms as standard_transforms
import yaml
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from data.data_loader import *
from data.transforms import dataToTensor, MaskToTensor
from utils import html

irange = range


def get_all_data_loaders(conf):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']

    # load min and max value per channel for domain a
    max_val_a = np.load(conf['data_root'] + '/max_val_a.npy')
    min_val_a = np.load(conf['data_root'] + '/min_val_a.npy')
    # load min and max value per channel for domain b
    max_val_b = np.load(conf['data_root'] + '/max_val_b.npy')
    min_val_b = np.load(conf['data_root'] + '/min_val_b.npy')
    if 'data_root' in conf:
        train_loader_a = get_data_loader(os.path.join(conf['data_root'], 'trainA'), batch_size, True,
                                                num_workers, conf['input_dim_a'], max_val_a, min_val_a)

        test_loader_a = get_data_loader(os.path.join(conf['data_root'], 'testA'), batch_size, False, num_workers,
                                               conf['input_dim_a'], max_val_a, min_val_a)

        train_loader_b = get_data_loader(os.path.join(conf['data_root'], 'trainB'), batch_size, True,
                                                num_workers, conf['input_dim_b'], max_val_b, min_val_b)

        test_loader_b = get_data_loader(os.path.join(conf['data_root'], 'testB'), batch_size, False, num_workers,
                                               conf['input_dim_b'], max_val_b, min_val_b)

    return train_loader_a, train_loader_b, test_loader_a, test_loader_b


def get_data_loader(items, batch_size, train, num_workers=4, channels=20, max_value=1, min_value=0):

    mean_val = [0.5] * channels
    std_val = [0.5] * channels

    transform_list = standard_transforms.Compose([dataToTensor(), standard_transforms.Normalize(mean_val, std_val)])
    target_transform = MaskToTensor()

    dataset = MRI(items, max_value, min_value, transform=transform_list, target_transform=target_transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=train, num_workers=num_workers)

    return loader


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


def plot_contour(target):
    grid_label = make_grid(target.unsqueeze(1), nrow=5, padding=5, normalize=False)
    grid_label = np.transpose(grid_label.cpu(), (1, 2, 0))

    grid_label = grid_label[:, :, 0]

    contours = measure.find_contours(grid_label, 0.5)
    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], c='red', linewidth=1)


def write_images(image_outputs, target_a, target_b, image_directory, postfix, hyperpars):
    results_dir_ab = (image_directory + '/iter_' + postfix + '/images_a2b/')
    if not os.path.isdir(results_dir_ab):
        os.makedirs(results_dir_ab)

    results_dir_ba = (image_directory + '/iter_' + postfix + '/images_b2a/')
    if not os.path.isdir(results_dir_ba):
        os.makedirs(results_dir_ba)

    for i in range(len(image_outputs) // 2):
        grid_img = make_grid(image_outputs[i][:, 0, :, :].unsqueeze(1), nrow=5, padding=5, normalize=True)
        grid_img = np.transpose(grid_img.cpu(), (1, 2, 0))
        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(grid_img[:, :, 0], cmap=cm.gray)
        plot_contour(target_a)
        plt.savefig(results_dir_ab + hyperpars['outputs'][i] + '.png',  bbox_inches='tight')
        plt.close()

    for i in range(len(image_outputs) // 2):
        grid_img = make_grid(image_outputs[len(image_outputs) // 2 + i][:, 0, :, :].unsqueeze(1), nrow=5, padding=5, normalize=True)
        grid_img = np.transpose(grid_img.cpu(), (1, 2, 0))
        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(grid_img[:, :, 0], cmap=cm.gray)
        plot_contour(target_b)
        plt.savefig(results_dir_ba + hyperpars['outputs'][i] + '.png', bbox_inches='tight')
        plt.close()


def write_html(filename, iterations, opts, hyperpars, set):
    webpage = html.HTML(filename + '/index_a2b_' + set + '.html', 'Experiment name = %s' % opts.name, reflesh=1)
    for j in range(iterations, 0, -int(hyperpars['image_save_iter'])):
        ims, txts, links = [], [], []
        webpage.add_header('iteration %d' % (j))
        for label in hyperpars['outputs']:
            img_path = 'iter_' + set + '_%08d/images_a2b/%s.png' % (j, label)
            ims.append(img_path)
            txts.append(label)
            links.append(img_path)
        webpage.add_images(ims, txts, links, width=256)
    webpage.save()

    webpage = html.HTML(filename + '/index_b2a_' + set + '.html', 'Experiment name = %s' % opts.name, reflesh=1)
    for j in range(iterations, 0, -int(hyperpars['image_save_iter'])):
        ims, txts, links = [], [], []
        webpage.add_header('iteration %d' % (j))
        for label in hyperpars['outputs']:
            img_path = 'iter_' + set + '_%08d/images_b2a/%s.png' % (j, label)
            ims.append(img_path)
            txts.append(label)
            links.append(img_path)
        webpage.add_images(ims, txts, links, width=256)
    webpage.save()


def make_grid(tensor, nrow=8, padding=2, normalize=True, range=None, scale_each=False, pad_value=0):
    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = tensor

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new_full((tensor.size(1), height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid


def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer) \
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and ('loss' in attr or 'grad' in attr or 'nwd' in attr)]
    errors_ret = OrderedDict()
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)
        if isinstance(m, str):
            errors_ret[m] = float(getattr(trainer, m))

# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def get_scheduler(optimizer, lr_policy, hyperparameters=None, iterations=-1):
    if lr_policy is None or lr_policy == 'constant':
        scheduler = None # constant scheduler
    elif lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))


def print_train_info(opt, config, nets):
    args = vars(opt)
    # save to the disk
    expr_dir = opt.output_path
    mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, 'train_info.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(args.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')
        opt_file.write('------------ Config -------------\n')
        for k, v in sorted(config.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')

        names = ['gen_a', 'gen_b', 'dis_a', 'dis_b', 'seg']
        opt_file.write('------------ Networks -------------\n')
        for name in names:
            if hasattr(nets, name):
                net = getattr(nets, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6), file=opt_file)
        print(nets, file=opt_file)
        opt_file.write('-------------- End ----------------\n')

    return


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)
