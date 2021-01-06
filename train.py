import argparse
import torch
import torch.backends.cudnn as cudnn
from trainer import MUNIT_Trainer
from utils.utils import get_all_data_loaders, prepare_sub_folder, print_train_info, write_html, write_loss, get_config, write_images, Timer
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import random
import numpy as np

seed_value = 29
np.random.seed(seed_value) # cpu vars
random.seed(seed_value) # Python
torch.manual_seed(seed_value) # cpu  vars
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value) #gpu vars


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='./datasets/', help='path to dataset folder')
parser.add_argument('--config', type=str, default='./configs/configs.yaml', help='path to the config file.')
parser.add_argument('--output_path', type=str, default='./results', help='outputs path')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--trainer', type=str, default='MUNIT', help='MUNIT')
parser.add_argument('--name', type=str, default='munit_semantic_loss', help='name of the experiment')

opts = parser.parse_args()
cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['data_root'] = opts.data_root


# Setup model and data loader
if opts.trainer == 'MUNIT':
    trainer = MUNIT_Trainer(config, opts)
else:
    sys.exit("Only support MUNIT")
trainer.cuda()

train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)

data_masks = [train_loader_a.dataset[i] for i in range(config['display_size'])]
train_display_images_a = torch.stack([dm[0] for dm in data_masks]).cuda()
train_display_target_a = torch.stack([dm[1] for dm in data_masks]).cuda()

data_masks = [train_loader_b.dataset[i] for i in range(config['display_size'])]
train_display_images_b = torch.stack([dm[0] for dm in data_masks]).cuda()
train_display_target_b = torch.stack([dm[1] for dm in data_masks]).cuda()

data_masks = [test_loader_a.dataset[i] for i in range(config['display_size'])]
test_display_images_a = torch.stack([dm[0] for dm in data_masks]).cuda()
test_display_target_a = torch.stack([dm[1] for dm in data_masks]).cuda()

data_masks = [test_loader_b.dataset[i] for i in range(config['display_size'])]
test_display_images_b = torch.stack([dm[0] for dm in data_masks]).cuda()
test_display_target_b = torch.stack([dm[1] for dm in data_masks]).cuda()

# Setup logger and output folders
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + '/logs'))
output_directory = os.path.join(opts.output_path + '/outputs')
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
print_train_info(opts, config, trainer)

# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
while True:
    for it, (images_target_a, images_target_b) in enumerate(zip(train_loader_a, train_loader_b)):
        trainer.update_learning_rate()
        images_a = images_target_a[0].cuda().detach()
        target_a = images_target_a[1].cuda().detach()
        images_b = images_target_b[0].cuda().detach()
        target_b = images_target_b[1].cuda().detach()
        ids_b = images_target_b[2]
        # Main training code
        with Timer("Elapsed time in update: %f"):
            trainer.dis_update(images_a, images_b, config)
            trainer.gen_update(images_a, images_b, config, target_a, iterations)

            if iterations >= config['train_seg_iters']:
                trainer.seg_update(images_a, images_b, target_a, target_b)

        # Dump training stats in log file
        if (iterations) % config['log_iter'] == 0:
            print('Iteration: %08d/%08d' % (iterations, max_iter))
            losses = write_loss(iterations, trainer, train_writer)

        # Write images
        if (iterations) % config['image_save_iter'] == 0:
            with torch.no_grad():
                train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
                test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b)

            write_images(train_image_outputs, train_display_target_a, train_display_target_b, image_directory, 'train_%08d' % (iterations), config)
            write_images(test_image_outputs, test_display_target_a, test_display_target_b, image_directory, 'test_%08d' % (iterations), config)
            # HTML
            write_html(output_directory, iterations, opts, config, 'train')
            write_html(output_directory, iterations, opts, config, 'test')

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')
