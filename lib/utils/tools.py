import torch
import os, re
import logging
import time
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

""" Check Device and Path for saving and loading """


def check_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def find_latest_ckpt(folder):
    """ find latest checkpoint """
    files = []
    for fname in os.listdir(folder):
        s = re.findall(r'\d+', fname)
        if len(s) == 1:
            files.append((int(s[0]), fname))
    if files:
        file = max(files)[1]
        file_name = os.path.splitext(file)[0]
        previous_iter = int(file_name.split("_")[1])
        return file, previous_iter
    else:
        return None, 0


""" Training Tool for model """


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def apply_gradients(loss, optim):
    optim.zero_grad()
    loss.backward()
    optim.step()


def infinite_iterator(loader):
    while True:
        for batch in loader:
            yield batch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cross_entroy_loss(logit, label):
    loss = torch.nn.CrossEntropyLoss()(logit, label)
    return loss


def accuracy(outputs, label):
    """ if you want to make custom accuracy for your model, you need to implement this function."""
    y = torch.argmax(outputs, dim=1)
    return (y.eq(label).sum())


def reduce_loss(tmp):
    """ will implement reduce_loss func """
    loss = tmp
    return loss

""" Logger """
def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print("=> creating {}".format(root_output_dir))
        root_output_dir.mkdir()
    dataset = cfg.DATASET.DATASET + "_" + cfg.DATASET.HYBRID_JOINTS_TYPE \
        if cfg.DATASET.HYBRID_JOINTS_TYPE else cfg.DATASET.DATASET
    
    dataset = dataset.replace(":", "_")
    
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]
    
    final_output_dir = root_output_dir / dataset / model / cfg_name
    
    if cfg.RANK == 0:
        print("=> creating {}", format(final_output_dir))
        final_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        pass
    
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
        (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    
    return logger, str(final_output_dir), str(tensorboard_log_dir)

def visualize_inference(img, label, batch_size):
    """ Visualize Image Batch"""
    fig, axes = plt.subplots(1, batch_size, figsize=(10, 10))
    for i in range(batch_size):
        axes[i].imshow(np.squeeze(img[i]), cmap="gray")
        axes[i].set_title(f"predicted: {label[i]}")
        axes[i].axis('off')
    plt.show()


def visualize_feature_map(model, image):
    model.network.eval()

    # transformer
    # transform = mnist_transform()

    feature_map = None

    # input_tensor
    input_tensor = image.to(check_device())

    def hook(module, input, output):
        nonlocal feature_map
        feature_map = output.detach().cpu()

    target_layer = model.network.layers[6]
    hook_handle = target_layer.register_forward_hook(hook)

    with torch.no_grad():
        model.network(input_tensor)

    hook_handle.remove()

    plt.figure(figsize=(12, 8))
    for i in range(feature_map.size(1)):
        plt.subplot(4, 8, i + 1)
        plt.imshow(feature_map[0, i], cmap='viridis')
        plt.axis('off')
    plt.show()


def show_img(img):
    """ Display an img"""
    img = np.array(img, dtype=np.uint8)
    img = Image.fromarray(img)
    img.show()


def data_transform(img_size):
    transform_list = [
        transforms.Resize(size=[img_size, img_size]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),  # [0, 255] -> [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),  # [0, 1] -> [-1, 1]
    ]
    return transforms.Compose(transform_list)


def mnist_transform():
    transforms_list = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ]
    return transforms.Compose(transforms_list)
