import torch
import torch.nn as nn
import os
import argparse

from datasets import get_images, get_dataset, get_data_loaders
from engine import train, validate
from segmentation_model import faster_vit_0_any_res
from config import ALL_CLASSES, LABEL_COLORS_LIST
from utils import save_model, SaveBestModel, save_plots, SaveBestModelIOU
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument(
    '--epochs',
    default=10,
    help='number of epochs to train for',
    type=int
)
parser.add_argument(
    '--lr',
    default=0.0001,
    help='learning rate for optimizer',
    type=float
)
parser.add_argument(
    '--batch',
    default=4,
    help='batch size for data loader',
    type=int
)
parser.add_argument(
    '--imgsz', 
    default=[512, 512],
    type=int,
    nargs='+',
    help='width, height'
)
parser.add_argument(
    '--scheduler',
    action='store_true',
)
args = parser.parse_args()
print(args)

if __name__ == '__main__':
    # Create a directory with the model name for outputs.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, 'outputs')
    out_dir_valid_preds = os.path.join(out_dir, 'valid_preds')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_valid_preds, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(out_dir, 'tb'))
    writer.flush()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = faster_vit_0_any_res(pretrained=True, resolution=args.imgsz).to(device)
    print(model)

    # classes to train and adjust final head BEFORE optimizer
    classes_to_train = ['background', 'bicycle', 'bus', 'car', 'motorbike', 'person', 'train']
    model.upsample_and_classify[13] = nn.Conv2d(
        512, len(classes_to_train), kernel_size=(1, 1), stride=(1, 1)
    ).to(device)

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    def get_images(root_path):
        with open(os.path.join(root_path, "ImageSets/Segmentation/train.txt")) as f:
            train_ids = f.read().splitlines()
        with open(os.path.join(root_path, "ImageSets/Segmentation/val.txt")) as f:
            val_ids = f.read().splitlines()

        train_images = [os.path.join(root_path, "JPEGImages", i + ".jpg") for i in train_ids]
        train_masks  = [os.path.join(root_path, "SegmentationClass", i + ".png") for i in train_ids]
        val_images   = [os.path.join(root_path, "JPEGImages", i + ".jpg") for i in val_ids]
        val_masks    = [os.path.join(root_path, "SegmentationClass", i + ".png") for i in val_ids]

        return train_images, train_masks, val_images, val_masks

    train_images, train_masks, valid_images, valid_masks = get_images(
        root_path='/Users/ortalhanuna/my-code/VOCdevkit/VOC2012'    
    )

    # classes_to_train is defined above

    train_dataset, valid_dataset = get_dataset(
        train_images, 
        train_masks,
        valid_images,
        valid_masks,
        ALL_CLASSES,
        classes_to_train,
        LABEL_COLORS_LIST,
        img_size=args.imgsz
    )

    train_dataloader, valid_dataloader = get_data_loaders(
        train_dataset, valid_dataset, batch_size=args.batch
    )

    # Initialize `SaveBestModel` class.
    save_best_model = SaveBestModel()
    save_best_iou = SaveBestModelIOU()
    # LR Scheduler.
    scheduler = MultiStepLR(
        optimizer, milestones=[30], gamma=0.1
    )
    EPOCHS = args.epochs
    train_loss, train_pix_acc, train_miou = [], [], []
    valid_loss, valid_pix_acc, valid_miou = [], [], []
    global_step = 0
    for epoch in range (EPOCHS):
        print(f"EPOCH: {epoch + 1}")
        train_epoch_loss, train_epoch_pixacc, train_epoch_miou, global_step = train(
            model,
            train_dataloader,
            device,
            optimizer,
            criterion,
            classes_to_train,
            writer=writer,
            global_step=global_step
        )
        valid_epoch_loss, valid_epoch_pixacc, valid_epoch_miou = validate(
            model,
            valid_dataset,
            valid_dataloader,
            device,
            criterion,
            classes_to_train,
            LABEL_COLORS_LIST,
            epoch,
            ALL_CLASSES,
            save_dir=out_dir_valid_preds,
            writer=writer,
            tb_tag='Images/val_overlay'
        )
        train_loss.append(train_epoch_loss)
        train_pix_acc.append(train_epoch_pixacc)
        train_miou.append(train_epoch_miou)
        valid_loss.append(valid_epoch_loss)
        valid_pix_acc.append(valid_epoch_pixacc)
        valid_miou.append(valid_epoch_miou)

        # TensorBoard scalars
        writer.add_scalar('Loss/train', train_epoch_loss, epoch)
        writer.add_scalar('Loss/valid', valid_epoch_loss, epoch)
        writer.add_scalar('Accuracy/pixel_train', train_epoch_pixacc, epoch)
        writer.add_scalar('Accuracy/pixel_valid', valid_epoch_pixacc, epoch)
        writer.add_scalar('mIoU/train', train_epoch_miou, epoch)
        writer.add_scalar('mIoU/valid', valid_epoch_miou, epoch)
        writer.flush()

        save_best_model(
            valid_epoch_loss, epoch, model, out_dir, name='model_loss'
        )
        save_best_iou(
            valid_epoch_miou, epoch, model, out_dir, name='model_iou'
        )

        print(
            f"Train Epoch Loss: {train_epoch_loss:.4f},",
            f"Train Epoch PixAcc: {train_epoch_pixacc:.4f},",
            f"Train Epoch mIOU: {train_epoch_miou:4f}"
        )
        print(
            f"Valid Epoch Loss: {valid_epoch_loss:.4f},", 
            f"Valid Epoch PixAcc: {valid_epoch_pixacc:.4f}",
            f"Valid Epoch mIOU: {valid_epoch_miou:4f}"
        )
        if args.scheduler:
            scheduler.step()
        print('-' * 50)

    save_model(EPOCHS, model, optimizer, criterion, out_dir, name='model')
    # Save the loss and accuracy plots.
    save_plots(
        train_pix_acc, valid_pix_acc, 
        train_loss, valid_loss,
        train_miou, valid_miou, 
        out_dir
    )
    writer.close()
    print('TRAINING COMPLETE')