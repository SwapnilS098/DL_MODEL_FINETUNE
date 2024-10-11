"""
    In previous version of the  model fine tuning the
    layers were not freezed and the whole network's weights were
    changed while doing the fine tuning using the new small dataset.
"""

import argparse
import random
import shutil
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.zoo import models 
from compressai.models import FactorizedPrior
from compressai.losses import RateDistortionLoss
from compressai.optimizers import net_aux_optimizer
from compressai.datasets import ImageFolder
print("Modules are imported")


class AverageMeter:
    """Compute running average.
     -Used for monitoring metrics liek loss, accuracy
     and other performance indicators."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods.
        It inherits the functionality to multiple GPUs"""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def net_aux_optimizer_modified(net, conf, trainable_params):
    # Manually gather auxiliary parameters if they exist
    aux_params = [p for name, p in net.named_parameters() if 'aux' in name or 'entropy_bottleneck' in name]
    
    # Define the parameters dictionary for optimizers
    params_dict = {
        "net": trainable_params,  # Main trainable parameters
        "aux": aux_params  # Auxiliary parameters (entropy bottleneck or others)
    }

    union_params = set(params_dict["net"]) | set(params_dict["aux"])

    # Adjust the assertion to account only for trainable parameters
    assert len(union_params) - len(params_dict["net"]) == 0, "Parameter mismatch detected."

    optimizers = {}
    for k, v in params_dict.items():
        optimizers[k] = torch.optim.Adam(v, lr=conf[k]["lr"])

    return optimizers

def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer."""
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
        "aux": {"type": "Adam", "lr": args.aux_learning_rate},
    }

    # Filter parameters that require gradients
    trainable_params = [p for p in net.parameters() if p.requires_grad]
    print(f"Trainable parameters: {len(trainable_params)}")

    # Modify the net_aux_optimizer to use only trainable parameters
    optimizer = net_aux_optimizer_modified(net, conf, trainable_params)

    return optimizer["net"], optimizer["aux"]



def freeze_layers(net):
    """
    Freeze early encoder layers (g_a), keep later encoder layers,
    decoder (g_s), and entropy_bottleneck trainable.
    """
    # Freeze first Conv2D and GDN layers in the encoder
    for name, param in net.named_parameters():
        if 'g_a.0' in name or 'g_a.1' in name:  # Assuming first Conv2D and GDN are in indices 0, 1
            param.requires_grad = False
        else:
            param.requires_grad = True  # Unfreeze other layers
            
    # Keep decoder (g_s) and entropy_bottleneck trainable by default
    #for name, param in net.named_parameters():
    #    if 'g_s' in name or 'entropy_bottleneck' in name:
    #        param.requires_grad = True  # Ensure decoder and bottleneck are trainable
    #    # You can add further freezing logic here if needed

    # Ensure all layers that should be trainable are properly unfrozen
    for name, param in net.named_parameters():
        print(f"Layer {name} requires_grad={param.requires_grad}")


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train() #set the model in the training mode
    device = next(model.parameters()).device #sends the model parameters to the device (Device can be GPU or CPU)

    for i, d in enumerate(train_dataloader): #i is the index of the batch and d is the batch of training data
        d = d.to(device) #transfer the batch data to the device

        #before performing the forward pass on each batch
        optimizer.zero_grad()   #the gradients from previous batch are zeroed out
        aux_optimizer.zero_grad() #so that the gradients from different batches don't get mixed

        out_net = model(d) #forward pass using the model and d is input as the batch data
        #out_net is the output from the forward pass, which is the compressed representation of the image

        out_criterion = criterion(out_net, d) #criterion function calculates the loss based on the model's output 'out_net' and the
                                                # original data 'd'
        out_criterion["loss"].backward()  # the loss component is used for the backpropagation
        if clip_max_norm > 0:  #gradients are calculated with respect to this loss and propagated back through the network
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm) #gradient clipping is applied to provide stability during training
        optimizer.step()  #finally the model's parameters are updated using the calculated gradients

        aux_loss = model.aux_loss() #loss calculaion and updation of the weights for the auxiliary network
        aux_loss.backward()
        aux_optimizer.step()

        if i % 10 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )



def test_epoch(epoch, test_dataloader, model, criterion):
    """
    Function for evaluating the performance of the model on the
    test dataset.
    """
    model.eval() #set the function to the evaluation mode 
    device = next(model.parameters()).device #send the parameters to the device

    loss = AverageMeter() #instances are created for the losses
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter() 

    with torch.no_grad():  #disabling the gradient calculation 
        for d in test_dataloader: #d is the batch from the dataloader
            d = d.to(device) #transfer the data to the device
            out_net = model(d) #run the forward pass from it
            out_criterion = criterion(out_net, d) 

            aux_loss.update(model.aux_loss())  #for each batch the forward pass then the loss calculation
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    #returning the average total loss
    return loss.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "checkpoint_best_loss.pth.tar")




def parse_args(argv):
    parser = argparse.ArgumentParser(description="Training script for FactorizedPrior.")
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",


    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args








def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print("Device used:", device)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    model_name = "bmshj2018-factorized"
    quality = 4
    metric = "ms-ssim"
    net = models[model_name](quality=quality, metric=metric, pretrained=True)
    net = net.to(device)
    
    print(f"Total parameters: {sum(p.numel() for p in net.parameters())}")

    # Freeze layers as per recommendation
    freeze_layers(net)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda=args.lmbda)#, metric="ms-ssim")

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
            )

if __name__ == "__main__":
    argv = sys.argv[1:]  # Collect command-line arguments excluding the script name
    main(argv)

