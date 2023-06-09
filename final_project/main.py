import argparse
import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from models.resent import ResNet50
from dataloader import get_dataloader
from optimizers.optimizer import get_optimizer
from utils import get_timestamp, convert_to_scientific, get_progress_bar, TxtLogWriter

LOGS = './logs'
SAVED_MODEL = './saved_model'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '--data', default='cifar10')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--use_pretrained', '--use-pretrained', action='store_true')
    parser.add_argument('--lr', '--learning_rate', '--learning-rate', type=float, default=1e-3)
    parser.add_argument('--bs', '--batch_size', '--batch-size', type=int, default=1024)
    parser.add_argument('--opt', '--optimizer', default='adam')
    parser.add_argument('--extra', default='')
    args = parser.parse_args()
    return args

def train_one_epoch(model, 
                    train_loader,
                    optimizer,
                    criterion,
                    epoch: int,
                    num_epochs: int):
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    total_num_data = 0
    with get_progress_bar('Training', len(train_loader), epoch, num_epochs) as progress_bar:
        for images, labels in train_loader:
            
            images = images.to(device)
            labels = labels.to(device)
             
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Compute accuracy
            size_of_batch = len(labels)
            total_num_data = total_num_data + size_of_batch
            _ , predictions = torch.max(outputs.data, 1)
            accuracy = (predictions == labels).sum().item() / size_of_batch
            
            # Accumulate loss and accuracy
            loss = float(loss.item())
            total_loss = total_loss + loss * size_of_batch
            total_accuracy = total_accuracy + accuracy * size_of_batch
            
            # Compute average loss and accuracy
            avg_accuracy = total_accuracy / total_num_data
            avg_loss = total_loss / total_num_data
            
            # Update progress bar
            progress_bar.set_postfix(avg_accuracy = avg_accuracy,
                                    avg_loss = avg_loss)
            progress_bar.update()
    
    avg_accuracy = total_accuracy / total_num_data
    avg_loss = total_loss / total_num_data
    return avg_accuracy, avg_loss

def validation(model, val_loader):
    model.eval()
    total_accuracy = 0.0
    total_num_data = 0
    with get_progress_bar('Validation', len(val_loader), epoch, num_epochs) as progress_bar:
        with torch.no_grad():
            for images, labels in enumerate(val_loader):
                images = images.to(device)
                labels = labels.numpy()

                outputs = model(images)
                
                # Compute accuracy
                size_of_batch = len(labels)
                total_num_data = total_num_data + size_of_batch
                _ , predictions = torch.max(outputs.data, 1)
                accuracy = (predictions == labels).sum().item() / size_of_batch
                
                # Accumulate accuracy
                total_accuracy = total_accuracy + accuracy * size_of_batch
                
                # Compute average loss and accuracy
                avg_accuracy = total_accuracy / total_num_data
            
                progress_bar.set_postfix(avg_accuracy=avg_accuracy)
                progress_bar.update()
    avg_accuracy = total_accuracy / total_num_data
    return avg_accuracy

if __name__ == '__main__':
    args = get_args()
    dataset = args.dataset
    opt_name = args.opt
    num_epochs = args.epochs
    learning_rate = args.lr
    batch_size = args.bs
    extra_exp_name = args.extra
    use_pretrained = args.use_pretrained
    config = {'dataset': dataset,
              'optmizer': opt_name,
              'num_epochs': num_epochs,
              'learning_rate': learning_rate,
              'use_pretrained': use_pretrained,
              'batch_size': batch_size}
    
    # Initialize dataloader, model and loss    
    train_loader, val_loader, num_classes, input_dims = get_dataloader(dataset, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50(input_dims = input_dims,
                     use_pretrained = use_pretrained,
                    num_classes = num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, opt_name, lr=learning_rate)

    # Initialize the name of experiment
    timestamp = get_timestamp()
    exp_name = '{}_{}{}_bs{}_epoch{}'.format(timestamp, 
                                            opt_name, 
                                            convert_to_scientific(learning_rate, 'lr'),
                                            batch_size,
                                            num_epochs)
    if extra_exp_name != '':
        exp_name = exp_name + f'_{extra_exp_name}'
    
    # Initialize folders
    saving_folder = os.path.join(SAVED_MODEL, exp_name)
    log_saving_folder = os.path.join(saving_folder, LOGS)
    log_txt_path = os.path.join(saving_folder, 'log.txt')
    
    os.makedirs(log_saving_folder, exist_ok=True)
    writer = SummaryWriter(log_dir=log_saving_folder, flush_secs=60)
    txt_writer = TxtLogWriter(config, log_txt_path)
    
    best_val_acc = 0.0
    best_epoch = 0
    for epoch in range(num_epochs):
        loss_history = []
        total_num_data = 0
        total_accuracy = 0.0
        
        # Training
        avg_tain_acc, avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, epoch, num_epochs)
        writer.add_scalar('Train/loss', avg_loss, epoch)
        writer.add_scalar('Train/accuracy', avg_tain_acc, epoch)
        txt_writer.write_metric('train', 
                                epoch = epoch, 
                                avg_acc = avg_tain_acc, 
                                avg_loss = avg_loss)
        # Validation
        avg_val_acc = validation(model, val_loader)
        writer.add_scalar('Validation/accuracy', avg_val_acc, epoch)
        txt_writer.write_metric('validation', 
                                epoch = epoch, 
                                avg_acc = avg_val_acc)
        # Record best epoch and val accuracy
        if avg_val_acc >= best_val_acc:
            best_val_acc = avg_val_acc
            best_epoch = epoch
            # Save model and optimizer
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, os.path.join(saving_folder, f'best.pth'))
        print()
    
    writer.close()
    txt_writer.write_best_metric(best_epoch, best_accuracy = best_val_acc)
    print('Best epoch = {:3d}, Best Validation Accuracy = {:.3f}'.format(best_epoch, best_val_acc))