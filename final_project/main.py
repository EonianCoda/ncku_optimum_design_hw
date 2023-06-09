import argparse
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score
from models.resent import ResNet50
from dataloader import get_dataloader
from optimizers.optimizer import get_optimizer
from utils import get_timestamp, convert_to_scientific, TxtLogWriter

LOGS = './logs'
SAVED_MODEL = './saved_model'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '--data', default='cifar10')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', '--learning_rate', '--learning-rate', type=float, default=1e-3)
    parser.add_argument('--bs', '--batch_size', '--batch-size', type=int, default=1024)
    parser.add_argument('--opt', '--optimizer', default='adam')
    parser.add_argument('--extra', default='')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    dataset = args.dataset
    opt_name = args.opt
    num_epochs = args.epochs
    learning_rate = args.lr
    batch_size = args.bs
    extra_exp_name = args.extra

    config = {'dataset': dataset,
              'optmizer': opt_name,
              'num_epochs': num_epochs,
              'learning_rate': learning_rate,
              'batch_size': batch_size}
    
    train_loader, test_loader, num_classes = get_dataloader(dataset, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50(num_classes = num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, opt_name, lr=learning_rate)

    
    timestamp = get_timestamp()
    exp_name = '{}_{}{}_bs{}_epoch{}'.format(timestamp, 
                                opt_name, 
                                convert_to_scientific(learning_rate, 'lr'),
                                batch_size,
                                num_epochs)
    if extra_exp_name != '':
        exp_name = exp_name + f'_{extra_exp_name}'
    
    saving_folder = os.path.join(SAVED_MODEL, exp_name)
    log_saving_folder = os.path.join(saving_folder, LOGS)
    log_txt_path = os.path.join(saving_folder, 'log.txt')
    
    os.makedirs(log_saving_folder, exist_ok=True)
    writer = SummaryWriter(log_dir=log_saving_folder, flush_secs=60)
    txt_writer = TxtLogWriter(config, log_txt_path)
    
    
    total_training_steps = len(train_loader)
    total_validation_step = len(test_loader)
    best_acc = 0.0
    best_epoch = 0
    for epoch in range(num_epochs):
        loss_history = []
        num_data = 0
        total_accuracy = 0.0
        
        model.train()
        # Training
        progress_bar = tqdm(total=len(train_loader), desc=f"Training   Epoch {epoch+1}/{num_epochs}", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, predictions = torch.max(outputs.data, 1)
            predictions = predictions.cpu().numpy()
            labels = labels.cpu().numpy()
            num_data = num_data + labels.shape[0]
            
            # Compute loss
            loss = float(loss.item())
            loss_history.append(loss * labels.shape[0])
            avg_loss = sum(loss_history) / num_data
            # Compute Accuracy
            total_accuracy = total_accuracy + accuracy_score(labels, predictions) * labels.shape[0]
            avg_accuracy = total_accuracy / num_data
            
            # Update progress bar
            progress_bar.set_postfix(avg_accuracy=avg_accuracy,
                                    avg_loss = avg_loss)
            progress_bar.update()
        progress_bar.close()
        
        
        avg_loss = sum(loss_history) / num_data
        avg_accuracy = total_accuracy / num_data
        writer.add_scalar('Train/loss', avg_loss, epoch)
        writer.add_scalar('Train/accuracy', avg_accuracy, epoch)
        txt_writer.write_metric('train', 
                                epoch = epoch, 
                                avg_accuracy = avg_accuracy, 
                                avg_loss = avg_loss)
        
        model.eval()
        num_data = 0
        total_recall = 0.0
        total_precision = 0.0
        total_f1_score = 0.0
        total_accuracy = 0.0
        with torch.no_grad():
            progress_bar = tqdm(total=len(test_loader), desc=f"Validation Epoch {epoch+1}/{num_epochs}", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            for i, (images, labels) in enumerate(test_loader):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predictions = torch.max(outputs.data, 1)
                predictions = predictions.cpu().numpy()
                labels = labels.cpu().numpy()
                num_data = num_data + labels.shape[0]
                
                total_accuracy = total_accuracy + accuracy_score(labels, predictions) * labels.shape[0]
                avg_accuracy = total_accuracy / num_data
                progress_bar.set_postfix(avg_accuracy=avg_accuracy)
                progress_bar.update()
        progress_bar.close()
        avg_accuracy = total_accuracy / num_data
        writer.add_scalar('Validation/accuracy', avg_accuracy, epoch)
        txt_writer.write_metric('validation', 
                                epoch = epoch, 
                                avg_accuracy = avg_accuracy)
        if avg_accuracy >= best_acc:
            best_acc = avg_accuracy
            best_epoch = epoch
            # Save model and optimizer
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, os.path.join(saving_folder, f'best.pth'))
        print()
    
    writer.close()
    txt_writer.write_best_metric(best_epoch, best_accuracy = best_acc)
    print('Best epoch = {:3d}, Best Accuracy = {:.3f}'.format(best_epoch, best_acc))