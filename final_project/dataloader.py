import torch
import torchvision
import torchvision.transforms as transforms

ROOT = './data'

def get_dataloader(dataset='cifar10', 
                   batch_size: int = 1024):
    if dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        dataset_template = torchvision.datasets.CIFAR10
        num_classes = 10
        input_dims = 3
    elif dataset == 'cifar100':
        transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        dataset_template = torchvision.datasets.CIFAR100
        num_classes = 100
        input_dims = 3
        
    elif dataset == 'mnist':
        transform_train = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
        transform_test = transform_train
        dataset_template = torchvision.datasets.MNIST
        num_classes = 10
        input_dims = 1
    train_dataset = dataset_template(root=ROOT, train=True, download=True, transform=transform_train)
    test_dataset = dataset_template(root=ROOT, train=False, download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, pin_memory=True,
                                            shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, pin_memory=True,
                                            shuffle=False, num_workers=4)
        
    
    return train_loader, val_loader, num_classes, input_dims