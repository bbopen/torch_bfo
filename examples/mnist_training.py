#!/usr/bin/env python
"""
MNIST Training Example with BFO-Torch Optimizers

This example demonstrates how to train a convolutional neural network
on the MNIST dataset using different BFO optimizer variants.

Author: Brett G. Bonner
Repository: https://github.com/bbopen/torch_bfo
"""

import argparse
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from bfo_torch import BFO, AdaptiveBFO, HybridBFO


class ConvNet(nn.Module):
    """Simple CNN for MNIST classification."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def get_data_loaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Get MNIST train and test data loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        './data', train=False, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, test_loader


def train_epoch(model, device, train_loader, optimizer, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        def closure():
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            return loss.item()
        
        # BFO optimization step
        loss = optimizer.step(closure)
        
        # Calculate accuracy
        with torch.no_grad():
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            total_loss += loss * target.size(0)
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss:.6f}')
    
    epoch_time = time.time() - start_time
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    
    print(f'\nTrain Epoch: {epoch} - Time: {epoch_time:.2f}s - '
          f'Avg Loss: {avg_loss:.6f} - Accuracy: {accuracy:.2f}%')
    
    return avg_loss, accuracy


def test(model, device, test_loader):
    """Evaluate model on test set."""
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    
    return test_loss, accuracy


def create_optimizer(model, args):
    """Create optimizer based on arguments."""
    optimizer_map = {
        'bfo': BFO,
        'adaptive': AdaptiveBFO,
        'hybrid': HybridBFO,
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam
    }
    
    if args.optimizer in ['bfo', 'adaptive', 'hybrid']:
        # BFO-based optimizers
        optimizer_class = optimizer_map[args.optimizer]
        optimizer_kwargs = {
            'lr': args.lr,
            'population_size': args.population_size,
            'chemotaxis_steps': args.chemotaxis_steps,
            'early_stopping': args.early_stopping
        }
        
        if args.optimizer == 'adaptive':
            optimizer_kwargs['adaptation_rate'] = 0.1
        elif args.optimizer == 'hybrid':
            optimizer_kwargs['gradient_weight'] = 0.3
            optimizer_kwargs['momentum'] = 0.9
        
        return optimizer_class(model.parameters(), **optimizer_kwargs)
    else:
        # Standard PyTorch optimizers for comparison
        optimizer_class = optimizer_map[args.optimizer]
        if args.optimizer == 'sgd':
            return optimizer_class(model.parameters(), lr=args.lr, momentum=0.9)
        else:  # adam
            return optimizer_class(model.parameters(), lr=args.lr)


def main():
    parser = argparse.ArgumentParser(description='MNIST training with BFO-Torch')
    parser.add_argument('--optimizer', type=str, default='bfo',
                        choices=['bfo', 'adaptive', 'hybrid', 'sgd', 'adam'],
                        help='Optimizer to use')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Input batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--population-size', type=int, default=30,
                        help='Population size for BFO optimizers')
    parser.add_argument('--chemotaxis-steps', type=int, default=10,
                        help='Chemotaxis steps for BFO optimizers')
    parser.add_argument('--early-stopping', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Device configuration
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'Using device: {device}')
    
    # Data loaders
    train_loader, test_loader = get_data_loaders(args.batch_size)
    
    # Model and optimizer
    model = ConvNet().to(device)
    optimizer = create_optimizer(model, args)
    
    print(f'\nTraining with {args.optimizer.upper()} optimizer')
    print(f'Learning rate: {args.lr}')
    if args.optimizer in ['bfo', 'adaptive', 'hybrid']:
        print(f'Population size: {args.population_size}')
        print(f'Chemotaxis steps: {args.chemotaxis_steps}')
    print(f'Batch size: {args.batch_size}')
    print(f'Epochs: {args.epochs}\n')
    
    # Training loop
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, epoch
        )
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Test
        test_loss, test_acc = test(model, device, test_loader)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        # Check early stopping
        if hasattr(optimizer, 'converged') and optimizer.converged:
            print(f'Early stopping triggered at epoch {epoch}')
            break
    
    # Final results
    print('\n' + '=' * 50)
    print('Training Complete!')
    print(f'Final Test Accuracy: {test_accuracies[-1]:.2f}%')
    print(f'Best Test Accuracy: {max(test_accuracies):.2f}%')
    print('=' * 50)
    
    # Save results for plotting
    results = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'optimizer': args.optimizer,
        'lr': args.lr
    }
    
    torch.save(results, f'mnist_results_{args.optimizer}.pt')
    print(f'\nResults saved to mnist_results_{args.optimizer}.pt')


if __name__ == '__main__':
    main()