import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import numpy as np
import random
from model import VisionTransformer
from tqdm.auto import tqdm
from helper import plot_metrics, evaluate, train, predict_and_plot_grid, save_model, plot_confusion_matrix
import json


print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("NumPy version:", np.__version__)
print("CUDA available:", torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

IMAGE_SIZE = 32
IN_CHANNELS = 3
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 3e-4
PATCH_SIZE = 4
NUM_CLASSES = 10
EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 6
MLP_DIM = 512
DROPOUT_RATE = 0.2


transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

print("Number of training samples:", len(train_dataset))
print("Number of test samples:", len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print("Number of training batches:", len(train_loader))
print("Number of test batches:", len(test_loader))


model = VisionTransformer(IMAGE_SIZE, PATCH_SIZE, IN_CHANNELS, NUM_CLASSES, 
                          EMBED_DIM, NUM_HEADS, NUM_LAYERS, MLP_DIM, DROPOUT_RATE).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

logger = {
    'train_loss': [],
    'train_acc': [],
    'train_precision': [],
    'train_recall': [],
    'train_f1': [],
    'train_conf_mat': [],
    'test_loss': [],
    'test_acc': [],
    'test_precision': [],
    'test_recall': [],
    'test_f1': [],
    'test_conf_mat': []
}
for epoch in tqdm(range(1, EPOCHS + 1)):
    train_loss, train_acc = train(model, device, train_loader, criterion, optimizer)
    acc, prec, rec, f1, conf_mat, _, _ = evaluate(model, device, train_loader, NUM_CLASSES, criterion)
    logger['train_loss'].append(float(train_loss))
    logger['train_acc'].append(float(train_acc))
    logger['train_precision'].append(float(prec))
    logger['train_recall'].append(float(rec))
    logger['train_f1'].append(float(f1))
    logger['train_conf_mat'].append(conf_mat.tolist())

    acc, prec, rec, f1, conf_mat, test_loss, test_acc = evaluate(model, device, test_loader, NUM_CLASSES, criterion)
    logger['test_loss'].append(float(test_loss))
    logger['test_acc'].append(float(test_acc))
    logger['test_precision'].append(float(prec))
    logger['test_recall'].append(float(rec))
    logger['test_f1'].append(float(f1))
    logger['test_conf_mat'].append(conf_mat.tolist())
try:
    with open('training_log.json', 'w') as outfile:
        json.dump(logger, outfile, indent=4)

except: 
    print("Could not save the training log.")

plot_metrics(logger['train_loss'], logger['test_loss'], logger['train_acc'], logger['test_acc'])
plot_confusion_matrix(np.array(logger['test_conf_mat'][-1]), class_names=train_dataset.classes)