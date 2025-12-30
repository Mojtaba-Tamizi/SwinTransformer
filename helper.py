import matplotlib.pyplot as plt
import os
import torch
from torchmetrics import ConfusionMatrix, Accuracy, Precision, Recall, F1Score
import itertools


os.makedirs('figs', exist_ok=True)


def plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig('figs/training_metrics.png')
    plt.show()


def predict_and_plot_grid(model, device, test_loader, class_names, grid_size=4):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    with torch.inference_mode():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    plt.figure(figsize=(10, 10))
    for i in range(grid_size * grid_size):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(images[i].cpu().permute(1, 2, 0))
        plt.title(f'Pred: {class_names[preds[i]]}\nTrue: {class_names[labels[i]]}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('figs/predictions_grid.png')
    plt.show()

def save_model(model, path='vision_transformer.pth'):
    torch.save(model.state_dict(), path)    

def evaluate(model, device, test_loader, num_classes, criterion):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        total_loss, correct = 0, 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * x.size(0)
            _, preds = torch.max(outputs, 1)
            correct += preds.eq(y.view_as(preds)).sum().item()
            all_preds.append(preds)
            all_labels.append(y)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    accuracy = Accuracy(num_classes=num_classes, task="multiclass").to(device)
    precision = Precision(num_classes=num_classes, average='macro', task="multiclass").to(device)
    recall = Recall(num_classes=num_classes, average='macro', task="multiclass").to(device)
    f1_score = F1Score(num_classes=num_classes, average='macro', task="multiclass").to(device)
    confusion_matrix = ConfusionMatrix(num_classes=num_classes, task="multiclass", normalize="true").to(device)

    acc = accuracy(all_preds, all_labels).item()
    prec = precision(all_preds, all_labels).item()
    rec = recall(all_preds, all_labels).item()
    f1 = f1_score(all_preds, all_labels).item()
    conf_mat = confusion_matrix(all_preds, all_labels).cpu().numpy()

    return acc, prec, rec, f1, conf_mat, total_loss / len(test_loader.dataset), correct / len(test_loader.dataset)


def train(model, device, train_loader, criterion, optimizer):
    model.train()
    total_loss, correct = 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(y.view_as(pred)).sum().item()
    return total_loss / len(train_loader.dataset), correct / len(train_loader.dataset)


def plot_confusion_matrix(conf_mat, class_names):
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Normalized)')
    plt.colorbar()
    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = conf_mat.max() / 2.
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        val = conf_mat[i, j]
        plt.text(j, i, f"{val:.2f}",
                 horizontalalignment="center",
                 color="white" if val > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('figs/confusion_matrix.png')
    plt.show()
