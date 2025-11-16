import torch    
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import os 
from torchsummary import summary
from tqdm import tqdm





######################
#  Dataset Cifar-10  #
######################
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset_full = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_size = int(0.8 * len(train_dataset_full))
val_size = len(train_dataset_full) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset_full, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

class_names = train_dataset_full.classes
num_classes = len(class_names)


######################
#        CNN         #
######################
class CNN_simples(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#################################
#        Modelo e treino        #
#################################
def train_one_epoch(model, train_loader, optimizer, criterion, epoch, max_epochs, device):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch+1}/{max_epochs}]")
    for batch_idx, (inputs, labels) in loop:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item(), acc=100.*correct/total)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc

def validation(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / len(val_dataset)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc

def train(model, train_loader, optimizer, criterion, max_epochs, device, model_save_path):
    best_acc = 0.0

    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    for epoch in range(max_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, epoch, max_epochs, device)
        val_loss, val_acc = validation(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch+1}/{max_epochs}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"Melhor modelo salvo com: ({best_acc:.2f}%) de acc")
    
    # plot the acc curve 
    plt.figure(figsize=(10,5))
    plt.plot(train_accuracies, label='Acuracia no Treino')
    plt.plot(val_accuracies, label='Acuracia na Validacao')
    plt.xlabel('Epocas')
    plt.ylabel('Acuracia (%)')
    plt.title('Acuracia do Treino e Validacao')
    plt.legend()
    plt.show()
    # guardar a figura
    plt.savefig('Acc_stage1.png')
        
    
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_simples().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    max_epochs = 50
    model_save_path = "best_stage_1.pth"
    print(device)
    if os.path.exists(model_save_path):
        print(f"Existe um modelo salvo.")
        summary(model, input_size=(3, 32, 32))
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        acc = validation(model, val_loader, criterion, device)[1]
        print(f"Acurácia do modelo carregado: {acc:.2f}%")

        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        cm = metrics.confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10,8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Matriz Confusao Stage 1")
        plt.colorbar()
        plt.xlabel("Previsoes")
        plt.ylabel("Rótulos Verdadeiros")
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        # Adicionar os valores na matriz
        thresh = cm.max() / 2.
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, int(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        plt.xlabel('Previsoes')
        plt.ylabel('Rótulos Verdadeiros')
        plt.tight_layout()
        plt.show()

    else:
        print("Nao existe modelo salvo. A treinar modelo do zero.")
        model = model.to(device)
        summary(model, input_size=(3, 32, 32))
        train(model, train_loader, optimizer, criterion, max_epochs, device, model_save_path)
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        cm = metrics.confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10,8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Matriz Confusao Stage 1")
        plt.colorbar()
        plt.xlabel("Previsoes")
        plt.ylabel("Rótulos Verdadeiros")
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        # Adicionar os valores na matriz
        thresh = cm.max() / 2.
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, int(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        plt.xlabel('Previsoes')
        plt.ylabel('Rótulos Verdadeiros')
        plt.tight_layout()
        plt.show()