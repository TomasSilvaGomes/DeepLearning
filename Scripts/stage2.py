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
from torchinfo import summary
from tqdm import tqdm
import time 





######################
#  Dataset Cifar-10  #O
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
# Cnn 
class CNN_double_channels(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), # duplicar o numero de channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU())
        self.second_block = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 8x8
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU())
        self.third_block = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 4x4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU())
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4 * 4, 128) # 4*4 porque a imagem é reduzida a 4x4 no ultimo maxpool
        self.fc2 = nn.Linear(128, num_classes)
        
        
    def forward(self, x):
        x = self.third_block(self.second_block(self.first_block(x)))
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class cnn_add_block(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU())
        self.second_block = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 8x8
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU())
        self.third_block = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 4x4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU())
        self.fourth_block = nn.Sequential(  # bloco adicional
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)) # reduz para 2x2
        self.fifth_block = nn.Sequential(  # bloco adicional
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)) # reduz para 1x1
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 1 * 1, 128) # 1*1 porque a imagem é reduzida a 1x1 no ultimo maxpool
        self.fc2 = nn.Linear(128, num_classes)
        
        
    def forward(self, x):
        x = self.fifth_block(self.fourth_block(self.third_block(self.second_block(self.first_block(x)))))
        x = self.flatten(x)
        x = self.fc1(x)
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
    # contar quanto tempo demora o treino
    start_time = time.time()

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
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Tempo total de treino: {elapsed_time/60:.2f} minutos")
    
    # plot the acc curve 
    plt.figure(figsize=(15,5))
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='Acuracia no Treino')
    plt.plot(val_accuracies, label='Acuracia na Validacao')
    plt.xlabel('Epocas')
    plt.ylabel('Acuracia (%)')
    plt.title('Acuracia do Treino e Validacao')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Loss no Treino')
    plt.plot(val_losses, label='Loss na Validacao')
    plt.xlabel('Epocas')
    plt.ylabel('Loss')
    plt.title('Loss do Treino e Validacao')
    plt.legend()
    plt.show()

        
    
if __name__ == "__main__":

    def set_seed(seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(42)

    
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_channels = CNN_double_channels().to(device)
    model_add_block = cnn_add_block().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_add_block = optim.Adam(model_add_block.parameters(), lr=0.001)
    optimizer_double_channels = optim.Adam(model_channels.parameters(), lr=0.001)
    max_epochs = 50
    model_save_double_channels = "models\\best_stage_2_1.pth"
    model_save_add_block = "models\\best_stage_2_2.pth"
    print(device)
    if os.path.exists(model_save_double_channels):
        print(f"Existe um modelo salvo relativamente ao double channels.")
        summary(model_channels, input_size=(64, 3, 32, 32), col_names=["input_size", "output_size", "num_params", "mult_adds"])
        # The fixed line
        model_channels.load_state_dict(torch.load(model_save_double_channels, map_location=torch.device('cpu')))
        model_channels.eval()
        acc = validation(model_channels, test_loader, criterion, device)[-1]
        print(f"Acurácia do modelo carregado: {acc:.2f}%")

        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model_channels(inputs)
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        cm = metrics.confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10,8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Matriz Confusao Stage 2_1")
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

    if not os.path.exists(model_save_double_channels):
        print("Nao existe modelo salvo. A treinar modelo do zero.")
        model_channels = model_channels.to(device)
        summary(model_channels, input_size=(3, 32, 32))
        train(model_channels, train_loader, optimizer_double_channels, criterion, max_epochs, device, model_save_double_channels)
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model_channels(inputs)
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        cm = metrics.confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10,8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Matriz Confusao Stage 2_1")
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
    
    if os.path.exists(model_save_add_block):
        print(f"Existe um modelo salvo relativamente ao add block.")
        summary(model_add_block, input_size=(64, 3, 32, 32), col_names=["input_size", "output_size", "num_params", "mult_adds"])
        # The fixed line
        model_add_block.load_state_dict(torch.load(model_save_add_block, map_location=torch.device('cpu')))
        model_add_block.eval()
        acc = validation(model_add_block, test_loader, criterion, device)[-1]
        print(f"Acurácia do modelo carregado: {acc:.2f}%")

        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model_add_block(inputs)
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        cm = metrics.confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10,8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Matriz Confusao Stage 2_2")
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
    
    if not os.path.exists(model_save_add_block):
        print("Nao existe modelo salvo. A treinar modelo do zero.")
        model_add_block = model_add_block.to(device)
        train(model_add_block, train_loader, optimizer_add_block, criterion, max_epochs, device, model_save_add_block)
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model_add_block(inputs)
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        cm = metrics.confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10,8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Matriz Confusao Stage 2_2")
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