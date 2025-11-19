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
import time 





######################
#  Dataset Cifar-10  #O
######################
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),      # Fundamental para CIFAR-10
    transforms.RandomCrop(32, padding=4),        # Obriga a rede a aprender invariância de posição
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')
])

# Transformações para VALIDAÇÃO / TESTE (Apenas normalização)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset_full = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
train_size = int(0.8 * len(train_dataset_full))
val_size = len(train_dataset_full) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset_full, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

class_names = train_dataset_full.classes
num_classes = len(class_names)


######################
#        CNN         #
######################
# Cnn 
class CNN_double_channels(nn.Module):
    def __init__(self, num_classes=10): # Adicionei num_classes como parametro
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False), # bias=False quando usamos BN
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 32x32 -> 16x16
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            
        )
        self.res1_adjust = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(64)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16x16 -> 8x8
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.res2_adjust = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 8x8 -> 4x4
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )
        self.res3_adjust = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256)
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4 * 4, 128)
        self.bn_fc1 = nn.BatchNorm1d(128) # BN nas camadas lineares ajuda a estabilizar
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)    # Dropout adicionado aqui
        self.fc2 = nn.Linear(128, num_classes)
        
        
    def forward(self, x):
        # --- Passagem pelo Bloco 1 ---
        identity = self.res1_adjust(x) # Prepara o residual (redimensiona)
        x = self.block1(x)             # Passa pelo bloco conv
        x += identity                  # Soma Residual (Skip Connection)
        x = self.relu(x)               # ReLU final do bloco
        
        # --- Passagem pelo Bloco 2 ---
        identity = self.res2_adjust(x)
        x = self.block2(x)
        x += identity
        x = self.relu(x)

        # --- Passagem pelo Bloco 3 ---
        identity = self.res3_adjust(x)
        x = self.block3(x)
        x += identity
        x = self.relu(x)

        # --- Classificador ---
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn_fc1(x) # Batch Norm
        x = self.relu(x)   # Ativação (Faltava no teu código original)
        x = self.dropout(x)# Dropout
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
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_channels = CNN_double_channels().to(device)
    model_add_block = CNN_double_channels().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_channels.parameters(), lr=0.001, weight_decay=1e-4)
    max_epochs = 50
    model_save_components = "best_stage_3.pth"
    print(device)
    if os.path.exists(model_save_components):
        print(f"Existe um modelo salvo relativamente ao double channels.")
        summary(model_channels, input_size=(3, 32, 32))
        # The fixed line
        model_channels.load_state_dict(torch.load(model_save_components, map_location=torch.device('cpu')))
        model_channels.eval()
        acc = validation(model_channels, val_loader, criterion, device)[1]
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
        plt.title("Matriz Confusao Stage 3")
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

    if not os.path.exists(model_save_components):
        print("Nao existe modelo salvo. A treinar modelo do zero.")
        model_channels = model_channels.to(device)
        summary(model_channels, input_size=(3, 32, 32))
        train(model_channels, train_loader, optimizer, criterion, max_epochs, device, model_save_components)
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
        plt.title("Matriz Confusao Stage 3")
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
    

