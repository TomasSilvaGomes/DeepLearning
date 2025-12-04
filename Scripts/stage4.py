import torch    
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import os 
from torchinfo import summary
from tqdm import tqdm
import time 

######################
#  Dataset Cifar-10  #
######################

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # Normalização precisa do CIFAR-10
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset_full = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)

train_size = int(0.8 * len(train_dataset_full))
val_size = len(train_dataset_full) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset_full, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

class_names = train_dataset_full.classes
num_classes = len(class_names)

##########################################
#  Arquitetura ResNet-18 (CIFAR Adapted) #
##########################################

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        # Conv3x3 -> BN -> ReLU
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Conv3x3 -> BN
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut (Skip Connection)
        self.shortcut = nn.Sequential()
        # Se as dimensões mudarem (stride > 1) ou canais mudarem, ajustamos o x original
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # Soma Residual
        out = self.relu(out)
        return out

class ResNet18_CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18_CIFAR, self).__init__()
        self.in_channels = 64
        
        # Camada Inicial Adaptada para CIFAR-10 
        # Removemos o Kernel 7x7 stride 2 e o MaxPool da ResNet original
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Camadas ResNet (4 estágios)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

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

def train(model, train_loader, optimizer, criterion, scheduler, max_epochs, device, model_save_path):
    best_acc = 0.0
    start_time = time.time()
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    for epoch in range(max_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, epoch, max_epochs, device)
        val_loss, val_acc = validation(model, val_loader, criterion, device)
        
        # Atualizar learning rate 
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch+1}/{max_epochs}] LR: {current_lr:.5f} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"--> Melhor modelo salvo! Acc: {best_acc:.2f}%")
            
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Tempo total de treino: {elapsed_time/60:.2f} minutos")
    
    # Plots
    plt.figure(figsize=(15,5))
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Val Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Accuracy Curve (Best: {best_acc:.2f}%)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"A usar dispositivo: {device}")
    
    
    # Configurações do Stage 4
    max_epochs = 200 
    model_save_path = "models\\best_stage_4_resnet18.pth"
    if not os.path.exists("models"):
        os.makedirs("models")

    model = ResNet18_CIFAR(num_classes=10).to(device)
    
    # Optimizer e Scheduler para ResNet 
    # SGD com Momentum e Weight Decay
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    
    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    if os.path.exists(model_save_path):
        print(f"Carregando modelo existente: {model_save_path}")
        summary(model, input_size=(64, 3, 32, 32), col_names=["input_size", "output_size", "num_params", "mult_adds"])
        model.load_state_dict(torch.load(model_save_path))

    else:
        print("Iniciando treino Stage 4 (ResNet-18 from scratch)...")
        summary(model, input_size=(64, 3, 32, 32), col_names=["input_size", "output_size", "num_params", "mult_adds"])
        train(model, train_loader, optimizer, criterion, scheduler, max_epochs, device, model_save_path)


    print("A gerar matriz de confusão no Test Set")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    test_acc = metrics.accuracy_score(all_labels, all_preds) * 100
    print(f"Acurácia Final no Test Set: {test_acc:.2f}%")

    cm = metrics.confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix Stage 4 (ResNet18) - Test Acc: {test_acc:.2f}%")
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    thresh = cm.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, int(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.xlabel('Predicted')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()



##############################################################################################################
#
#  Tempo total de treino: 46.06 minutos
#  Acurácia Final no Test Set: 95.62%
# 
##############################################################################################################