import torch    
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import metrics
import os
from torchinfo import summary
from stage4 import validation


#############################
# Configurações & Dados     #
#############################
BATCH_SIZE = 128
MAX_EPOCHS = 200
WARMUP_EPOCHS = 5
ALPHA_MIXUP = 1.0

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)

########################################################
#       1. Architecture Enhancement: SE Block          #
########################################################
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class BasicBlockSE(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlockSE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels) # SE Block

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18_SE(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18_SE, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(BasicBlockSE, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlockSE, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlockSE, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlockSE, 512, 2, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

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

###########################################################
#            2. Pipeline Enhancement: MixUp               #
###########################################################
def mixup_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_one_epoch_mixup(model, train_loader, optimizer, criterion, epoch, max_epochs, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{max_epochs}]", leave=True)
    
    for inputs, targets in loop:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # MixUp
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, ALPHA_MIXUP, device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        # Cálculo de Acurácia Ponderada (Weighted Accuracy) para MixUp
        _, predicted = outputs.max(1)
        total += targets.size(0)
        # Se lam > 0.5, a label A é a dominante. Somamos o peso lambda se acertar.
        correct += (lam * predicted.eq(targets_a).float() + (1 - lam) * predicted.eq(targets_b).float()).sum().item()
        
        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

##############################################################
# 3. Regularization & Stability: Warmup                      #
##############################################################
def get_scheduler(optimizer, warmup_epochs, max_epochs):
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs)
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, main_scheduler], 
        milestones=[warmup_epochs]
    )
    return scheduler

def train_stage5(model, train_loader, val_loader, optimizer, scheduler, device, epochs):
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    # Inicializar dicionário para guardar as 4 métricas
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch_mixup(model, train_loader, optimizer, criterion, epoch, epochs, device)
        
        # Validation retorna (loss, acc)
        val_loss, val_acc = validation(model, val_loader, criterion, device)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1} | LR: {current_lr:.5f} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/best_stage5_se_mixup_warmup.pth")
            print(f"--> SOTA Improved Model Saved: {best_acc:.2f}%")

    return history

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18_SE(num_classes=10).to(device)
    print(f"Hardware: {device}")
    print(summary(model, input_size=(BATCH_SIZE, 3, 32, 32), col_names=["input_size", "output_size", "num_params", "mult_adds"]))

    if os.path.exists("models\\best_stage5_se_mixup_warmup.pth"):
        print("Modelo pré-treinado encontrado. Carregando pesos...")
        model.load_state_dict(torch.load("models\\best_stage5_se_mixup_warmup.pth"))

        # matriz de confusão
        criterion = nn.CrossEntropyLoss()
        val_loss, val_acc = validation(model, val_loader, criterion, device)
        print(f"Validação do modelo carregado - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        all_preds = []
        all_labels = []
        running_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        print("A avaliar o modelo final...")
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        final_loss = running_loss / len(test_dataset)
        final_acc = metrics.accuracy_score(all_labels, all_preds) * 100


        print(f"Resultados Finais Stage 5:")
        print(f"Test Accuracy: {final_acc:.2f}%")
        print(f"Test Loss: {final_loss:.4f}")

        # --- Plot Matriz de Confusão ---
        cm = metrics.confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10,8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
        plt.title(f"Stage 5 Confusion Matrix (Acc: {final_acc:.2f}%)")
        plt.colorbar()
        class_names = test_dataset.classes
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        thresh = cm.max() / 2.
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()



    else:

        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = get_scheduler(optimizer, WARMUP_EPOCHS, MAX_EPOCHS)
        
        print("Iniciando Treino Stage 5: SE-Blocks + MixUp + LR Warmup")
        
        history = train_stage5(model, train_loader, val_loader, optimizer, scheduler, device, MAX_EPOCHS)
        
        # --- PLOT FINAL COM 2 SUBPLOTS ---
        plt.figure(figsize=(15,5))
        
        # Subplot 1: Accuracy (Train vs Val)
        plt.subplot(1, 2, 1)
        plt.plot(history['train_acc'], label='Train Accuracy (Weighted)', color='blue', alpha=0.7)
        plt.plot(history['val_acc'], label='Validation Accuracy', color='orange', linewidth=2)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy: Train (MixUp) vs Validation')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Subplot 2: Loss (Train vs Val)
        plt.subplot(1, 2, 2)
        plt.plot(history['train_loss'], label='Train Loss (MixUp)', color='blue', alpha=0.7)
        plt.plot(history['val_loss'], label='Validation Loss', color='red', linewidth=2)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss: Train vs Validation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


        