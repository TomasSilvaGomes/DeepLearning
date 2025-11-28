import torch    
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os 
from tqdm import tqdm
from stage3 import validation


#############################
# Configurações & Dados     #
#############################
BATCH_SIZE = 128
MAX_EPOCHS = 200
WARMUP_EPOCHS = 5  # 5 Épocas de aquecimento (Categoria: Stability)
ALPHA_MIXUP = 1.0  # (Categoria: Pipeline)

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
# 1. Architecture Enhancement: SE Block [cite: 85, 86] #
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
        
        # Modificação: Adicionar SE Block
        self.se = SEBlock(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out) # Aplica Attention
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18_SE(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18_SE, self).__init__()
        self.in_channels = 64
        # Stem CIFAR-10
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Camadas com SE Blocks
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
# 2. Pipeline Enhancement: MixUp [cite: 92, 93]           #
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
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{max_epochs}]", leave=True)
    
    for inputs, targets in loop:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Aplicação MixUp
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, ALPHA_MIXUP, device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return running_loss / len(train_loader)

##############################################################
# 3. Regularization & Stability: Warmup       #
##############################################################
# Função auxiliar para criar Scheduler com Warmup + Cosine
def get_scheduler(optimizer, warmup_epochs, max_epochs):
    # Scheduler principal: Cosine Annealing
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs)
    
    # Scheduler de aquecimento: Linear de 0 até LR base
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    
    # Encadeamento: Warmup -> Cosine
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, main_scheduler], 
        milestones=[warmup_epochs]
    )
    return scheduler

def train_stage5(model, train_loader, val_loader, optimizer, scheduler, device, epochs):
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    history = {'train_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        train_loss = train_one_epoch_mixup(model, train_loader, optimizer, criterion, epoch, epochs, device)
        
        val_metrics = validation(model, val_loader, criterion, device)
        val_acc = val_metrics[1]
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1} | LR: {current_lr:.5f} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/best_stage5_se_mixup_warmup.pth")
            print(f"--> SOTA Improved Model Saved: {best_acc:.2f}%")

    return history

if __name__ == "__main__":
    if not os.path.exists("models"):
        os.makedirs("models")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware: {device}")
    
    # Modelo modificado (Categoria 1: Architecture)
    model = ResNet18_SE(num_classes=10).to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    
    # Scheduler com Warmup (Categoria 3: Stability)
    scheduler = get_scheduler(optimizer, WARMUP_EPOCHS, MAX_EPOCHS)
    
    print("Iniciando Treino Stage 5: SE-Blocks + MixUp + LR Warmup")
    # O treino usa a função MixUp internamente (Categoria 2: Pipeline)
    history = train_stage5(model, train_loader, val_loader, optimizer, scheduler, device, MAX_EPOCHS)
    
    plt.figure(figsize=(10,5))
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f'Stage 5 Results (Best Acc: {max(history["val_acc"]):.2f}%)')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()