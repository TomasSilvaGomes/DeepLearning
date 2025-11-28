import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from stage5 import ResNet18_SE
# Importa a tua arquitetura
# Se estiver no mesmo notebook, não precisas de copiar a classe ResNet18_SE de novo.
# Se for noutro script, copia a definição da classe ResNet18_SE e SEBlock para aqui.

# --- Configurações ---
BATCH_SIZE = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dados de Teste ---
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Carregar Modelo ---
model = ResNet18_SE(num_classes=10).to(device)
path = "models/best_stage5_se_mixup_warmup.pth" # Verifica o caminho

if torch.cuda.is_available():
    model.load_state_dict(torch.load(path))
else:
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

model.eval()

# --- Avaliação ---
all_preds = []
all_labels = []
running_loss = 0.0
criterion = nn.CrossEntropyLoss()

print("A avaliar o modelo final...")
with torch.no_grad():
    for inputs, labels in test_loader:
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