# DeepLearning - CIFAR-10 Classification

Projeto de classificacao de imagens usando o dataset CIFAR-10, explorando diferentes arquiteturas de redes neurais convolucionais com melhorias progressivas.

## Descricao

Este projeto implementa e compara 5 estagios de evolucao de modelos de Deep Learning para classificacao de imagens do CIFAR-10, desde uma CNN simples ate uma ResNet-18 otimizada com tecnicas avancadas.

## Dataset

**CIFAR-10**: 60,000 imagens coloridas 32x32 em 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- Treino: 50,000 imagens (split 80/20 para treino/validacao)
- Teste: 10,000 imagens

## Estrutura do Projeto

```
DeepLearning/
├── data/
│   └── cifar-10-batches-py/
├── models/
│   ├── best_stage_1.pth
│   ├── best_stage_2_1.pth
│   ├── best_stage_2_2.pth
│   ├── best_stage_3.pth
│   ├── best_stage_4_resnet18.pth
│   └── best_stage5_se_mixup_warmup.pth
├── Scripts/
│   ├── stage1.py
│   ├── stage2.py
│   ├── stage3.py
│   ├── stage4.py
│   └── stage5.py
└── images/
```

## Estagios de Desenvolvimento

### Stage 1: CNN Simples
- Arquitetura basica com camadas convolucionais e pooling
- **Parametros**: 960,298
- **Problema**: Overfitting significativo sem BatchNormalization

### Stage 2: Aumento de Capacidade
Duas variantes testadas:
1. **Double Channels**: Duplicacao dos canais convolucionais
   - **Parametros**: 1,671,114
   - **Acuracia**: 73.08%
   - **Tempo**: 6min 26s

2. **Add Block**: Adicao de bloco convolucional extra
   - **Parametros**: 4,041,514
   - **Acuracia**: 71.47%
   - **Tempo**: 10min 52s

### Stage 3: Batch Normalization + Residual Connections
- Implementacao de conexoes residuais
- BatchNormalization para estabilizacao
- Data augmentation (RandomHorizontalFlip, RandomCrop, RandomErasing)
- **Parametros**: 1,714,314
- **Tempo**: 13min 11s
- **Melhoria**: Convergência mais rápida e acurácia de 87.85%.

### Stage 4: ResNet-18 Adaptada
- Implementacao customizada para CIFAR-10
- Ajuste da camada inicial (3x3 conv sem MaxPool vs 7x7 conv com MaxPool)
- Preservacao da resolucao espacial 32x32
- **Parametros**: 11,173,962
- **Justificativa**: ResNet standard (ImageNet) reduziria imagens 32x32 para 8x8 imediatamente

### Stage 5: Otimizacoes Avancadas
- **SE Blocks** (Squeeze-and-Excitation): Atencao de canal
- **Mixup**: Data augmentation com interpolacao de amostras
- **Warmup**: Esquema de learning rate progressivo
- **Parametros**: 11,219,018
- **Tecnicas**: Otimizador SGD com Momentum e Label smoothing, dropout, normalizacao precisa.

## Requisitos

```
torch
torchvision
numpy
matplotlib
seaborn
scikit-learn
pandas
tqdm
torchinfo
```

## Instalacao

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn pandas tqdm torchinfo
```

## Uso

Executar cada estagio individualmente:

```bash
python Scripts/stage1.py
python Scripts/stage2.py
python Scripts/stage3.py
python Scripts/stage4.py
python Scripts/stage5.py
```

Os modelos treinados sao salvos automaticamente no diretorio `models/`.

## Resultados

| Estagio | Parametros | Acuracia | Tempo | Tecnicas |
|---------|-----------|----------|-------|----------|
| Stage 1 | 960K | 73.64% | 3m 52s | CNN basica |
| Stage 2.1 | 1.67M | 91.94% | 6m26s | Double channels |
| Stage 2.2 | 4.04M | 86.72% | 10m52s | Add block |
| Stage 3 | 1.71M | 87.85% | 13m11s | BN + Residual |
| Stage 4 | 11.17M | 95.62% | 46m 06s | ResNet-18 |
| Stage 5 | 11.22M | 96.11% | 2h 47m | SE + Mixup + Warmup |

## Pontos Principais

1. **BatchNormalization** e essencial para evitar overfitting e estabilizar o treino
2. **Residual connections** permitem redes mais profundas e convergencia mais rapida
3. Arquiteturas para **CIFAR-10 requerem adaptacoes** (imagens 32x32 vs 224x224 do ImageNet)
4. **Data augmentation** melhora significativamente a generalizacao
5. Tecnicas avancadas (SE blocks, Mixup, Warmup) refinam o desempenho final

## Autor

Tomas Gomes - Nº51726
