# Week 2 Results — Transfer Learning and Data Augmentation

## Setup
- Dataset: CIFAR-10 (10 classes)
- Backbone: ResNet18
- Input resolution: 96x96
- Epochs per experiment: 4
- Optimizer: Adam
- Pretrained runs: ImageNet-pretrained ResNet18; train classifier head first, then fine-tune layer4 and head using lower learning rates.
- Model selection: best validation accuracy; final metrics measured on the official test set.

## Ablation results

| Experiment                             |   Best validation accuracy (%) |   Test accuracy (%) |   Test loss |   Training time (min) |
|:---------------------------------------|-------------------------------:|--------------------:|------------:|----------------------:|
| ResNet18 from scratch, no augmentation |                          74.94 |               74.93 |        0.75 |                  2.46 |
| Pretrained ResNet18, no augmentation   |                          89.04 |               88.63 |        0.4  |                  1.81 |
| Pretrained ResNet18, with augmentation |                          87.9  |               87.94 |        0.35 |                  3.47 |

## Interpretation
Transfer learning changed test accuracy by **+13.70 percentage points** relative to the from-scratch baseline.

Data augmentation reduced test accuracy by **-0.69 percentage points** compared with pretrained ResNet18 without augmentation. A negative effect is still a valid experimental result; augmentation strength, training duration, and random variation can affect the outcome.

The best measured run was **Pretrained ResNet18, no augmentation**, with test accuracy of **88.63%**. These results apply to this run and configuration, not all possible training settings.

## Artifacts
- `ablation_results.csv`: comparison table
- `confusion_matrix.png`: final confusion matrix
- `per_class_metrics.csv`: precision, recall, F1-score, support
- `validation_accuracy_comparison.png`: validation curves
- `*_best.pth`: saved model checkpoints
