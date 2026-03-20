# DeepLense Task 2 (Specific Task 5) — Binary Gravitational Lens Detection

Binary classification of gravitational lensing images as **lens** or **non-lens** using transfer learning with ResNet-18.

## Task

Classify `.npy` lensing images into two classes: `lens` (1) and `non-lens` (0). The core challenge is severe class imbalance — the training set has roughly 17× more non-lens samples than lens samples, so standard accuracy is a misleading metric. The primary evaluation metric is **AUC**.

## Dataset

The dataset provides separate train and test directories:

- `train_lenses/`, `train_nonlenses/` — used for training and validation
- `test_lenses/`, `test_nonlenses/` — locked test set, used only for final evaluation

Data is pre-normalized to `[0, 1]` and stored as 3-channel `.npy` float32 arrays.

**Training set class counts:** 25,807 non-lens / 1,557 lens (~17:1 ratio)

## Approach

**Model:** ResNet-18 (ImageNet pretrained), final FC layer replaced with a 2-output binary classifier.

**Class imbalance handling:** Weighted `CrossEntropyLoss` with dynamically computed class weights:

$$w_c = \frac{N}{k \times n_c}$$

This penalizes lens misclassifications ~16× more than non-lens misclassifications, preventing the model from collapsing to the majority class.

**Data pipeline:**

- 90/10 stratified train/val split within the train folders
- Augmentation: `RandomHorizontalFlip` + `RandomVerticalFlip` (physically motivated — lensing images have no preferred orientation)
- Batch size: 32

**Optimizer:** AdamW, `lr=1e-4`, `weight_decay=1e-4`

## Experiments

| Experiment          | Epochs | Scheduler             | Best Val Acc | Test Acc   | Val AUC    | Test AUC   |
| ------------------- | ------ | --------------------- | ------------ | ---------- | ---------- | ---------- |
| 1                   | 15     | None (static LR)      | 97.37%       | 98.11%     | 0.9852     | 0.9839     |
| **2 — Final Model** | **15** | **ReduceLROnPlateau** | **97.44%**   | **98.07%** | **0.9915** | **0.9841** |

**Experiment 2** is the submitted model. Both experiments run for 15 epochs but the scheduler in Experiment 2 improves AUC on both val and test sets and reduces missed lenses from 31 to 20 on the test set. Since missing real lenses is the costly error on an imbalanced detection task, Experiment 2 is the better model despite marginally lower raw accuracy.

## Results

**Final model (Experiment 2):**

- Test Accuracy: **98.07%**
- Test AUC: **0.9841**
- Lenses detected: **175 / 195** on test set
- Lenses missed: **20**

## Requirements

```
torch
torchvision
numpy
scikit-learn
matplotlib
tqdm
```

## Usage

Set `PATH` to point to the dataset root and run the notebook cells in order. The best model weights are saved to `Best Model.pth` after Experiment 2.
