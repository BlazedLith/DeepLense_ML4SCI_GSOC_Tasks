# DeepLense Task 1 — Gravitational Lens Substructure Classification

A PyTorch project for classifying strong gravitational lensing images into three categories of dark matter substructure, built as part of the ML4SCI DeepLense challenge.

## The Problem

Strong gravitational lensing images can reveal the distribution of dark matter around massive objects. The task is to look at a lensing image and classify what kind of substructure is present:

- **no** — no dark matter substructure
- **sphere** — spherical subhalo substructure
- **vort** — vortex substructure

## Approach

A pretrained ResNet-18 is fine-tuned for this 3-class problem. The model's final layer is replaced (512 inputs to 3 outputs) while all earlier convolutional layers keep their ImageNet weights. This means the model already knows how to detect edges and textures, and just needs to learn which patterns distinguish the three lens types.

The images are single-channel `.npy` floating-point arrays, not standard photos. Keeping the data in float32 throughout the pipeline is important because converting to standard 8-bit images would clip the faint intensity gradients that separate the classes.

To feed the grayscale data into a 3-channel model without modifying the architecture, each single channel is simply repeated three times (`image.repeat(3, 1, 1)`). The model handles it fine.

## Setup

```bash
pip install torch torchvision scikit-learn matplotlib tqdm numpy
```

Data should be organized as:

```
data/task1/
  train/
    no/
    sphere/
    vort/
  val/
    no/
    sphere/
    vort/
```

The notebook pools train and val together and re-splits at 90/10 with stratification, so the folder split in the raw data doesn't affect training.

## Experiments

Three experiments were run, each building on the previous one.

| Experiment   | Augmentations                  | Scheduler         | Epochs | Best Val Accuracy |
| ------------ | ------------------------------ | ----------------- | ------ | ----------------- |
| 1 (baseline) | Flips only                     | None              | 10     | 92.4%             |
| 2            | Flips + Rotation + ColorJitter | ReduceLROnPlateau | 30     | 94.2%             |
| 3 (final)    | Flips only                     | ReduceLROnPlateau | 25     | **95.7%**         |

**Experiment 1** established a solid baseline fast. Horizontal and vertical flips are physically meaningful here since lensing images have no preferred orientation. No scheduler meant the learning rate bounced around a fixed value and validation loss zig-zagged rather than converging cleanly.

**Experiment 2** added heavier augmentations (random rotation up to 180°, color jitter) and a `ReduceLROnPlateau` scheduler. Accuracy improved to 94.2%, but epoch time nearly doubled. Rotation augmentations in tensor space are CPU-bound and expensive.

**Experiment 3** stripped the heavy augmentations back out and kept only the scheduler. Accuracy improved further to 95.7%, in less wall-clock time. The scheduler was doing most of the work all along. Simpler augmentations gave the model a cleaner gradient signal to learn from.

## Results

The final model (Experiment 3) achieves **95.7% validation accuracy** with per-class AUC scores near or above 0.99. The vortex class is cleanly separable from the others (AUC ~1.0).

## Key Decisions

**Stratified 90/10 split**: The dataset is re-split from scratch rather than using the preexisting train/val folders, with stratification to guarantee equal class representation in both sets. Required by the ML4SCI evaluation guidelines.

**AdamW optimizer**: Decoupled weight decay gives better generalization than standard Adam. Learning rate `1e-4`, weight decay `1e-4`.

**Best-checkpoint saving**: The training loop saves a copy of the model weights every time validation accuracy improves. The final returned model is always the best checkpoint, not just the last epoch.

**ROC evaluation**: Per-class ROC curves with AUC scores are generated after training using a One-vs-Rest strategy. They give a more complete picture of where the model is confident versus uncertain compared to accuracy alone.

## Files

- `DeepLense_Common_Task.ipynb` — the full notebook with all experiments, reasoning, and results
- `best_model.pth` — saved weights from the best validation checkpoint
