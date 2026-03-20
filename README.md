# Google Summer of Code 2026 - ML4SCI Evaluation Tasks
**Candidate:** Wasiq Amir | BS Data Science, NUST 
**Target Organization:** Machine Learning for Science (ML4SCI)
**Target Projects:** Data Processing Pipeline for the LSST & Gravitational Lens Finding

This repository contains my solutions for the GSoC 2026 evaluation tests for the **DeepLense** sub-projects. The tasks are implemented in PyTorch, focusing on robust data handling, handling class imbalances, and generating required evaluation metrics.

## Repository Structure
* `Task 1 (Common Task)/DeepLense_Common_Task.ipynb`: Solution for Common Test I (Multi-Class Classification).
* `Task 2 (Specific Task V)/DeepLense_Specific_Task_V.ipynb`: Solution for Specific Test V (Lens Finding & Data Pipelines).
* `README.md`: Project overview and summary of results.

---

## Task Summaries

### **Common Test I: Multi-Class Classification**
* **Goal:** Classify simulated strong gravitational lensing images into three categories: `no substructure`, `sphere (subhalo)`, and `vort (vortex)`.
* **Approach:** Implemented a Convolutional Neural Network (ResNet-18 backbone) in PyTorch. The normalized datasets were utilized to train the model, incorporating a learning rate scheduler for optimized convergence.
* **Evaluation:** Maintained a strict **90:10 train-test validation split** as per the submission guidelines.
* **Metrics:** Evaluated using ROC curves and AUC scores for each class, demonstrating the model's ability to distinguish between subtle dark matter substructures. Plots are available at the end of the notebook.

### **Specific Test V: Lens Finding & Data Pipelines**
* **Goal:** Build a binary classifier to distinguish between lensed and non-lensed galaxies using observational data.
* **Data Handling:** Handled the `(3, 64, 64)` image arrays representing three distinct astronomical filters. Implemented custom PyTorch `Dataset` and `DataLoader` classes to manage the severe class imbalance (non-lenses significantly outnumbering lenses).
* **Evaluation:** A 90:10 split was utilized on the training directories for validation, while the final evaluation metrics were computed on the dedicated `test_lenses` and `test_nonlenses` holdout directories.
* **Metrics:** Generated ROC curves and calculated AUC scores to confirm the model's reliability on highly imbalanced observational data.

---

## Technology Stack
* **Language:** Python
* **Deep Learning Framework:** PyTorch
* **Data Science Libraries:** NumPy, Pandas, Matplotlib, Scikit-Learn
* **Environment:** Jupyter Notebook

## 📬 Contact
* **GitHub:** [@BlazedLith](https://github.com/BlazedLith)
* **LinkedIn:** [Wasiq Amir](https://www.linkedin.com/in/wasiq-amir)
* **Email:** wasiqamir3@gmail.com
