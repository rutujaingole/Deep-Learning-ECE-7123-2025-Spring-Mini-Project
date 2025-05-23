# CIFAR-10 Image Classification with Custom ResNet - Perceptions (ECE 7123)

This project focuses on designing a **custom ResNet architecture from scratch with fewer than 5 million parameters.** The goal is to effectively classify images from the CIFAR-10 dataset while optimizing both performance and efficiency.  

The architecture leverages the concept of **BasicBlocks**, which consist of two convolutional layers with batch normalization and ReLU activation. By stacking multiple layers of these blocks, the model is able to extract hierarchical features from the dataset.

Through extensive experimentation with model configurations and hyperparameters, a custom ResNet model was successfully trained with 3 stages that achieves a test accuracy of 95.56% with 4916284 parameters, staying within the constraint of 5M parameters.  


| Parameter             | Value                  |
|----------------|----------------------|
| **B (Number of Blocks)** | [3, 4, 3] |
| **C (Channels per Block)** | [66, 132, 264] |
| **F (Filter Size)** | 3x3 |
| **P (Pooling Size)** | 8x8 |
| **Optimizer** | SGD |
| **Learning Rate Scheduler** | Cosine Annealing with Warmup |
| **Loss Function** | Smooth Cross-Entropy Loss (0.05) |
| **Best Test Accuracy** | **95.56%** |
| **Total Parameters** | **4,916,284** |

---
## 📂 Project Structure

```
📂 Deep-Learning-ECE-7123-2025-Spring-Mini-Project/
│
├── 📂 code/
│      └── ResNet_CIFAR10.ipynb         # Main notebook for training and inference
│      
│
├── 📂 trained_model/
│      └── model_weights.pth            # Trained model weights
│
├── 📂 docs/   
│      └── project_report.pdf            # Project documentation         
│
├── 📂 plots/
│      └── loss_curve.png
│      └── accuracy_curve.png
│      └── confusion_matrix.png
├                     
│
└── ── 📂 predictions/
│         └──submissions.csv
  ── 📝 README.md        
```

---

##  **Approaches Used**
 - Custom ResNet Architecture with **3 Residual Blocks**  
 -  Data Augmentation: **Random Crop, RandAugment, Random Erasing**  
 - **Label Smoothing CrossEntropy Loss**  
 -  **Cosine Annealing LR Scheduler with Warmup**  
 -  Optimizer: **SGD with Momentum**  
 -  Normalization adapted for **Custom Kaggle Dataset**  
 - Inference Pipeline for **PKL file testing & submissions.csv generation**

---

##  **Results**
| Metric             | Score         |
|----------------|---------------------|
| Train Accuracy        | **96.22%** |
| Test Accuracy | **95.56%** |
---

##  **Installation**
### Clone the repository
```bash
git clone https://github.com/rutujaingole/Deep-Learning-ECE-7123-2025-Spring-Mini-Project.git
cd Deep-Learning-ECE-7123-2025-Spring-Mini-Project
```

---

##  **Reproduce Results**

### Run the Jupyter Notebook:
```bash
cd code/
jupyter notebook ResNet_CIFAR10.ipynb
```

### For Inference on Custom PKL Dataset:
```bash
python inference.py
```

---

##  **Final Submission**
- Generated `submissions.csv` for Kaggle using `cifar_test_nolabel.pkl`
---

##  Reference Papers
- Deep Residual Learning for Image Recognition (He et al.)
- RandAugment: Practical Automated Data Augmentation Strategies

---

## 🔗 Project Report
[Deep Learning Mini Project Report (PDF)](./docs/project_report.pdf)

---

##  **GitHub Repository Link for Gradescope Submission**
```
https://github.com/rutujaingole/Deep-Learning-ECE-7123-2025-Spring-Mini-Project
```
