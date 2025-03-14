# 🛡️ CIFAR-10 Image Classification with Custom ResNet

This repository contains our **Deep Learning Mini Project** for CIFAR-10 image classification using a **Custom ResNet Architecture** under **5M parameters**, achieving **95.56% test accuracy** 

---

## 📂 Project Structure

```
📂 Deep-Learning-ECE-7123-2025-Spring-Mini-Project/
│
├── 📂 code/
│      └── ResNet_CIFAR10.ipynb         # Main notebook for training and inference
│      └── model.py                     # Custom ResNet Model
│
├── 📂 trained_model/
│      └── model_weights.pth            # Trained model weights
│
├── 📂 docs/
│      └── project_report.pdf           # Project report (AAAI format)
│
├── 📂 plots/
│      └── loss_curve.png
│      └── accuracy_curve.png
│
├── 📝 README.md                        # Project documentation
│
└── 📄 requirements.txt                 # Python dependencies
```

---

## 🚀 **Objective**
Design a **custom lightweight ResNet architecture under 5M parameters** that **achieves high test accuracy on CIFAR-10** and **generalizes well on unseen Kaggle dataset.**

---

## 🛠️ **Approaches Used**
✅ Custom ResNet Architecture with **3 Residual Blocks**  
✅ Data Augmentation: **Random Crop, RandAugment, Random Erasing**  
✅ **Label Smoothing CrossEntropy Loss**  
✅ **Cosine Annealing LR Scheduler with Warmup**  
✅ Optimizer: **SGD with Momentum**  
✅ Normalization adapted for **Custom Kaggle Dataset**  
✅ Inference Pipeline for **PKL file testing & submissions.csv generation**

---

## 🎯 **Results**
| Metric             | Score         |
|----------------|---------------------|
| Train Accuracy        | **96.24%** |
| Validation Accuracy | **95.50%** |

---

## 📉 **Training Loss Curve**
<img src="./plots/loss_curve.png" alt="Loss Curve" width="500">

---

## 🏗️ Model Architecture
```
ResNet-18 Custom
-------------------
Conv2d → BN → ReLU
→ 3 Residual Blocks (66-132-264 Channels)
→ Global Average Pooling
→ Fully Connected Layer
```

---

## 🛠️ **Installation**
### Clone the repository
```bash
git clone https://github.com/rutujaingole/Deep-Learning-ECE-7123-2025-Spring-Mini-Project.git
cd Deep-Learning-ECE-7123-2025-Spring-Mini-Project
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📚 **Reproduce Results**

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

## 🏁 **Final Submission**
- Generated `submissions.csv` for Kaggle using `cifar_test_nolabel.pkl`
---

## 👤 **Contributors**
- **Rutuja Ingole**  Net ID: rdi4221
- **Tanvi Takavane**  Net ID: rdi4221
- **Abhishek Agrawal** Net ID: rdi4221

---

## 📌 Reference Papers
- Deep Residual Learning for Image Recognition (He et al.)
- RandAugment: Practical Automated Data Augmentation Strategies

---

## 🛡️ License
MIT License

---

## 🔗 Project Report (AAAI Format)
[Deep Learning Mini Project Report (PDF)](./docs/project_report.pdf)

---

## 🌟 **GitHub Repository Link for Gradescope Submission**
```
https://github.com/rutujaingole/Deep-Learning-ECE-7123-2025-Spring-Mini-Project
```
