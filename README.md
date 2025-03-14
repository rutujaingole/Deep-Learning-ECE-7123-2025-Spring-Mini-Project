
# 🛡️ CIFAR-10 Image Classification with Custom ResNet

This repository contains our **Deep Learning Mini Project** for CIFAR-10 image classification using a **Custom ResNet Architecture** under **5M parameters**, achieving **95.50% test accuracy** and **0.85694 Kaggle score**.

---

## 📂 Project Structure

```
📂 CIFAR10-ResNet/
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
| Kaggle Score          | **0.85694** |

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
git clone https://github.com/<your-username>/CIFAR10-ResNet
cd CIFAR10-ResNet
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
- Final Score: **0.85694**

---

## 👤 **Contributors**
- **Rutuja Ingole**  
- **Tanvi Takavane**  
- **Abhishek Agrawal**

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

## 🔗 Kaggle Leaderboard
[Final Kaggle Submission](https://www.kaggle.com/competitions/cifar10-resnet/)

---

## 🌟 **GitHub Repository Link for Gradescope Submission**
```
https://github.com/<your-username>/CIFAR10-ResNet
```
