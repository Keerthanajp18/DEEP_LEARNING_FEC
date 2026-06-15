# Neural Clustering Based Visual Representation Learning (FEC-Inspired)

A simplified implementation inspired by the research paper **Neural Clustering Based Visual Representation Learning (FEC)**. This project explores clustering-based feature extraction using pretrained deep learning models and demonstrates both image-level clustering and FEC-style segmentation.

---

## 📌 Project Overview

Traditional CNNs process images using fixed grid structures. The FEC paper introduces clustering-based feature extraction, where similar visual regions are grouped together to create meaningful representations.

In this project, we implement a simplified version of this idea using:

- Deep feature extraction with ResNet101
- Image-level semantic clustering
- Region-level clustering (FEC-style segmentation)
- Unsupervised learning techniques

---

## 🎯 Objectives

- Understand clustering-based feature extraction
- Explore semantic image grouping
- Implement FEC-style image segmentation
- Demonstrate interpretability through visual outputs
- Build a practical FEC-inspired prototype

---

## 🗂 Dataset

**ImageNet_100**

A subset of the ImageNet dataset containing multiple object categories.

> Note: Labels are available in the dataset but were not used during clustering, making the implementation unsupervised.

---

## ⚙️ Methodology

### 1. Feature Extraction
- Pretrained ResNet101 model
- Final classification layer removed
- Deep feature vectors extracted from images

### 2. Feature Processing
- Standardization
- PCA (Principal Component Analysis)
- Feature normalization

### 3. Image-Level Clustering
- MiniBatchKMeans clustering
- Groups semantically similar images

### 4. FEC-Style Segmentation
- Image divided into patches
- Features extracted from patches
- KMeans clustering applied to group similar regions

---

## 🏗 Project Structure

```bash
neural_clustering_project/
│
├── data/
│   └── imagenet_100/
│
├── models/
│   └── clustering_layer.py
│
├── dataset.py
├── kmeans_clustering.py
├── fec_segmentation.py
├── requirements.txt
└── README.md
```

---

## 🛠 Technologies Used

- Python
- PyTorch
- TorchVision
- Scikit-Learn
- NumPy
- Matplotlib

---

## 📊 Results

### Feature Clustering

- ResNet101 used for feature extraction
- PCA used for dimensionality reduction
- MiniBatchKMeans used for clustering

| Metric | Value |
|----------|---------|
| Dataset | ImageNet_100 |
| Backbone Network | ResNet101 |
| Clustering Algorithm | MiniBatchKMeans |
| Number of Clusters | 8 |
| Silhouette Score | 0.1009 |

### Observations

✅ Similar images are grouped together

✅ Semantic patterns are visible across clusters

✅ Demonstrates clustering-based visual representation learning

---

## 🖼 FEC-Style Segmentation

The segmentation module clusters image regions based on visual similarity.

### Process

1. Divide image into patches
2. Extract deep features
3. Apply clustering
4. Generate clustered segmentation map

### Evaluation

The segmentation results were evaluated qualitatively through visual inspection.

> Quantitative metrics such as IoU require ground-truth segmentation masks, which were not available in this setup.

---

## ⚠️ Limitations

- Full FEC architecture not implemented
- No end-to-end clustering-based training
- Limited computational resources
- Depends on pretrained CNN features
- Some cluster overlap exists

---

## 🚀 Future Work

- Implement complete FEC architecture
- Train clustering modules end-to-end
- Improve segmentation quality
- Explore advanced clustering algorithms
- Evaluate on larger datasets

---

## ✅ Conclusion

This project successfully demonstrates a simplified FEC-inspired framework using deep feature extraction and clustering. The implementation explores both image-level semantic grouping and region-level segmentation while capturing the core intuition behind clustering-based visual representation learning.

---

## 📚 Reference

**Neural Clustering Based Visual Representation Learning (FEC)**

Research paper used as the primary reference for this implementation.
