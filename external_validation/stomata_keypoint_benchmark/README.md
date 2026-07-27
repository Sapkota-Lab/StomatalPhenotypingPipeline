---
viewer: false
license: cc-by-nc-4.0
pretty_name: "Stomata Keypoint Detection (CVPR AgriVision 2026)"
task_categories:
  - keypoint-detection
  - object-detection
language:
  - en
tags:
  - stomata
  - plant-phenotyping
  - biology
  - microscopy
  - domain-shift
  - keypoint-detection
  - computer-vision
  - agriculture
  - maize
  - cvpr
  - agrivision
size_categories:
  - n<1K
---

# Stomata Keypoint Detection: Dataset


This repository contains a multi-domain dataset for stomata keypoint detection. It was created to evaluate how well keypoint detection models hold up under distribution shift across imaging environments, plant groups, species, and microscope systems.

The benchmark accompanies the following paper:

**Towards Morphology Aware Stomata Keypoint Detection: Benchmarking Foundation Models Under Distribution Shift**  
*Accepted at the CVPR 2026 AgriVision Workshop*

- **Paper:** Coming soon
- **Code:** Coming soon on GitHub
- **Models:** [stomata-keypoint-benchmark-cvpr-agrivision-2026](https://huggingface.co/Sainath001/stomata-keypoint-benchmark-cvpr-agrivision-2026-models)


---

## Overview

The benchmark includes **1 training split** and **9 test splits**. Together, they cover several types of distribution shift, including field-to-lab transfer, monocot-to-dicot transfer, species changes, and sensor changes.

Each stomata instance is annotated with **4 keypoints** in COCO format:
- two **polar tips** along the stomatal length axis
- two **lateral endpoints** along the stomatal width axis

### Dataset Summary

| Split | Environment | Botanical Class | Microscope | Species | Images | Stomata |
|------|------|------|------|------|------:|------:|
| **KP-Train** | Field | Monocot | ProScope HR | Maize | 344 | 12,503 |
| **T-MZLP** | Field | Monocot | ProScope HR | Maize | 86 | 3,188 |
| **T-MZA** | Field | Monocot | ProScope HR | Maize | 115 | 4,349 |
| **T-SBGH** | Greenhouse | Dicot | ProScope HR | Soybean | 13 | 524 |
| **T-MSAA** | Field | Dicot | ProScope HR | Nokaidō | 12 | 444 |
| **T-HR5MZ** | Lab | Monocot | ProScope HR5 | Maize | 10 | 924 |
| **T-HR5WH** | Greenhouse | Monocot | ProScope HR5 | Wheat | 6 | 153 |
| **T-TCMZ** | Lab | Monocot | Toupcam S500-GS | Maize | 25 | 270 |
| **T-NKBY** | Lab | Monocot | Nikon DS-Fi3 | Barley | 15 | 175 |
| **T-TCAB** | Lab | Dicot | Toupcam S500-GS | Arabidopsis | 13 | 134 |

**Total:** 639 images and 22,664 annotated stomata

---

## What This Benchmark Measures

The test splits were designed to probe robustness under several kinds of distribution shift relative to **KP-Train**.

| Shift Type | Test Splits |
|------|------|
| **Location shift** | T-MZA |
| **Plant-level shift** | T-MZLP |
| **Environment shift** | T-SBGH, T-HR5MZ, T-TCMZ |
| **Taxonomic shift** | T-SBGH, T-MSAA, T-TCAB |
| **Species shift** | T-HR5WH, T-NKBY, T-MSAA |
| **Sensor/system shift** | T-HR5MZ, T-HR5WH, T-TCMZ, T-TCAB, T-NKBY |

This setup makes the benchmark useful for testing not only in-domain accuracy, but also how well a model generalizes to new biological and imaging conditions.

---

## Annotation Format

Annotations are provided in **COCO keypoint format**.

Each stomata instance includes:
- a bounding box: `[x, y, width, height]`
- four keypoints: `[x0, y0, v0, x1, y1, v1, x2, y2, v2, x3, y3, v3]`

### Keypoint Definition
- `p0, p1`: polar tips along the stomatal length axis
- `p2, p3`: lateral endpoints along the stomatal width axis

### Annotation Source
For each dataset split:
- `annotations/` contains the original annotation files
- `coco.json` contains the COCO-format annotations

The original annotations inside `annotations/` include Pascal VOC XML files, and the COCO annotations were generated from those annotations using RectLabel Pro.

---

## Repository Structure

```text
datasets/
├── KP-Train/
│   ├── annotations/
│   ├── images/
│   └── coco.json
├── T-MZLP/
│   ├── annotations/
│   ├── images/
│   └── coco.json
├── T-MZA/
│   ├── annotations/
│   ├── images/
│   └── coco.json
├── T-SBGH/
│   ├── annotations/
│   ├── images/
│   └── coco.json
├── T-MSAA/
│   ├── annotations/
│   ├── images/
│   └── coco.json
├── T-HR5MZ/
│   ├── annotations/
│   └── coco.json
├── T-HR5WH/
│   ├── annotations/
│   └── coco.json
├── T-TCMZ/
│   ├── annotations/
│   └── coco.json
├── T-NKBY/
│   ├── annotations/
│   └── coco.json
└── T-TCAB/
    ├── annotations/
    └── coco.json
```

---

## External Data Sources

Some test splits contain images derived from previously published dataset. These sources are listed below for proper attribution.

Splits T-HR5MZ, T-TCMZ, T-TCAB, T-HR5WH, and T-NKBY include images originating from the following works:

1. Xiaohui Yang, Jiahui Wang, Fan Li, Chenglong Zhou, Minghui Wu, Chen Zheng, Lijun Yang, Zhi Li, Yong Li, Siyi Guo, et al.
   RotatedStomataNet: A Deep Rotated Object Detection Network for Directional Stomata Phenotype Analysis.
   Plant Cell Reports, 43(5):126, 2024.
   Repository: [https://github.com/AITAhenu/RotatedStomataNet/tree/main/test-images](https://github.com/AITAhenu/RotatedStomataNet/tree/main/test-images)

2. Phetdalaphone Pathoumthong, Zhen Zhang, Stuart J. Roy, and Abdeljalil El Habti.
   Rapid Non-destructive Method to Phenotype Stomatal Traits.
   Plant Methods, 19(1):36, 2023.
   Repository: [https://github.com/rapidmethodstomata/rapidmethodstomata](https://github.com/rapidmethodstomata/rapidmethodstomata)

3. Na Sai, James Paul Bockman, Hao Chen, Nathan Watson Haigh, Bo Xu, Xueying Feng, Adriane Piechatzek, Chunhua Shen, and Matthew Gilliham.
   StomataAI: An Efficient and User-friendly Tool for Measurement of Stomatal Pores and Density Using Deep Computer Vision.
   New Phytologist, 238(2):904–915, 2023.
   Repository: [https://github.com/xdynames/sai-app](https://github.com/xdynames/sai-app)

The keypoint annotations for these benchmark splits were newly created by Sainath Reddy Gummi.

If you redistribute this benchmark, please source the external images from above citations and github links.


---

## License

| Component                                              | License                                           |
| ------------------------------------------------------ | ------------------------------------------------- |
| Images and annotations for internal splits             | CC BY-NC 4.0                                      |
| Keypoint annotations created for external-image splits | CC BY-NC 4.0                                      |


---

## Citation

If you use this dataset and annotations, please cite:

```bibtex
@inproceedings{gummi2026stomata,
  author    = {Gummi, S. R. and Pack, C. and Zhang, H. K. and Solanki, S. and Chang, Y.},
  title     = {Towards Morphology Aware Stomata Keypoint Detection: Benchmarking Foundation Models Under Distribution Shift},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  year      = {2026},
  note      = {Accepted}
}
```
---

## Contact

Sainath Reddy Gummi
South Dakota State University
Email: [gummisainath@gmail.com](mailto:gummisainath@gmail.com)