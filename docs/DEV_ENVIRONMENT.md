# Required Resources and Development Environment

## Software

| Tool | Version | Purpose |
| --- | --- | --- |
| Python | 3.10 or newer | Main language |
| OpenCV (`opencv-python`) | 4.8+ | RGB to HSV, thresholds, binary masks, boxes |
| NumPy | 1.24+ | Image arrays |
| PyTorch, torchvision | 2.1+, 0.16+ | CNN models (MobileNetV3, ResNet-18) |
| scikit-learn, pandas | 1.3+, 2.0+ | Data splits, metrics, results tables |
| Matplotlib | 3.7+ | Figures |
| Jupyter | 1.0+ | Notebooks for experiments and demos |
| pytest | 7.4+ | Unit tests |
| CVAT (app.cvat.ai or self-hosted) | latest | Drawing lesion boxes |
| Git and GitHub | — | Version control, pull requests, issues, dataset download |

## Hardware and compute

| Resource | Used for |
| --- | --- |
| Team laptops | Coding, HSV work, labeling, small tests |
| Google Colab (free GPU) | CNN training |
| CSUB GPU machine (if available) | Longer training runs |

## Data

| Dataset | Use |
| --- | --- |
| PlantVillage, apple classes ([GitHub](https://github.com/spMohanty/PlantVillage-Dataset); same images as [Hugging Face](https://huggingface.co/datasets/mohanty/PlantVillage)) | 3,171 leaf images: scab 630, black rot 621, cedar apple rust 275, healthy 1,645 |
| Our own lesion boxes (`data/labels/`) | Ground truth for training and scoring |

## Cost

All tools and data are free. No paid services are required.

## Quick check

```bash
pip install -r requirements.txt
python scripts/check_env.py
pytest -q
python -m src.hsv.demo --synthetic
```
