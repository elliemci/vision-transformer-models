# vision-transformer-models

The use and deployment of vision transformer models for image classification, object detection, and image segmentation of tumors in MRI data.

**Live demo:** [MRI tumor segmentation on Hugging Face Spaces](https://huggingface.co/spaces/elliemci/mri_segment)

## Repository structure

| Folder | Contents |
|---|---|
| [image_classification/](image_classification/) | MRI data processing and ViT tumor classification |
| [image_segmentation/](image_segmentation/) | Data preprocessing and tumor segmentation with MaskFormer and SegFormer |
| [object_detection/](object_detection/) | Detection notebooks for cracks, vehicles and damage |
| [model_implementation/](model_implementation/) | MRI classification app implementation |
| [model_deployment/](model_deployment/) | Gradio apps for deploying the classification and segmentation models |

## MRI tumor segmentation

### Overview

| Step | Where |
|---|---|
| Data preprocessing | [data_preproces.ipynb](image_segmentation/data_preproces.ipynb) |
| Train MaskFormer | [maskformer_seg_training.ipynb](image_segmentation/maskformer_seg_training.ipynb) |
| Train SegFormer (comparison) | [transformer_seg_training.ipynb](image_segmentation/transformer_seg_training.ipynb) |
| Gradio app | [app.ipynb](model_deployment/app.ipynb) |
| Deployed demo | [elliemci/mri_segment](https://huggingface.co/spaces/elliemci/mri_segment) |

### Data

Both models are trained on the [akar49/MRI_Segmentation-1](https://huggingface.co/datasets/akar49/MRI_Segmentation-1) dataset. Images and masks are loaded with the Hugging Face `imagefolder` loader.

### Models

| Model | Pretrained checkpoint | Notebook |
|---|---|---|
| **MaskFormer** (deployed) | `facebook/maskformer-swin-base-ade` | [maskformer_seg_training.ipynb](image_segmentation/maskformer_seg_training.ipynb) |
| SegFormer | `nvidia/segformer-b0-finetuned-ade-512-512` | [transformer_seg_training.ipynb](image_segmentation/transformer_seg_training.ipynb) |

Training settings for both: AdamW, learning rate 5e-5, 20 epochs, batch size 2.

### Results

Mean IoU is evaluated on a 100-item batch after each epoch.

| Model | Epoch 1 mean IoU | Best mean IoU | Epoch 20 mean IoU | Final train loss |
|---|---|---|---|---|
| MaskFormer | 0.447 | 0.474 (epoch 4) | 0.462 | 0.0153 |
| SegFormer | 0.364 | 0.434 (epoch 8) | 0.408 | 0.0021 |

MaskFormer test results on a small sample of test images:

- **Dice:** 0.948 average over 10 test images.
- **Per-sample IoU:** 0.996 to 0.999 for the first 9 samples.

MaskFormer reaches a higher mean IoU than SegFormer, even though SegFormer's training loss is lower. That is why MaskFormer is the model that was deployed.

#### Notes on the metrics

- The per-epoch mean IoU (about 0.46) and the per-sample IoU (about 0.998) are computed by different code. The first uses the Hugging Face `mean_iou` metric. The second uses a hand-written function.
- The hand-written function iterates over `range(int(np.max(true_mask)))`. With classes {0, 1} this only evaluates the background class, so the per-sample IoU is probably inflated. The Dice score is the more representative test number.

### Deployment

The [Hugging Face Space](https://huggingface.co/spaces/elliemci/mri_segment) is a Gradio app built from [app.ipynb](model_deployment/app.ipynb). It runs the fine-tuned MaskFormer model published as `elliemci/maskformer_tumor_segmentation`:

```python
model = MaskFormerForInstanceSegmentation.from_pretrained("elliemci/maskformer_tumor_segmentation")
image_processor = MaskFormerImageProcessor.from_pretrained("elliemci/maskformer_tumor_segmentation")
```

How the app works:

1. Upload one or more MRI images, or pick an example image and click "Process Example Image".
2. The images are converted to RGB and batched, and the model predicts segmentation masks.
3. The masks are post-processed to the size of the original images.
4. A gallery shows each original image next to its segmented mask. If an image fails, the gallery shows an error label instead.

## Object detection

Three notebooks fine-tune DETR-family models on Hugging Face object detection datasets. Each dataset has train, validation and test splits, with bounding-box annotations that the notebooks convert to COCO format before training. All three train for 10 epochs.

| Notebook | Dataset | Source | Images (train / val / test) | Classes | Model |
|---|---|---|---|---|---|
| [crack_detection.ipynb](object_detection/crack_detection.ipynb) | `elliemci/building_cracks` | Roboflow Universe [crack-detection](https://universe.roboflow.com/crack-7rsjb/crack-detection-ol3yi/dataset/1), exported to Hugging Face | 1,673 / 477 / 244 | `crack` | Conditional DETR (`microsoft/conditional-detr-resnet-50`) |
| [damage_detection.ipynb](object_detection/damage_detection.ipynb) | `elliemci/building_damages` (private) | Roboflow Universe [building-damages](https://universe.roboflow.com/elliemci/building-damages-nnk2b/dataset/2), merged from several open-source crack and building-defect datasets | 5,255 / 1,266 / 736 | 9 classes: combined damage, flexural, minorrotation, moderaterotation, severerotation, shear, undamage, cracks, water | Deformable DETR (`facebook/deformable-detr-box-supervised`) |
| [vehichile_detection.ipynb](object_detection/vehichile_detection.ipynb) | [`keremberke/excavator-detector`](https://huggingface.co/datasets/keremberke/excavator-detector) | Public Hugging Face dataset, loaded directly | 2,245 / 267 / 144 | 3 classes: excavators, dump truck, wheel loader | DETR (`facebook/detr-resnet-50`) |

### Crack detection

Detects building cracks with Conditional DETR, a DETR variant that uses conditional cross-attention for faster training. The Roboflow dataset is exported to the Hugging Face Hub with the `roboflow2huggingface` pipeline, which needs a Roboflow API key. The notebook then resizes and augments the images, trains the model, and publishes it as `elliemci/condDetr_crack_detection`. It ends with inference on a random image and a mean IoU evaluation.

### Building damage detection

Detects nine types of structural damage with Deformable DETR. The dataset is a merge of several open-source crack and building-defect datasets, hosted in a private Hugging Face repo and loaded from there. It is the largest of the three datasets. Training is tracked with Weights & Biases, and the model is published as `elliemci/deformDetr_damage_detection`. The notebook ends with inference at a 0.2 confidence threshold and a visualization of the detections.

### Construction vehicle detection

Detects excavators, dump trucks and wheel loaders with DETR on the public `keremberke/excavator-detector` dataset. It is the only notebook that evaluates on the test split with COCO metrics: the test images are saved in `CocoDetection` format and scored with the `ybelkada/cocoevaluate` metric. It ends with inference on a single image.

## Image classification

Notebook: [vit_tumor_classification.ipynb](image_classification/vit_tumor_classification.ipynb)

A `ViTForImageClassification` model fine-tuned on MRI images loaded with the Hugging Face `imagefolder` loader. It is published as `elliemci/vit_tumor_classification_model` and used in [mri_classification_app.ipynb](model_deployment/mri_classification_app.ipynb). Training settings: batch size 64, 3 epochs, learning rate 3e-5.

| Split | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Final training epoch | 1.0 | 1.0 | 1.0 | 1.0 |
| Test | 0.9705 | 1.0 | 0.9314 | 0.9645 |

Test precision of 1.0 means no false positives. Recall of 0.93 means some tumors were missed. The gap between training and test metrics suggests mild overfitting.

All metrics in this README come from the saved notebook outputs. The object detection section describes datasets and setup only, not detection metrics.
