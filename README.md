# COV Net - Oxford-IIIT Pet Classification

This project was created to test the performance of the Convolutional layers and the Residual Network. The purpose of the project is to classify 37 races of Cats and Dogs from the **Oxford-IIIT Pet Dataset**
We're comparing custom-made Convolutional Network with Fine-tuned professional model called **"ResNet-18"** which lies on skip connections
## Architecture and Configuration of Models

We approached the topic by preparing 2 different scenarios and 4 differently configured models, thanks to the **Pytorch** library.
### 1. Custom CNN (`PetNet`)

We've prepared 3 variants of hyperparameters for this architecture:

* **`Base_NoAug`:** Model which doesn't have any augmentation, Batchnorm, dropout or weight_decay configured. Which means **fast training and fast overfitting**.
* **`Aug_LowDrop`:** Added **basic augmentation** and small **Dropout** (0.3).
* **`Aug_HighDrop`:** Implemented everything and reduced Dropout.

### 2. Transfer Learning (`PetResNet`)

Professional approach. We've used the **ResNet-18** architecture, which is trained on ImageNet storage (which is 1 million images)

---

## Experimental Results

The visualization (available in the attached graphs) clearly shows two extremes: massive overfitting in the simple network vs. the total dominance of Transfer Learning.

| Model                   | Epochs (to Early Stop) | Max Train Acc | **Max Test Acc** |
|:------------------------|:-----------:| :---: | :---: |
| `Base_NoAug`            | 10 | 100.00% | **12.6%** |
| `Aug_LowDrop`           | 34 | 4.76% | **4.9%** |
| `Aug_HighDrop`          | 8 | 2.72% | **2.8%** |
| **`ResNet18_Transfer`** | **43** | **100.00%** | **88.3%** |

### Summary and Conclusions

1. **Model Capacity Limits:** The custom 3-block `PetNet` hit a "ceiling" during fine-grained classification. The "from scratch" models (`Aug_LowDrop`, `Aug_HighDrop`) with high-resolution images (224x224) and heavy distortions experienced architectural collapse, resulting in accuracy near random chance (~2.7% for 37 classes). 

    Conversely, the `Base_NoAug` model perfectly (100%) memorized the training set while failing to generalize.
2. **The Power of Transfer Learning:** Replacing the shallow network with ResNet-18 resulted a huge increase in effectiveness. By "knowing" how to recognize textures and basic shapes from ImageNet, ResNet crossed the 80% threshold in just a few epochs, peaking at an impressive **88.3%**.

This project proves that for complex Computer Vision problems and small datasets, **Transfer Learning is an absolutely essential and highly effective engineering tool**.