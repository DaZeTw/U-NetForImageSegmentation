# U-NetForImageSegmentation

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)

## Overview

This project implements a U-Net architecture for the segmentation of water regions in flood-affected areas. The dataset consists of flood images and manually annotated water masks. The goal is to train a segmentation model that can automatically detect water regions in new flood images. The model is built using TensorFlow and Keras, with data augmentation techniques to handle the limited dataset size.

## U-Net Architecture

U-Net is a convolutional neural network architecture designed for image segmentation tasks. It consists of an encoder-decoder structure that allows for high-resolution segmentation maps. The network was originally developed for biomedical image segmentation but can be applied to various segmentation problems.

### Dataset Overview

The dataset contains 274 flood images with corresponding water region masks. The masks are binary images where the water region is highlighted. The data is loaded from directories for images and masks:

- Images Directory: data/images/
- Masks Directory: data/masks/

The dataset is augmented to improve model performance and prevent overfitting, given the small number of samples.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/DaZeTw/U-NetForImageSegmentation.git
   cd U-NetForImageSegmentation
   ```

2. **Create a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Preprocessing**: The dataset is preprocessed to normalize the images and resize them to fit the input requirements of the U-Net model. Data augmentation (random rotations, flips, and zooms) is applied to increase the size of the training set.
2. **Model Training**: The model is trained using the U-Net architecture with a combination of cross-entropy and dice loss. The training is monitored using validation accuracy and loss.
3. **Visualization**: Visualize the results using Matplotlib.

### Model Architecture

The U-Net model consists of:

- **Encoder**: A series of convolutional and pooling layers that capture high-level features from the input images.
- **Decoder**: A symmetric series of up-sampling and convolutional layers that help in generating the segmentation mask.
- **Skip Connections**: Help retain fine-grained spatial information.

### Key Hyperparameters

- Input Size: 128x128 pixels
- Batch Size: 16
- Loss Function: Binary cross-entropy + Dice Loss
- Optimizer: Adam
- Learning Rate: 0.0001

### Running the Script

To run the main script, use:

```bash
python u-netsegmentation.py
```

### Google Colab Notebook

For an interactive version of this project, you can view and run the code in Google Colab: [Google Colab Notebook]([https://drive.google.com/file/d/1zHPCsRSwrSCzTfnrUMC1YcakywzPQ1YD/view?usp=sharing](https://drive.google.com/file/d/1zHPCsRSwrSCzTfnrUMC1YcakywzPQ1YD/view?usp=sharing))

## Results

### Training Performance

- Accuracy: 0.7374
- Validation Accuracy: 0.7912
- Loss: 0.4821
- Validation Loss: 0.3967

### Evaluation Results

The model performed well with data augmentation, achieving validation accuracy of ~79%. Further improvements can be made by using more advanced models or transfer learning techniques.

### Visualizations

- **Segmentation Maps**: Visualization of original images, ground truth masks, and model predictions.

## Conclusion

This U-Net-based flood water segmentation model demonstrates the effectiveness of deep learning for image segmentation tasks with limited data. The use of data augmentation helps generalize the model better. Future work can include the use of transfer learning and expanding the dataset to further improve the model's performance.
