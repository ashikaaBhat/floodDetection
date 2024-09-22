Flood Detection Using U-Net Model
Project Overview
This project aims to detect flood-affected regions in satellite images using a U-Net model, a type of Convolutional Neural Network (CNN) designed for image segmentation tasks. Flood detection is crucial for disaster management and mitigation, allowing authorities to respond quickly and allocate resources efficiently. By leveraging multi-spectral satellite data, this project provides a precise and automated solution for identifying flooded areas.

Dataset
The dataset used in this project comprises satellite images with multiple spectral bands (12 bands per image). Each band represents different ranges of the electromagnetic spectrum, allowing for more detailed analysis of the earth's surface. These images provide essential information for identifying water bodies, urban areas, vegetation, and other geographical features.

Input: Each image in the dataset is of size 128x128 with 12 spectral bands.
Output: The goal is to segment the image into flooded and non-flooded regions, producing a binary mask.
U-Net Architecture
The model architecture follows a typical U-Net design with an encoder-decoder structure:

Encoder: The encoder compresses the input image by extracting feature maps through a series of convolutional layers and downsampling (max-pooling). The number of filters increases progressively, allowing the model to capture higher-level spatial features.

Bottleneck: At the center of the network, the bottleneck layer holds the most condensed representation of the image features.

Decoder: The decoder progressively upsamples the feature maps back to the original image size, concatenating the corresponding feature maps from the encoder (skip connections) to retain spatial information and improve segmentation accuracy.

Output Layer: The final layer uses a 1x1 convolution followed by a sigmoid activation function to generate a binary mask, indicating flood regions with a pixel-wise classification.

Model Training
The model was trained using binary crossentropy loss and the Adam optimizer. The input images are resized to 128x128 for efficient processing. The multi-band satellite images provide a rich set of features, enhancing the model's ability to distinguish between flooded and non-flooded regions.

Key training details:

Loss Function: Binary Crossentropy
Optimizer: Adam
Metrics: Accuracy
Results
The U-Net model performed well in segmenting flood-affected areas. The model's use of multi-spectral data allowed for more accurate flood detection, especially in regions where visual cues are limited to the human eye. Feature importance analysis showed that certain spectral bands played a crucial role in differentiating between water and land, particularly during flood events.

Additionally, we experimented with a Random Forest classifier to understand feature importance, but U-Net proved to be more efficient for image segmentation.
