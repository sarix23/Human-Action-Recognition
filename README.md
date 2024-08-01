# Human Action Recognition with Neural Networks

This project, developed by **Sara Nava** and **Giulia Saresini**, aims to develop a neural network capable of classifying seven human actions from the HMDB51 video dataset.

## Project Overview

The goal of this project is to build a robust neural network model for human action recognition. The model will be trained to classify videos into one of seven predefined human actions: **kick, sword, kiss, hug, shake_hands, fencing, punch**. The HMDB51 dataset contains videos showcasing various human actions, and this project focuses on a subset of these actions.

### Key Objectives

- **Data Preparation:** Preprocess the video data to extract relevant frames and normalize them for training models.
  
- **Model Development:** Design and train **3D CNN** and **LRCN** architectures suitable for video classification tasks.
  
- **Evaluation:** Assess the performance of the trained models using metrics such as accuracy and loss.

### Challenges

This project represents an initial attempt, acknowledging that the results obtained may not represent the optimal performance achievable. Due to **computational constraints** on our machines, we were limited in experimenting with a smaller number of parameters and simpler models, as more complex models caused kernel crashes. This is why we focused on classifying only 7 out of the 51 action classes available in the HMDB51 dataset. **Our goal was** not to surpass existing benchmarks but **to present a logically structured approach that reflects thoughtful decisions made under these constraints**.

## Dataset

The HMDB51 dataset includes a diverse range of human actions. Each action category consists of multiple video clips showcasing different instances of the action being performed. For more details, visit the [HMDB51 dataset page](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/).

## Project Structure

### Data Loading and Visualization

The dataset was downloaded manually and decompressed. We created a copy of the directory containing the videos and removed all videos from that copy that did not belong to the seven classes of interest.

### Dataset Preprocessing

We performed several preprocessing steps:

- Extracting and resizing frames from the videos.
- Normalizing the frames to have pixel values in the range [0, 1].
- Ensuring each video has a consistent number of frames.

### Model Development

#### 3D CNN Approach

We implemented a 3D Convolutional Neural Network (CNN) to process the spatiotemporal data from the video frames. The architecture includes multiple convolutional layers followed by max-pooling and dropout layers, a flattening layer, and dense layers.

#### LRCN Approach

We also implemented a Long-term Recurrent Convolutional Network (LRCN) which combines CNN and LSTM layers. The CNN layers extract spatial features from the frames, and the LSTM layers handle the temporal sequence modeling.

### Evaluation

We evaluated the trained models using accuracy and loss metrics. Due to computational constraints, we focused on achieving a balanced and well-structured approach rather than optimizing for maximum performance.

## Results

The models were able to classify the seven actions with varying degrees of accuracy. We provide visualizations of the training and validation loss and accuracy, as well as confusion matrices to illustrate the performance.

## Conclusion

This project demonstrates the process of developing neural networks for video classification under computational constraints. While the models achieved reasonable performance, there is room for improvement with more computational resources and further experimentation.

## References

- [HMDB51 Dataset](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
- [3D CNN for Video Classification](https://www.tensorflow.org/tutorials/video/video_classification)
- [Long-term Recurrent Convolutional Networks for Visual Recognition and Description](https://arxiv.org/abs/1411.4389)

## Acknowledgments

This project was developed as part of our coursework, and we thank our instructors and peers for their support and feedback.
