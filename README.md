# Transportation Mode Detection con CNN

This repository contains my master's thesis work, where I explored the problem of transportation mode detection using Convolutional Neural Networks (CNNs). The primary goal was to surpass the results presented in some benchmark papers in the field.

# Table of Contents

 - Introduction
 - Dataset
 - Methodology
 - Results
 - References

# Introduction

Transportation mode detection is a key problem in the field of urban mobility and transportation research. Accurately identifying the mode of transport can have significant implications for urban planning, traffic management, and environmental research.

# Dataset
The dataset used for this project is Geolife. It's an open-source dataset that contains GPS trajectories of various users and their modes of transport.

[**Link to the Geolife dataset**](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/)

# Methodology

## Trajectory Creation

The GPS data from the Geolife dataset was processed to create trajectories. Each trajectory represents a sequence of GPS points, capturing the movement of an individual over time.

## Dataset Division: Stops and Moves

The dataset was divided into two main categories:
 - Stops: These represent periods where an individual remained stationary or moved very little.
 - Moves: These represent periods of active movement.
This division allowed for a more granular analysis of the data and provided insights into different transportation modes.

## Experimentation
Several experiments were conducted to determine the best features and model configurations. Initial experiments focused on individual features like speed, while subsequent experiments combined multiple features to enhance model performance.

## CNN Model
A Convolutional Neural Network (CNN) was employed to classify different modes of transport based on the GPS trajectories. The CNN was designed to capture spatial patterns in the data, making it well-suited for this task.

## Ensemble Approach
In addition to the standalone CNN model, an ensemble approach was also explored. This involved training multiple CNN models and aggregating their predictions to achieve a more robust and accurate classification.

# Results
The results obtained showed a significant improvement over the benchmark papers. The ensemble approach, in particular, demonstrated superior performance in several experiments.

## Detailed Results:
Speed Features:
Using only the Speed features as input (set X), the model achieved an accuracy of 86.33% and an F1 score of 79.82%. This indicates that considering only the Speed might be sufficient to achieve good results in certain experiments.

## Speed and Acceleration Features:
When the input set X was composed of both Speed and Acceleration features, the model achieved an accuracy of 89.25% and an F1 score of 85.44%. This suggests that the combination of these two features can enhance the model's performance compared to using just one feature.

## Dataset Composition:
For the dataset composed of both Stops and Moves, it's important to note that every time the CNN model was evaluated, the sequence stop id was also considered. Thus, when considering two kinematic features, technically three features were used as input for the CNN model. The dataset composed solely of Movs sometimes achieved results as good as the dataset with Stops. In certain instances, it even surpassed the results of the dataset composed of both Stops and Movs.

## Padding Experiments:
For the Movs dataset, better results were obtained when using 2, 3, 4, and 5 kinematic features. However, when only one feature was used, the results were quite similar, with the best performance achieved using 0-padding.

## Ensemble Approach:
The ensemble approach involved training multiple CNN models and aggregating their predictions to achieve a more robust and accurate classification. By leveraging the strengths of multiple models, the ensemble method aimed to reduce the variance and improve the overall performance.

The best result achieved using the ensemble approach was with the Speed and Acceleration features on the Movs dataset, where the model achieved an accuracy of 88.42% and an F1 score of 82.96%. This result underscores the effectiveness of the ensemble approach, especially when combined with the right feature set.


# References:

1. **Sina Dabiri, Kevin Heaslip**  
   *Inferring transportation modes from GPS trajectories using a convolutional neural network*,  
   Transportation Research Part C, (2018)

2. **Hugues Moreau, Andrea Vassilev, and Liming Chen**  
   *The Devil Is in the Details: An Efficient Convolutional Neural Network for Transport Mode Detection*,  
   IEEE transactions on intelligent transportation systems, (2022)

3. **Hancheng Cao, Fengli Xu, Jagan Sankaranarayanan, Yong Li and Hanan Samet**  
   *Habit2vec: Trajectory Semantic Embedding for Living Pattern Recognition in Population*,  
   IEEE transactions on mobile computing, (2020)

4. [**Scikit-Mobility documentation**](https://scikit-mobility.github.io/scikit-mobility/)

5. [**Ptrail documentation**](https://ptrail.readthedocs.io/en/latest/)

6. [**Geolife official user guide**](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/)