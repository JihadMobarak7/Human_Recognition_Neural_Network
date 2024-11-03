# Human_Recognition_Neural_Network

This repository contains a project that applies a simple feed-forward neural network to perform human activity recognition based on smartphone sensor data. The dataset, sourced from Kaggle, includes various attributes derived from the accelerometer and gyroscope of a smartphone, labeled with activities like walking, sitting, standing, etc.

## Project Overview
The goal of this project is to use data collected from smartphone sensors to accurately predict the type of activity performed by the subject. This involves preprocessing the data, designing a neural network, training the model, and evaluating its performance.

## Dataset
The dataset used is the "Human Activity Recognition Using Smartphones Dataset" which can be found on [Kaggle](https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones). It includes time-series data from 30 subjects performing activities of daily living (ADL) while carrying a waist-mounted smartphone with embedded inertial sensors.

## Features
- **Data Preprocessing**: Scripts to clean and prepare the data for training.
- **Neural Network Model**: A Python implementation of a feed-forward neural network using TensorFlow/Keras.
- **Training Script**: Code to train the neural network on the processed data.
- **Evaluation Script**: Code to evaluate the model's performance on a test set.

## Requirements
This project requires Python and several Python libraries which are listed in `requirements.txt`. To install these dependencies, run the following command:

## Usage
To run this project, follow these steps:
1. Clone this repository:
2. Install the required packages:
3. Run the Jupyter notebook to train the model and make predictions:

## License
This project is licensed under the MIT License - see the `LICENSE.md` file for details.
