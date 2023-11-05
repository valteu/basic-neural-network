# Basic Neural Network Implementation

This repository contains a basic neural network implementation in Python using NumPy. The neural network is designed with customizable layers for flexibility in building different architectures.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Example](#example)
- [License](#license)

## Installation

1. Clone the repository to your local machine:
```bash
git clone https://github.com/yourusername/basic-nn.git
```
2. Navigate to the project directory:
```bash
cd basic-nn
```
3. Ensure you have NumPy and Handout installed. You can install them using pip:
```bash
pip install numpy handout
```
You are now ready to use the Basic Neural Network.
## Usage
You can use this basic neural network implementation in your Python projects by following these steps:

1. Import the necessary modules:
```python
import numpy as np
import handout
import json
from Layer import Layer, Linear, ReLu, Sigmoid
from Network import Network
```
2. Create a neural network architecture by instantiating different layers (Linear, ReLu, Sigmoid) and specifying the input and output dimensions.

3. Initialize the network with the desired architecture, and optionally provide weights and biases for the layers.

4. Train the network by providing your data and targets using the train method, specifying the number of epochs and learning rate.

5. Test the network on new data using the test method.

6. Save the learned weights and biases to a JSON file for later use, if needed.

## Architecture
### The neural network architecture consists of the following key components:

- Layer: A base class for defining the structure of each layer in the network. You can derive custom layer types from this class.

- Linear: A fully connected layer that performs a linear transformation.

- ReLu: A rectified linear unit (ReLU) activation layer.

- Sigmoid: A sigmoid activation layer.

- Network: The main class for building and training the neural network. It allows you to specify the network architecture and perform forward and backward passes.

## Example
An example script is provided in the main.py file to demonstrate how to use this neural network for regression tasks. It generates random 1D data and targets, creates a network with a simple architecture, trains the network, and tests it on new data.

To run the example, execute the following command:

```bash
python main.py
```
This script can be easily adapted for your specific use cases and data.

## License
This project is licensed under the [MIT License](LICENSE).
