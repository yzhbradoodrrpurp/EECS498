# EECS498

## Version

[EECS498 Fall 2019](https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2019)

## Assignment Details

### A1

- [PyTorch101](Assignments/A1/pytorch101.ipynb): The first assignment of A1 is a coarse introduction to PyTorch which includes the basic utilization of it. More specifically, it contains functions like `torch.tensor` , `torch.arange` , `torch.view` , `torch.reshape` , `torch.contiguous` , `torch.argmax` , `torch.argmin` , `torch.to` , `torch.zeros` , `torch.randn` and so on. Aside from that, it also demonstrates how to apply boolean indexing, array indexing, **broadcast mechanism** and running on GPUs.

- [kNN](Assignments/A1/kNN.ipynb): kNN means k Nearest Neighbors. It's a lazy-learning algorithm with no parameters to learn. The point of kNN is to find k nearest neighbors of a vectors which might indicate their similarity to some extent. In this assignment, you will implement kNN.

### A2

- [SVM Classifier and Softmax Classifier](Assignments/A2/linear_classifier.ipynb):In this assignment, you will implement a linear classifier and apply **multi-class SVM loss** and **Softmax loss** to it separately. The reference on how to calculate multi-class SVM loss and Softmax loss is [here](Notes/005LinearClassifier).
- [Two-Layer Nets](Assignments/A2/two_layer_net.ipynb): After gaining an insight into what an **activation function** does, you will implement a simple two-layer network just by stacking a linear layer, an activation function and another linear layer together. Besides, you will also train the model on CIFAR-10 dataset and tune hyper parameters like learning rate, regularization strength, batch size and number of epochs to achieve better results.

### A3

- [Fully-Connected Neural Networks](Assignments/A3/fully_connected_networks.ipynb): This is the further version of Two-Layer Nets that consists of much more linear layers and activation functions. In addition, you will also utilize **different gradient descent strategies like SGD, SGD + Momentum, RMSProp and Adam** to rev up the process and optimize the results. Last but not least, the assignment will also walk you through another regularization strategy called **Dropout** which differs between training time and test time. The network is also trained on CIFAR-10 dataset. The reference to gradient descent strategies is [here](Notes/006optimization.md) and reference to Dropout is [here](Notes/010ConvolutionalNeuralNetworks.md#Dropout).

- [Convolutional Neural Networks](Assignments/A3/convolutional_networks.ipynb): This assignment will walk you through the implementation of **convolutional layer, Kaiming Initialization, pooling layer and batch normalization both in 1D and 2D**. Finally you will build up your own deep CNN network on your own without the aid of PyTorch modules. You should see a huge progress in the accuracy of Image Classification trained on CIFAR-10 dataset due to the powerful CNN architecture and know why CNN is far better than just MLP. The reference to CNN is [here](Notes/010ConvolutionalNeuralNetworks.md).

### A4

- [PyTorch AutoGrad and NN Modules](Assignments/A4/pytorch_autograd_and_nn.ipynb): You aren't allowed to use autograd and PyTorch modules by now which are definitely going to spare you a whole lot of time and efforts! This assignment is designed to get you well-acquainted with different levels of PyTorch including `autograd` , `torch.nn.functional` , `torch.nn.Module` , `torch.nn.init` , `torch.optim` , `torch.nn.Sequential` , `torch.nn.Parameter` and so on. After you are familiar with all these hands-on tools, you will implement a really really deep **ResNet**.
- [RNN, LSTM, Attention and Image Captioning](Assignments/A4/rnn_lstm_attention_captioning.ipynb): In this part, you will implement **Vanilla RNN, LSTM and Attetion LSTM** and then train them on the COCO dataset to do Image Captioning task. In oder to do that, you are also required to implement a **word embedding layer** to transform the ID of a certain word into a word vector. The reference to RNN is [here](Notes/013RecurrentNeuralNetworks.md) and reference to Attention is [here](Notes/014Attention.md).

- [Network Visualization and Adversarial Examples](Assignments/A4/network_visualization.ipynb): In the past, what we usually do is to apply gradient descent to the model parameters. However in this part, what we do basically is to **keep the parameters of a pre-trained model unchanged and apply back propagation to the input image**. We can use this method to check which part of the image influences the outcome of the model the most through a **saliency map**. On top of that, this method is also applied to generate **adversarial examples** to examinate the robustness of the model.
- [Style Transfer](Assignments/A4/style_transfer.ipynb): This is a simple version of a hands-on and cutting-edge work from ["Image Style Transfer Using Convolutional Neural Networks" (Gatys et al., CVPR 2015)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf). What it intends to do basically is to extract the content of a image, extract the style of another image and combine the content and the style together to generate a new image. For instance, image A is a photo of the Great Wall and image B is Starry Night by Van Gogh. Image Tranfer extracts the content of the Great Wall and the style of Starry Night thus yielding the Starry-Night-style Great Wall.

### A5

- [Single-Staged Object Detector](Assignments/A5/single_stage_detector_yolo.ipynb): This assignment will walk you through a R-CNN and a fast R-CNN to do Object Detection task on PASCAL VOC 2007 dataset.
- [Two-Staged Object Detector](Assignments/A5/two_stage_detector_faster_rcnn.ipynb): This assignment introduces a new way to define RoI which is through a region proposal network.

### A6

- [Generative Adversarial Networks](Assignments/A6/generative_adversarial_networks.ipynb): The content of this assignment is about **Generative Adversarial Networks** or GANs for short. It will demonstrate how to stack up a **generator** and a **discriminator**. Aside from that, it will also talk about how to implement the generator loss and discriminator loss in two different approaches. Since GANs are extremely sensitive and finicky to hyperparameters, they are trained on MNIST dataset for the sake of simplicity. The reference to GANs is [here](Notes/018GenerativeModels.md/#Generative-Adversarial-Networks).

## Schedule

2025.2.7: Initialize the repository.

### A1

- 2025.2.8: Finish [PyTorch101](Assignments/A1/pytorch101.ipynb)
- 2025.2.9: Finish [kNN](Assignments/A1/kNN.ipynb)

### A2

- 2025.2.20: Finish [SVM Classifier and Softmax Classifier](Assignments/A2/linear_classifier.ipynb)
- 2025.2.26: Finish [Two-Layer Nets](Assignments/A2/two_layer_net.ipynb)

### A3

- 2025.3.2: Finish [Fully-Connected Neural Networks](Assignments/A3/fully_connected_networks.ipynb)
- 2025.3.7: Finish [Convolutional Neural Networks](Assignments/A3/convolutional_networks.ipynb) 

### A4

- 2025.3.9: Finish [PyTorch AutoGrad and NN Modules](Assignments/A4/pytorch_autograd_and_nn.ipynb)
- 2025.3.13: Finish [RNN, LSTM, Attention and Image Captioning](Assignments/A4/rnn_lstm_attention_captioning.ipynb) 

- 2025.3.14: Finish [Network Visualization and Adversarial Examples](Assignments/A4/network_visualization.ipynb) 

- 2025.3.15: Finish [Style Transfer](Assignments/A4/style_transfer.ipynb) 

### A5

- 2025.3.20: Finish [Single-Staged Object Detector](Assignments/A5/single_stage_detector_yolo.ipynb) 
- 2025.3.21: Finish [Two-Staged Object Detector](Assignments/A5/two_stage_detector_faster_rcnn.ipynb) 

### A6

- 2025.3.24: Finish [Generative Adversarial Networks](Assignments/A6/generative_adversarial_networks.ipynb) 

2025.3.24: Finish all the assignments, congratulations.
