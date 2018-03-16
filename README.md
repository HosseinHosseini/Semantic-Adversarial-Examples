# Semantic Adversarial Examples

Deep neural networks are known to be vulnerable to adversarial examples, i.e., images that are maliciously perturbed to fool the model. Generating adversarial examples has been mostly limited to finding small perturbations that maximize the model prediction error. Such images, however, contain artificial perturbations that make them somewhat distinguishable from natural images. This property is used by several defense methods to counter adversarial examples by applying denoising filters or training the model to be robust to small perturbations. 

In our paper, we introduced a new class of adversarial examples, namely *Semantic Adversarial Examples*, as images that are arbitrarily perturbed to fool the model, but in such a way that the modified image semantically represents the same object as the original image. We developed a method for generating such images, by first converting the RGB image into HSV (Hue, Saturation and Value) color space and then randomly shifting the Hue and Saturation components, while keeping the Value component the same. This approach generates mostly smooth and natural-looking images that represent the same object, but with different colors. 

The code implements the attack on VGG16 network and CIFAR10 dataset. The pretrained weights of VGG16 network can be downloaded from [here](https://github.com/geifmany/cifar-vgg). The maximum number of trials is set to 1000. The results show that, for about 94.5% of CIFAR10 test images, it is possible to change the colors such that the modified image is misclassified by the model. 

Paper can be found here:  
**Semantic Adversarial Examples**  
*Hossein Hosseini and Radha Poovendran*  
<sub>Network Security Lab (NSL), Department of Electrical Engineering, University of Washington  
Link
