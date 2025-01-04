
# Emotion detection


## Overview 

This project focuses on detecting emotions from images using deep learning techniques. The implementation includes a combination of custom Convolutional Neural Networks (CNNs) and transfer learning with pre-trained models. The final solution is deployed using Gradio to provide an interactive interface for emotion classification.

## Models Used

1. Custom Model (CNN)

A Convolutional Neural Network (CNN) was designed from scratch for emotion detection.

The architecture consists of multiple convolutional layers for feature extraction, followed by dense layers for classification.

This model provides baseline performance and serves as a foundation for further experimentation.

2. Custom Model + Image Augmentation

The custom CNN was enhanced by applying image augmentation techniques such as rotation, flipping, and zooming.

Image augmentation helps increase the diversity of the training dataset, reducing overfitting and improving generalization.

This approach improved the robustness of the model, especially for unseen data.

3. Transfer Learning with VGGNet

Pre-trained VGGNet was used as a feature extractor by freezing its convolutional layers and fine-tuning the fully connected layers.

VGGNet's deep architecture captures rich feature representations, leading to better accuracy on emotion detection tasks.

This method leverages the knowledge of pre-trained models to accelerate training and enhance performance.

4. Transfer Learning with ResNet

ResNet, a model known for its skip connections and ability to mitigate vanishing gradient issues, was utilized for transfer learning.

Fine-tuning ResNet allowed the model to learn task-specific features while benefiting from its pre-trained weights.

This approach demonstrated superior performance compared to the custom model and VGGNet in terms of accuracy and efficiency.

### Deployment with Gradio

Gradio was used to deploy the emotion detection system, providing a user-friendly web interface.

Users can upload an image, and the system will classify the emotion displayed in the image.

The interface includes features such as real-time predictions and a clear visualization of results.

## Conclusion

This project demonstrates the effectiveness of combining custom deep learning models with transfer learning techniques for emotion detection. The deployment through Gradio ensures accessibility and ease of use, making the system available for practical applications.

## Future Work

Explore multi-modal emotion detection by combining text and audio inputs.

Enhance deployment scalability with cloud-based platforms.

