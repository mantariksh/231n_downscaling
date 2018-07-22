# Enhancing Climate Data Resolution using Residual Networks

Climate data at around 1km2 resolution is essential for predicting local impacts of climate change, but current climate models are only able to produce 25-100km2 resolutions. Statistical downscaling methods do exist to increase the resolution of climate model outputs, but they vary in reliability and require specific local parameters. On the other hand, data-based approaches could produce more general and convenient methods for downscaling. The current state-of-the-art deep learning architecture for climate data super-resolution is called Super-Resolution Convolutional Neural Net (SRCNN), used [here](https://arxiv.org/pdf/1703.03126.pdf).

Here, we test the effectiveness of convolutional neural nets with residual connections (ResNets) as well as Generative Adversarial Networks (GANs) in downscaling climate data. We find that the ResNet is able to achieve lower test root-mean-squared error (RMSE) than SRCNNs on temperature and precipitation data, while the GAN performs worse in terms of RMSE than both the ResNet and SRCNN. However, the GAN is better than both the ResNet and SRCNN at resolving extreme precipitation events. We hypothesize that the adversarial loss in the GAN encourages it to produce sharper features and predict more extreme values, which we observe qualitatively in its outputs, while pure MSE loss encourages other models to produce more blurry features and moderate values. Overall, our results show that ResNets could be useful for general downscaling of climate data, while GANs could be useful for resolving extreme weather events.

[Drive link](https://drive.google.com/drive/u/1/folders/1fsSLVWnVBO82oWKeeOpxMPbA8szzYDTp)
Contains all our data as well as models that could not be uploaded to GitHub.

SRCNN.ipynb contains the implementation of the SRCNN2 model.

SRCNN_precip.ipynb contains the implementation of the SRCNNp model.

SRGAN_no_pixleShuffle.ipynb contains the implementations of the SRGAN and SRResNet models.

performance_metrics.ipynb contains the extreme events analysis code.
