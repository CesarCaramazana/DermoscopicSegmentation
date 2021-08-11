# Dermoscopic Imaging Semantic Segmentation
Local Attribute Segmentation of pigmented skin lesion images.

*This project constitutes a bachelor thesis in Sound and Image Engineering, carried out during the first semester of 2021*. 

It is framed in the ISIC 2018 Challenge, a public challenge that took place in 2018 and is available here: https://challenge.isic-archive.com/landing/2018

A modified ad hoc version of ISIC DB 2017 was used in order to evaluate the proposal, which was stored in my personal Google Drive account and mounted in the notebook. The original version of the dataset is available here: https://challenge.isic-archive.com/data 



We propose a Fully Convolutional U-net like model with emphasis on the preprocessing of the images and the feature extraction path, that performs multiclass semantic segmentation in order to localize the local structures of a pigmented skin lesion. These structures are: milia-like cysts, pigment networks, negative networks, streaks and globules.   
![Structures](https://github.com/CesarCaramazana/DermoscopicSegmentation/blob/main/images/structures.PNG)

Image source: https://dermoscopedia.org

From the beginning, **U-Net** was considered as the starting point for our proposal. The original arquitecture was modified in order to fit the requirements of the ISIC dataset (output number of classes and input resolution), and Resnet101 was used as backbone (pre-trained in ImageNet). Additionally, the last volume of the encoder was incorporated an **Atrous Spatial Pyramid Pooling** block (ASPP), from Deeplab v3, with dilation rates 2, 3 and 4, which slighyly improved the results. The last implementation carried out was the replacement of Resnet101 by **Inception v3**, which parallelizes convolutions and significantly reduces the number of parameters, something important given the computational power limitations of Google Colab. The arquitecture is shown in the following figure:

![arq](https://github.com/CesarCaramazana/DermoscopicSegmentation/blob/main/images/unet_inception.PNG)


The design of the **loss function** was approached with the highly pronounced class imbalance problem of ISIC DB in mind. We used Cross Entropy with class weights and tried out the Focal Loss for further penalization of easy samples (although the combination of both didn't provide better results than CE with weights alone). The coefficients are calculated as the inverse number of samples and normalized. 


The **preprocessing** of the images consists in three operational blocks: 

1. Elimination of background pixels. 
2. Resize + Random low resolution cropping (128x128).
3. Data augmentation (random flips and color jitter).

![preproc](https://github.com/CesarCaramazana/DermoscopicSegmentation/blob/main/images/preproc_pipeline.png?raw=True)

The output masks were **postprocessed** with morphological operations (erosion + dilation) in order to reduce the amount of False Positives generated by the weighting of the loss function.


We achieved an average Jaccard score of 0.0896 in the test set, and an average Area under the curve value of 0.6986. These results are relatively low due to the class imbalance problem and the limitations of memory usage in the GPUs available with Colab. We couldn't work with a higher input resolution without compromising the batch size, so a more defensive approached was taken. However, the architecture has potential for dermoscopic image segmentation. Due to the visual characteristics of the database, the emphasis placed on the feature extraction path has been key in the advancement of the project, and so the model can be further used in a scenario where more computational resources are available.
