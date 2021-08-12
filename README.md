# Dermoscopic Imaging Semantic Segmentation
Local Attribute Segmentation of pigmented skin lesion images.

*This project constitutes a bachelor thesis in Sound and Image Engineering, carried out during the first semester of 2021. The notebook is implemented in PyTorch*. 

It is framed in the ISIC 2018 Challenge, a public challenge that took place in 2018 and is available here: https://challenge.isic-archive.com/landing/2018

A modified ad hoc version of ISIC DB 2017 was used in order to evaluate the proposal, which was stored in my personal Google Drive account and mounted in the notebook. The folder structure is explained at the end of this README- The original version of the dataset is available here: https://challenge.isic-archive.com/data 


We propose a Fully Convolutional U-net like model with emphasis on the preprocessing of the images and the feature extraction path, that performs multiclass semantic segmentation in order to localize the local structures of a pigmented skin lesion. These structures are: milia-like cysts, pigment networks, negative networks, streaks and globules.   
![Structures](https://github.com/CesarCaramazana/DermoscopicSegmentation/blob/main/images/structures.PNG)

Image source: https://dermoscopedia.org

<h2> Model description </h2>

From the beginning, **U-Net** was considered as the starting point for our proposal. The original arquitecture was modified in order to fit the requirements of the ISIC dataset (output number of classes and input resolution), and Resnet101 was used as backbone (pre-trained in ImageNet). Additionally, the last volume of the encoder was incorporated an **Atrous Spatial Pyramid Pooling** block (ASPP), from Deeplab v3, with dilation rates 2, 3 and 4, which slighyly improved the results. The last implementation carried out was the replacement of Resnet101 by **Inception v3**, which parallelizes convolutions and significantly reduces the number of parameters, something important given the computational power limitations of Google Colab. The arquitecture is shown in the following figure:

![arq](https://github.com/CesarCaramazana/DermoscopicSegmentation/blob/main/images/unet_inception.PNG)


The design of the **loss function** was approached with the highly pronounced class imbalance problem of ISIC DB in mind. We used Cross Entropy with class weights and tried out the Focal Loss for further penalization of easy samples (although the combination of both didn't provide better results than CE with weights alone). The coefficients are calculated as the inverse number of samples and normalized. 


The **preprocessing** of the images consists in three operational blocks: 

1. Elimination of background pixels. 
2. Resize + Random low resolution cropping (128x128).
3. Data augmentation (random flips and color jitter).

<img src="https://github.com/CesarCaramazana/DermoscopicSegmentation/blob/main/images/preproc_pipeline.png?raw=True" width="731px">


The output masks were **postprocessed** with morphological operations (erosion + dilation) in order to reduce the amount of False Positives generated by the weighting of the loss function.



<h2> Results and Conclusions </h2>

We achieved an average Jaccard score of 0.0896 in the test set, and an average Area under the curve value of 0.6986. These results are relatively low (with respect to the proposals submitted to the ISIC Challenge: https://challenge.isic-archive.com/leaderboards/2018) due to the class imbalance problem and the limitations of memory usage of Colab. We couldn't work with a higher input resolution without compromising the batch size. However, the architecture has potential for dermoscopic image segmentation. Due to the visual characteristics of the database, the emphasis placed on the feature extraction path has been key in the advancement of the project, and so the model can be further used in a scenario where more computational resources are available.

Some examples of the predicted masks (for each pair, the upper row) compared to the Ground Truth (lower row):

<img src="https://github.com/CesarCaramazana/DermoscopicSegmentation/blob/main/images/output.png?raw=True" width = "500px">

*The columns represent the five labeled structures in this order: cysts, negative network, pigment network, streaks and globules*.

The IoU of "pigment network", the class with the most number of samples, is 0.25, a significant margin with respect to streaks, the class with the least number of samples, in which the model achieves an IoU score of 0.001 (almost any pixel is classified correctly). Dermoscopic images, as well as other types of medical imaging, have an intrinsic difficulty that forces us to adopt a defensive approach to the resolution of the problem ('compensate' for class imbalance, 'mitigate' small datasets, 'reduce' memory usage, etcetera). A qualitative and quantitative improvement in ISIC DB would of course return better results. 


The main line of future work is the shift to a **Weakly Supervised Learning** scenario, in order to train the FCN with fewer (or none) ground truth labels.


<h2> Code description </h2>

The code was fully implemented in a Google Colab notebook (unetInception_final.ipynb), writen in PyTorch. The main libraries used are:

<ul>
  <li> Matplotlib </li>
  <li> CV2 </li>
  <li> Torchvision </li>
  <li> Sklearn </li>
  <li> Numpy </li>
</ul>

The notebook was synched to my Google Drive account, where the dataset was stored, following this folder structure:

<pre>
<code>
  db_isic/
      train.txt
      val.txt
      test.txt
      
      idx/
          isic_2017.mat
      
      ISIC-2017_Training_Data/
          ISIC_0000421.jpg
          ...          
      ISIC-2017_Training_Part2_GroundTruth/
          gtann/
              ISIC_0000421.mat
              ...
      ISIC-2017_Validation_Data/
          ISIC_0006651.jpg
          ...        
      ISIC-2017_Validation_Part2_GroundTruth/
          gtann/
            ISIC_0006651.mat
            ...
      ISIC-2017_Test_v2_Data/
          ISIC_0016072.jpg
          ...
      ISIC-2017_Test_v2_Part2_GroundTruth/
          gtann/
              ISIC_0016072.mat
    
</code>
</pre>

*The dataset class <code> db_isic_Dataset(torch.utils.data.Dataset)</code> may need to be modified in order to fit the folder structure of your Google Drive. In the code, /db_isic is a subfolder of /My_Drive, so it is accessed via this path:*.

<pre>
<code>  
  if(self.subset == 'train'):
      self.imgRoot = '/content/drive/My Drive/db_isic/ISIC-2017_Training_Data/'
      self.gtRoot = '/content/drive/My Drive/db_isic/ISIC-2017_Training_Part2_GroundTruth/gtann/'
      self.dataDir = '/content/drive/My Drive/db_isic/train.txt' 
</code>
</pre>

The text files with the indices of the data (train.txt, val.txt and test.txt) are not included in the original ISIC dataset, but are available in this repository. 

The checkpoint paths also need to be manually specified. In my implementation, I saved the models in a folder called /Checkpoints, using .tar file extension.

<pre>
<code>
checkpoint_path = "/content/drive/My Drive/Checkpoints/model.tar"

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
</code>
</pre>

The loss function and the model arquitectue cells are also available in the Python files <code>model.py</code> and <code>loss.py</code>
