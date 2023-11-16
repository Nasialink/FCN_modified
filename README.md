# FCN_modified
Modifications of the code from [the implementation](https://github.com/IAmSuyogJadhav/3d-mri-brain-tumor-segmentation-using-autoencoder-regularization) of the "[3D MRI brain tumor segmentation using autoencoder regularization (FCN)](https://arxiv.org/abs/1810.11654)" paper by Andriy Myronenko.
We divided the main script into 3 scripts, to separately implement the individual phases of the segmentation process : 
1. functions.py: it contains custom functions made by the author of the code
2. preprocess.py: the preprocessing phase -> reading the images and creating the data arrays that will be used in later steps
3. train.py: it contains the random split of the data into training and testing set, as well as the training and saving of the model. During training, the training set is divided into a new training and validation set, while the proposed learning scheduler is also implemented
   In addition, we added the Evaluation.py script, to make predictions with the trained model. The code was also modified to support Tensorflow 2.0.

