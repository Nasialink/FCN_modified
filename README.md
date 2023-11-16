# FCN_modified
Modifications of the code for the neural network based on the "3D MRI brain tumor segmentation using autoencoder regularization (FCN)" paper by Andriy Myronenko.
We divided the main script into 3 scripts, to separately implement the individual phases of the segmentation process : 
1. functions.py: it contains custom functions made by the author of the code
2. preprocess.py: the preprocessing phase -> reading the images and creating the data arrays that will be used in later steps
3. train.py: it contains the random split of the data into training and testing set, as well as the training and saving of the model. During training, the training set is divided into a new training and validation set, while the proposed learning scheduler is also implemented
   Finally, we added the Evaluation.py script, to make predictions with the trained model.
