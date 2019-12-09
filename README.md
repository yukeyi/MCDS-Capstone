# MCDS-Capstone

## 1. Goal: Improve Image Segmentation with Self-supervised learning
The goal of the project is to improve the performance of U-Net, a state-of-the-art image segmentation model, through self-supervised learning. Specifically, we apply registration, which is a way of mapping one brain to another and find point to point correspondence. These correspondences are used as positive pairs that should be close to each other in feature space. By first training a distance feature learner with constrastive correspondence loss, we aim to minimize the distance between positive pairs and maximize the distance between negative pairs in feature space. Consequently, we believe that the distance feature learner will give us better features and these features are used as input to a standard U-Net instead of the raw images.

## 2. File Structure
+ code4step1 & code4step2: code for Spring semester, not used.
+ code4step3
* KNN: code for offline compting KNN score from retrieved results, as well as saved results.
* MLP: code for directly using multi-layer percepton to do image segmentation from embedding.
* log: saved log in format of tensorboard log and numpy file.
* plot: code for plotting distance and loss.
* registration_visualization: jupyter notebook for generating registration maps and corresponding pairs.
* sift_usage_explore: offline code to test usage of opencv sift library.
* feature_learner_data_loader_util.py: data loader for featureLearner.
* feature_learner_model.py: model for featureLearner.
* featureLearner_spl.py: main function to train featureLearner in a self-paced learning manner (gradually changing size of bounding box for negative points).
* featureLearner.py: main function to train a featureLearner.
* generateParameterMapsLisa.py: code to generate registration maps.
* gpu_moniter.py: help function to moniter gpu usage on bridge.
* main3D_with_embedding.py: main function to train 3D Unet with embedding as input.
* main3D.py: main function to train baseline 3D Unet with original grey scale input.
* model_util.py: Common neural network utils for both featureLearner and Unet.
* unet_data_loader_util.py: data loader for training 3D Unet.
* unet_model.py: model for 3D Unet.
+ interpretability: code for interpret deep neural network for classification task.
+ Readings.md: Paper read for this Capstone project.
+ slurm: code for running slurm job.

## 3. Steps for Reproducing:
(a) We first register the images using … (b) Then we tried … to find correspondences, and we run into the issue that… and we solve it by … (c)Before training the feature learning network, we pertained it by minimizing the distance between output features and SIFT features. Then we borrow the idea of correspondence contrastive loss from universal correspondence network to train the feature learner network. Feature learner encourages the positive pairs to be close and negative pairs to be far from each other in the feature space. Positive pairs are selected using point correspondence from self-registration results, and negative pairs are selected in a dynamic manner. First, we feed the network with “easy” negative points, which are at least 20 apart from each other. Then we feed the network with negative point that are closer to each other. After obtaining the feature embedding of the brain data, we use it as the input to U-net. 

