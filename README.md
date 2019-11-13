# MCDS-Capstone

1. Goal: Self-supervised learning
The goal of the project is to improve the performance of U-Net, a state-of-the-art image segmentation model, through self-supervised learning. Specifically, we apply registration, which is a way of mapping one brain to another and find point to point correspondence. These correspondences are used as positive pairs that should be close to each other in feature space. By first training a distance feature learner with constrastive correspondence loss, we aim to minimize the distance between positive pairs and maximize the distance between negative pairs in feature space. Consequently, we believe that the distance feature learner will give us better features and these features are used as input to a standard U-Net instead of the raw images.
