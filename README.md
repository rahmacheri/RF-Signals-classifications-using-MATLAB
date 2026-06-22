# RF-Signals-classifications-using-MATLAB
classifying wireless signals using deep learning.
to solve the classification problem, the Deep Canonically Correlated Autoencoder
(DCCAE) algorithm is implemented. The DCCAE is a multi-view-based algorithm which 
combines both DCCA architecture two autoencoders which uses two DNNs to extract 
nonlinear features for each view and the canonical correlation between the extracted 
features f (X) and g(Y) is maximized. And two encoders minimizing the reconstruction 
errors.
DATA GENERATION:
Before discussing the implementation of such model to solve the RF signals 
classification problem, we need to generate training data. To do so we use MATLAB
toolboxes to synthesizes RF signals. In this project three main toolbox are used :
We generate 5G signals using 5G Toolbox, LTE signals using LTE Toolbox, and WLAN 
signals using WLAN Toolbox.
The helperSpecSenseTrainingData function in Spectrum Sensing with Deep Learning to 
Identify 5G, LTE, and WLAN Signals is used to generate dataset. 
Due to the time constraints only a batch of 700 spectrogram images were generated for 
each class. And therefore, data augmentation techniques were also employed to 
artificially expand the training set.
MODEL ARCHITECTURE:
The clean spectrogram images are treated as view 1, and the noisy ones fed as the 
second view. The goal of the DCCAE is to be able to strip away the noise and perfectly 
separate the data. 
Testing: the unseen data is fed to the first trained encoder that compresses it to a latent 
space of dimension K. then the encoder output is fed to a simple SVM classifier that 
identify and classify the RF signal.

 this project implement DCCA architecture algorithm described in the paper:

  Weiran Wang, Raman Arora, Karen Livescu, and Jeff Bilmes. 
  On Deep Multi-View Representation Learning. 
  The 32nd International Conference on Machine Learning (ICML 2015).
