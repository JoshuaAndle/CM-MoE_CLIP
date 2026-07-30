This ReadMe covers the organization and use of the accompanying code for Leveraging Subnetworks for Continual Online Learning Under Distribution Shift

The directory layout is as follows:

 - Relative Root
 	- data
 		- <Datasets>
 		- RotMNIST is provided
 		- Offline PMNIST and CIFAR-100 can be generated with the functions in clstreams.py
 			- Afterwards the Online datastream prep ipynb script can be used to convert them to the necessary online stream formats
 	- checkpoints
 		- <Saved model checkpoints from training>
 	- src
 		- logs
 			- <text outputs from experiment runtime>
 		- Scripts and .ipynb files for running experiments
 		- AuxiliaryScripts
 			- <Python scripts implementing executing experiments>


.ipynb Files:
	 The provided .ipynb file contains sample code for running experiments, evaluating results, and preparing online data streams

Experiment Scripts:
	Main.py: Directs the Online Learning of Baseline and Pruning-only methods
	Replay Main.py: Directs the Online Learning of Replay-only methods

Auxiliary Scripts:
	Manager.py: Manager is the primary class which orchestrates pruning, training, and weight sharing of the model
	Network.py: A class which holds the model and performs operations on the model such as switching out the classifier for each task
	clmodels.py: Defines the VGG16 and Modified ResNet-18 networks used in this work
	clstreams.py: Loads the necessary datasets. Includes the setup code for the offline datasets dataset.
	DataGenerator.py: A simple data generator for pytorch
	Utils.py: Performs miscellaneous functions including mask operations and activation collections.

Note: The names of the data streams in the paper differ slightly from the code:
OMCP is "online_mixed_cifar_pmnist", OMCPR is "online_cifar_rotmnist", and OMCPR-20 and 30 are "online_cifar_jump20rotmnist" and "online_cifar_jump30rotmnist"