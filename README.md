# fashion_mnist_assing

On the first part of this project we train a CNN network that learns to 
predict one of the 10 classes of Fashion-Mnist dataset (a dataset 
that contains grayscale images of 10 different classes of clothes).

###Setup instructions:
1. Clone the github repo: 
>git clone git@github.com:manzar96/fashion_mnist_assing.git

Do not forget to add your generated ssh keys at your github profile for 
cloning. 

2. Setup a virtual env for the project (outside of the project dir):
> cd ../

> python3 venv -m myvenv

Note that for doing this you must have python3 pre-installed.

3. Activate your virtual env and upgrade setuptools:
> source myvenv/bin/activate

> pip install --upgrade setuptools pip

4. Install all required packages:
> cd fashion_mnist_assing

> pip install -r requirements.txt

Note that we used a CUDA 11.1 driver  for that project. So the torch and 
torchvision packages installed are the proper ones for that driver. If you're 
facing troubles with the installation please remove the 'torch' and 'torchvision'
and install the default manually by running:
> pip install torch

> pip install torchvision

###Project Structure-Training-Evaluation:
####Structure:
The **experiments** folder contains the main python scripts for each task of 
the assignment.
The **src** folder contains all the classes used in the main scripts.  
####Training:
You can run any of the python scripts contained on the **experiments** 
folder or to create yours. For example, if you want to run the first task 
of the assignment run:
> export PYTHONPATH=./

> python experiments/task1.py

Several input arguments can also be specified for training and validation.
We mention the most important ones (you can find the rest in utils/parser.py)

- --lr 0.0001 (specifies the learning rate)
- --es 10 (specifies the number of epochs)
- --modelckpt (specifies the dir to save the model - checkpoints/draft is 
  used as default)
- --not_use_early_stopping (disables early stopping during training)
- --skip_val (skips validation process during training)

###Part 1:
 The first part of 