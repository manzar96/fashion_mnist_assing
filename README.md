# fashion_mnist_assing

On the first part of this project we train a CNN network that learns to 
predict one of the 10 classes of Fashion-Mnist dataset (a dataset 
that contains grayscale images of 10 different classes of clothes).

<h3>Setup instructions:</h3>
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

<h3>Project Structure-Training-Evaluation:</h3>
<h4>Structure:</h4>
The <b>experiments</b> folder contains the main python scripts for each 
task of 
the assignment.
The <b>src</b> folder contains all the classes used in the main scripts.  
<h4>Training:</h4>
You can run any of the python scripts contained on the <b>experiments</b> 
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

<h3>Part 1:</h3>
This task focuses on the implementation of several models in order to learn to 
classify images illustrating clothes into the correct category.  More 
specifically, at first we experiment with an MLP and a CNN network on the 
Fashion-Mnist dataset. Afterwards, we try improving upon the aforementioned 
models by experimenting with other architecture (AlexNet,ResNet).
To reproduce our experiments run:

- task1a MLP training-evaluation:

> python experiments/task1a.py --lr 0.0001 --es 10

-task1b CNN training-evaluation:

>python experiments/task1b.py --lr 0.0001 --es 10 --not_use_early_stopping

-task1c AlexNet training-evaluation:

>python experiments/task1calexnet.py --lr 0.00001 

-task1c ResNet training-evaluation:
>python experiments/task1cresnet.py --lr 0.0001 

<h3>Part 2:</h3>
This task focuses on using the transfer learning technique on a 
pneumonia detection task. More specifically, we obtain the XCeption model,
a model pre-trained on the ImageNet dataset, and we fine-tune it on the
CXR dataset. Finally, we evaluate the model's performance using
5-fold cross-validation and discuss the obtained results.
To reproduce our experiments run:

- task2 Xception training-evaluation:

> python experiments/task2.py --lr 0.001 --kfold_dir checkpoints/draft/xception/


