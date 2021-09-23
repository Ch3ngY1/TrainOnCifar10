# TrainOnCifar10
## Table of Contents
* [Validation errors](#validation-errors)
* [Training curves](#training-curves)
* [User's guide](#users-guide)
   * [Pre-requisites](#pre-requisites)
   * [Overall structure](#overall-structure)
   * [Hyper-parameters](#hyper-parameters)
   * [Data augmentation](#data-augmentation)
   * [CNN Strcuture](#cnn-structure)
   * [Training](#training)
   * [Result](#result)


## Validation errors
The highest validation accuracy of different LearningRate-BatchSize sets respectively. 

Lr-Batchsize | Highest Validation Acc | Lowest Validation Loss epoch
------- | ----------------------- | -----------------------
0.1-64 | 11.9% | 5
0.01-32(nan-inf) | 72.7% | 2
0.01-16 | 55.4% | 1
0.01-32 | 72% | 2
0.01-64 | 79.2% | 3
0.001-16 | 78.68% | 6
0.001-32 | 78.5% | 7
0.001-64 | 79.5% | 10

## Training curves
<table>
    <tr>
            <th>lr=0.01,batchsize=32,naninf</th>
            <th>lr=0.01,batchsize=16</th>
            <th>lr=0.01,batchsize=32</th>
            <th>lr=0.01,batchsize=64</th>
    </tr>
    <tr>
        <td><img src=https://github.com/Ch3ngY1/TrainOnCifar10/blob/master/mytensorboard/Fig/lr001bs32naninf.png width=100% /></td>
        <td><img src=https://github.com/Ch3ngY1/TrainOnCifar10/blob/master/mytensorboard/Fig/lr001bs16.png width=100% /></td>
        <td><img src=https://github.com/Ch3ngY1/TrainOnCifar10/blob/master/mytensorboard/Fig/lr001bs32.png width=100% /></td>
        <td><img src=https://github.com/Ch3ngY1/TrainOnCifar10/blob/master/mytensorboard/Fig/lr001bs64.png width=100% /></td>
    </tr>
</table>

<table>
    <tr>
            <th>lr=0.1,batchsize=64</th>
            <th>lr=0.001,batchsize=16</th>
            <th>lr=0.001,batchsize=32</th>
            <th>lr=0.001,batchsize=64</th>
    </tr>
    <tr>
        <td><img src=https://github.com/Ch3ngY1/TrainOnCifar10/blob/master/mytensorboard/Fig/lr01bs64.png width=100% /></td>
        <td><img src=https://github.com/Ch3ngY1/TrainOnCifar10/blob/master/mytensorboard/Fig/lr0001bs16.png width=100% /></td>
        <td><img src=https://github.com/Ch3ngY1/TrainOnCifar10/blob/master/mytensorboard/Fig/lr0001bs32.png width=100% /></td>
        <td><img src=https://github.com/Ch3ngY1/TrainOnCifar10/blob/master/mytensorboard/Fig/lr0001bs64.png width=100% /></td>
    </tr>
</table>

## User's guide
You can run main.py. You can test its convergence by command line: `CUDA_VISIBLE_DEVICES=9 python3 main.py --download True --epoch 25 --lr 0.01 --batchsize 64 --numtrain 4900` and train model by command line `CUDA_VISIBLE_DEVICES=9 python3 main.py --epoch 25 --lr 0.001 --batchsize 64 --numtrain 49000` and adjust these parameters to train model on different parameter-set

The saved tensorboard is saved at "LogDir", command line `tensorboard --logdir <LogDir>` is used to see the tensorboard curve.

### Pre-requisites
tensorboardX(2.4), torchvision(0.2.1), pytorch(1.4.0)

### Overall structure
There are four python files in the repository. main.py, CNN.py, check_accuracy.py, train_model.py.

main.py includes is used for setting parameters.
CNN.py defines CNN architecture.
check_accuracy.py is responsible for calculate accuracy and valiation loss.
train_model.py is responsible for training and validation. 

### Hyper-parameters
**lr**: float. The fixed learning rate for optimizer.

**batchsize**: int. How many batches to train and validate the model.

**epoch**: int. The number of training round.

**download**: bool. Whether to download Cifar dataset(default=False)

**numtrain**: int. The number of images for training, 4900 is for testing whether model can converge, 49000(default) is for training

**momentum**: float. The fixed momentum for optimzer

### Data augmentation
**RandomCrop**: Randomly crop images "T.RandomCrop(32, padding=4)" to 32x32 with padding=4.

**Normalize**: Normalize the input images for better performance.

**RandomHorizontalFlip**: Randomly horizental flip each image to increase generalization.

### CNN Structure
Here I use 3 convolution layers and 3 full-connection layers
```
model(
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Conv2d(3, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (layer2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (layer3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (FC1): Linear(in_features=4096, out_features=1024, bias=True)
  (FC2): Linear(in_features=1024, out_features=1024, bias=True)
  (classifier): Linear(in_features=1024, out_features=10, bias=True)
)
```

### Training
Run the following commands in the command line:
It is able to adjust --lr and --batchsize to training model on different LearningRate-BatchSize
```
CUDA_VISIBLE_DEVICES=9 python3 main.py --download True --epoch 25 --lr 0.01 --batchsize 64 
```

### Result
1. Larger batchsize leads slower non-converge point.
2. Smaller learning rate has slower convergence and higher best-validation-acc.
3. Large learning rate(lr=0.1 here) results in model misfit.
4. Relatively large learning rate(0.01 here) leads INF after some layers which leads Nan in loss, by replacing INF to 80 and Nan to 0, the best-validation-acc increases slightly.
   
