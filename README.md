# PFLD

## 

This is another PFLD implementation, we changed the normal used MTCNN detector to retinaface. Final result like:

![image-20200612131211787](https://i.loli.net/2020/06/12/Ts6ZmdSjcQ18RwJ.png)



tAlso, be note that, this effect can be directly show by run:

```
python3 demo.py
```

We have put models inside repo (include retinaface and PLFD)

this model can runs on realtime, if you using Retinaface tensorrt version acceleration which can be obtained in our toolchain.

**this repo also include vovnet backbone for PLFD model.**. vovnet performances better than mobilenets on GPU.



## Dataset Download

### 300W Dataset
The 300W Dataset consists of 3148 training images and 689 testing images. All the separate components of the dataset can be downloaded [here](https://ibug.doc.ic.ac.uk/resources/300-W/). The training set consist of the entire AFW dataset, and the training set for LFPW and HELEN. The common testing set includes the testing sets of LFPW and HELEN, and the challenging set consists of the iBUG dataset. Together, they make the full testing dataset. 

To prepare the dataset, place all the train images in data/300W/train, and all the test images in data/300W/test.
```
$ cd data
$ python3 set_preparation_68.py
```
### WFLW Dataset
The WFLW Dataset consists of 7500 training images and 2500 testing images. It can be downloaded [here](https://pan.baidu.com/s/1paoOpusuyafHY154lqXYrA). 

To prepare the dataset, simply place WFLW_images folder within the WFLW folder. Then, run:
```
$ cd data
$ python3 set_preparation_98.py
```
## Training

Please specify --dataroot and --val_dataroot to link to the list.txt file containing the train and test annotations. For a 68-point dataset such as 300W, --num_landmark should be 68, while for a 98-point dataset such as WFLW, --num_landmark should be 98. You can change the backbone to either MobileNet or VoVNet with the --backbone argument. For simple training, create a folder named "1" within checkpoint, and run:
```
python3 train.py
```

### 

## Cite

```
PFLD: A Practical Facial Landmark Detector https://arxiv.org/pdf/1902.10859.pdf
```




