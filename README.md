# Active-mixup
Code for Active Mixup in 2020 CVPR

## Preparation:
1. You need a well-trained teacher model with sufficient accuracy, such as top-1 ~93% for CIFAR-10.
2. For CIFAR-10, convert CIFAR10 Data to PNG or JPG format. (https://github.com/knjcode/cifar2png.git) For example, run git clone https://github.com/knjcode/cifar2png.git at current directory. Follow the instruction to install it and make sure all converted png images are stored in active_mixup/cifar2png/cifar10png/ , which contains train and test folders.
3. Locate train data directory and specify root in 00_data_prep.sh with it. For example, with downloaded cifar10 above, we use root=./cifar2png/cifar10png/train .

## Run:
************* Stage 0 ********************************

1. 00_data_prep.sh (randomly select real images and mix them to a candidate pool)
2. 01_data_prep.sh (query teacher model and obtain the first real image training set)
3. active_train.sh (train the model with real images data)

************* Stage 1 ********************************

4. 10_data_prep.sh (query student network)
5. 11_data_prep.sh (query teacher model and extend training data)
6. active_rain.sh  (train the model with new dataset)

************* Stage 2 (repeat the above) *************

7. 10_data_prep.sh
8. 11_data_prep.sh
9. active_train.sh

************* Stage 3 (repeat the above) *************

...


## Note:
### For 00_data_prep .sh or .py
1. Specify the number of real_images with 1000, 2000, ...
2. The code will generate combination indieces pool. To save the space, the dumped pkl only stores paths instead of arrays.
3. The real images will be dumped into ./images folder.

### For 01_data_prep .sh or .py
1. Specify checkpoint in py file for teacher query model. For example, checkpoint = torch.load('./checkpoint/cifar10_vgg16_teacher.pth')
2. In my_loader, specify real_img_query_loader with the pkl of real images. For example, datainfo = pickle.load(open('./images/cifar10_real_images_1000.pkl', 'rb'))
3. The code will generate real image query results and dump the results into query folder ./images/query. For example, ./images/query/cifar10_query_label_###.pkl  

### For 10_data_prep .sh or .py
1. Specify checkpoint in py file for student query model. For example, checkpoint = torch.load('./active_student_models/cifar10_vgg_student_model_1000.pth'). Note that you need at least one student model for query.
2. Specify the number of selected mixed images for each stage. For example, local_indices = np.argsort(unc_tags)[:10000], which implies that we select top uncertain 10K images. 
3. In my loder, specify mix_img_query_loader with the pkl of mixed image candidate pool. For example,  datainfo = pickle.load(open('./images/cifar10_mix_images_499000.pkl', 'rb')).
4. The code will generate two sets, including left unlabeled images and selected images. To save the space, unlabeled images are dumped in the format of paths. Selected images here haven't queried teacher model yet.

### For 11_data_prep .sh or .py
1. Specify checkpoint in py file for teacher query model. For example, checkpoint = torch.load('./checkpoint/cifar10_vgg16_teacher.pth') 
2. Specify current labeled images. For example, datainfo = pickle.load(open('./images/query/cifar10_query_label_1000.pkl', 'rb')) 
3. In my loder, specify active_query_loader with the pkl of selected images from 10_data_prep.py. For example, datainfo = pickle.load(open('./images/query/cifar10_query_new_label_10000.pkl', 'rb')).
4. The code will concatenate the new query labeled images to the previously obtained labeled images, dumping the new query set. 

### For active_train .sh or .py
1. Specify lr, model_out, and log generation path. 
2. In my loader, specify active_learning_loader with obtained training set. For example, datainfo = pickle.load(open('./images/query/cifar10_query_label_1000.pkl', 'rb')).
3. The code will generate a well-trained model with current training set.

### P.S.
1. The first set of real images are randomly selected and training process is stochastic, so the training accuracies can show variations (maybe +/- 2%).
2. Active learning is an incremental process, so the previous student model can affect the next student training. Some variation can occur.
3. You may try the example with run_00.sh and run_01.sh for Stage 0 and Stage 1 with 1000 images. Please remember to modify active_learning_loader in my loader after running run_00.sh for new training with run_01.sh.

If you have any questions, please feel free to contact me. Thanks very much for your interest and questions in advance.

daniel.wang@knights.ucf.edu

## Citation
```yaml
@inproceedings{wang2020neural,
  title={Neural networks are more productive teachers than human raters: Active mixup for data-efficient knowledge distillation from a blackbox model},
  author={Wang, Dongdong and Li, Yandong and Wang, Liqiang and Gong, Boqing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1498--1507},
  year={2020}
}
```
