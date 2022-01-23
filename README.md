# Object Detection in an Urban Environment

### Project overview
The main objective of this project is to conduct a training and evaluation process, to do so the images has to be combined with its respective bounding boxes data from the tf records to be visualized and analyzed for the next steps. 

### Set up
In my case, I was using the virtual desktop provided by Udacity, therefore, the GPU was setup on the environment. I started with launching the Jupyter Notebook from the Terminator window. Terminator was also used to run all other functions in this repository.

### Dataset
#### Dataset analysis
As explained multiple times in the course, we should become one with the data we have, which means that we need to know the quality and the quantity of what we have. Therefore, the `Exploratory Data Analysis.ipynb` Notebook was used to implement functions from dependencies that will help in vizualizing the data we have, such as `display_instances(batch)`
The following are some random images from the dataset:

![](https://github.com/oalsheikh2/nd013-c1-vision-starter/blob/main/output.png)
![](https://github.com/oalsheikh2/nd013-c1-vision-starter/blob/main/output2.png)
![](https://github.com/oalsheikh2/nd013-c1-vision-starter/blob/main/output3.png)
![](https://github.com/oalsheikh2/nd013-c1-vision-starter/blob/main/output4.png)
![](https://github.com/oalsheikh2/nd013-c1-vision-starter/blob/main/output5.png)
![](https://github.com/oalsheikh2/nd013-c1-vision-starter/blob/main/output6.png)
![](https://github.com/oalsheikh2/nd013-c1-vision-starter/blob/main/output7.png)
![](https://github.com/oalsheikh2/nd013-c1-vision-starter/blob/main/output8.png)
![](https://github.com/oalsheikh2/nd013-c1-vision-starter/blob/main/output9.png)
![](https://github.com/oalsheikh2/nd013-c1-vision-starter/blob/main/output10.png)

#### Cross validation
Prior to cross validation, the quantitative measures of the dataset was analyzed to study the class distribution across the tf records, keeping in mind that we have 3 classes. After a careful study it was determined that the data should be split by running the `create_splits.py` in a ratio of 70%, 20%, and 10% for training, validation, and testing respectfully. This ratio was chosen to be optimal due to the diverse similarity across the data, such as most having cars and roads, with the variation being the brightness and color.

![](https://github.com/oalsheikh2/nd013-c1-vision-starter/blob/main/Screenshot2022-01-23125659.png)

### Training
#### Reference experiment
Data augmentation decreased the loss values, but I also chose to use Adam optimizer with the cosine decay function to prevent losses from getting stuck at the local minmum. I also adjusted the learning rate and the regularizer's weight decay value.
* a training process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/improved4/ --pipeline_config_path=experiments/reference/improved4/pipeline_improved4.config
```
Once the training is finished, launch the evaluation process:
* an evaluation process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/improved4/ --pipeline_config_path=experiments/reference/improved4/pipeline_improved4.config --checkpoint_dir=experiments/reference/improved4/
```
![](https://github.com/oalsheikh2/nd013-c1-vision-starter/blob/main/Screenshot_2022-01-23TensorBoard.png)


#### Inference
Multiple pipeline configurations were expiremented on, however, `improved4.config` showed the best results due to the lower learning rate
![](https://github.com/oalsheikh2/nd013-c1-vision-starter/blob/main/animation.gif)
