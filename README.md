
## Requirement:
### For training:
* scikit-learn
* pytorch

### For deployment on board:
* MDK5


# How to train our model
We implemented custom feature extraction and used  classifier from scikit-learn.
the training script is present in `model_training_design\train_model.py`
Our implementation is based on the feature extraction function implemented in the train_model.py file.
The best intercept from the classifier are extracted and used for deployment on the board.

To run the training, simply run the python script `train_model.py` with following parameters `path_data` and `path_indices`, 
the result will display the Coefficients learned from the classifier.


# How to validate UBPercept model 

### Note: He have not used X-CUBE-AI for generating our model, rather we implemented our own from scratch.

In the folder `deploy_design`, we have the design implemented in the `main.c` and used the template provided by `TEST_OwnModel.zip`. 
In the project, the file `main.c` contains all the functions needed for classification result.

### Note:  no need to implement `Model_Init()` method is the function as we do not have any neural network to be loaded, rather all our classifier coefficient are hardcoded in the `main.c`,  `extract_peaks_features_optimized_v3()` function.

We only impelemnted `aiRun` function to inference the input IEMG segment. The rest of the code, including data reception, data transmission and serial communication, is retained as a template. 


Use the same steps as defined in `Load Program to Board` section of [README-Cube.md](https://github.com/tinymlcontest/tinyml_contest2022_demo_example/blob/master/README-Cube.md)
Also mentioned below:
## Load Program to Board

1. Connet the boadr to computer.

    <img src="https://raw.githubusercontent.com/AugustZTR/picbed/master/img/image-20220627203515997.png" alt="image-20220627203515997" style="zoom: 25%;" />

2. Open project in MDK5 and build.

    ![build](https://raw.githubusercontent.com/AugustZTR/picbed/master/img/build.png)

3. Check if the debugger is connected.

    First, click ***Options for Target***.

    ![targets](https://raw.githubusercontent.com/AugustZTR/picbed/master/img/targets.png)

    Then switch to <u>Debug</u> and click ***Settings***.

    <img src="https://raw.githubusercontent.com/AugustZTR/picbed/master/img/debug.png" alt="debug"  />

    If the debugger is connected, you can see the IDCODE and the Device Name. 

    <img src="https://raw.githubusercontent.com/AugustZTR/picbed/master/img/swdio.png" alt="swdio"  />

    Finally, switch to <u>Flash Download</u> and check <u>Reset and Run</u>

    ![full chip](https://raw.githubusercontent.com/AugustZTR/picbed/master/img/full%20chip.png)

4. Now you can load program to the board.

    ![load](https://raw.githubusercontent.com/AugustZTR/picbed/master/img/load.png)


## Validation

We use usb2micro usb cable to connect the development board and the upper computer to achieve communication. 

<img src="https://raw.githubusercontent.com/AugustZTR/picbed/master/img/image-20220827121203762.png" alt="image-20220827121203762" style="zoom:50%;" />

Afering connect the board to PC, run the `validation.py` , when seeing output like below, press the **reset button** shown in the picture, and the validation will start.

![iShot_2022-08-27_12.04.57](https://raw.githubusercontent.com/AugustZTR/picbed/master/img/iShot_2022-08-27_12.04.57.png)


#