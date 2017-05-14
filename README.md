[//]: # (Image References)
[car_vs_nocar_1]: assets/car_vs_no_car.png
[car_vs_nocar_2]: assets/car_vs_no_car_v2.png
[hog]: assets/hog.png
[hog2]: assets/hog2.png
[hsv]: assets/HSV.png
[sliding_windows]: assets/sliding_windows.png
[sliding_windows2]: assets/sliding_windows2.png
[heatmap]: assets/heatmap.png
[heatmap1]: assets/heatmap1.png
[heatmap2]: assets/heatmap2.png
[heatmap3]: assets/heatmap3.png
[heatmap4]: assets/heatmap4.png

# Vehicle Detection

I used two different methods to detect vehicles

- **Traditional CV Method (traditional_approach.ipynb)**
    - By traditional, I mean extracting features from the image and put into a classifier to detect if it's an vehicle
    - Hand picked input features are required. That's why it's traditional
    - For example, a histogram of oriented gradients (HOG) is used to extract features
    - Then I used traditional sliding windows techniques to detect objects
    - Detailed information can be found below

- **Deep Learning Method (deep_learning_approach.ipynb)**
    - Thanks to [Vehicle Image Database](https://www.gti.ssr.upm.es/data/Vehicle_database.html), I was able to collect enough car and non car images to train a model
    - I used a pre-trained VGG Net and fine tuning on the above datasets
    - Then, I created a heat map from the output of the final convolution layers
    - After creating a heatmap, it was breeze to detect vehicles
    - Since the traditional method works way too slow in processing vidoes, I ended up using the deep learning method to create a final video


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Writeup / README

### 1. Provide a Writeup / README that includes all the rubric points
You're reading it!

### Histogram of Oriented Gradients (HOG)
#### 1. Explain how you extracted HOG features from the training images.

The code for this step is contained in the HOG section of [traditional_approach.ipynb](traditional_approach.ipynb#Histogram-of-Oriented-Gradients-21). 

Before explaning the HOG features, I need to mention that the datasets I used are from [Vehicle Image Database](https://www.gti.ssr.upm.es/data/Vehicle_database.html).

I used *8,792 car images* and *8,968 non car images* in total.
Some of these images are shown below:

![car_vs_nocar][car_vs_nocar_1]
![car_vs_nocar_2][car_vs_nocar_2]


After loading the above images, I used `skimage.features.hog` with parameters

- orientations: 9
- pixels_per_cell: (8, 8)
- cells_per_block: (2, 2)

After applying `hog`,

![hog][hog]
![hog2][hog2]


#### 2. Explain how you settled on your final choice of HOG parameters.

I settled on the above mentioned parameters by comparing the visualizations on the car image and the non-car image.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

There are 3 features concatenated in total. The codes for each step is located under the section called `Image features & Transformation`.

- HOG features
- Bin spatial features
- Histogram features

When all these 3 features are concatenates as a single array, its dimension becomes 8460.
So, the shape of the training data would be `(n_samples, 8460)`.

Then I used `Keras` to build a simple neural network to train the model

The architecture was shown below:

```text
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 8460)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 512)               4332032   
_________________________________________________________________
batch_normalization_1 (Batch (None, 512)               2048      
_________________________________________________________________
activation_1 (Activation)    (None, 512)               0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 512)               262656    
_________________________________________________________________
batch_normalization_2 (Batch (None, 512)               2048      
_________________________________________________________________
activation_2 (Activation)    (None, 512)               0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 513       
_________________________________________________________________
activation_3 (Activation)    (None, 1)                 0         
=================================================================
Total params: 4,599,297
Trainable params: 4,597,249
Non-trainable params: 2,048
_________________________________________________________________
```

Note how the final layer dimension is 1 because this is a binary classification problem (whether to detect there is a vehicle on the image)

Its final accuracy was 99.62% on test sets


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

First, I implemented sliding windows and visualize on the image to get a feeling

![sliding_windows][sliding_windows]

Notice that I don't have to slide windows in the upper section of the image because it's only sky.

I set a height range of the sliding windows from 300 to 720 (full height of the image)

Then for ever window, I test if there is a vehicle detected.

![sliding_windows2][sliding_windows2]

Notice there is also a false positive in the bottom left corner. The false positive boxes are filtered out by generating a heat map, which will be explained below.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

- **Traditional methods**
    - Convert the image to YCrCb
    - 3 Channel Hog features
    - Spatially binned features
    - Color histogram features

- **Deeplearning methods**
    - Uses RGB image as a raw
    - Uses a pre trained vgg net as initial weights
    - Fine tunes on the same dataset
    - After this step, it's the same as the traditional method. Instead of hand picking features, it uses the output of the final CNN layer



### Video Implementation

#### 1. Provide a link to your final video output.

Here's a [link to my video result (deep learning method)](./out_dl.mp4).  
The traditional method takes way too long to process it...


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In order to false positives, I created a heatmap and filter out the weak bounding box.

For example, the same image from the above will look like this:

![heatmap][heatmap]

Notice the color in the bottom left corner is darker which means its scale is different. So I can simply threshold values to filter out false positives.

After that, I used `scipy.ndimage.mearsuremetns.label()` to separate each heatmap sections.

Then my final images with the bounding boxes would look like this

![heatmap1][heatmap1]
![heatmap2][heatmap2]
![heatmap3][heatmap3]
![heatmap4][heatmap4]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?


- The biggest problem: it's so slow! For example, the traditional method would take 2 hours to process 50 seconds vidoes. I need more fancier algorithms such as `YOLO` or `Faster R-CNN` to detect vehicles in real time.
- The second problem: the size of a bounding box is quite different from the object size. It probably can be overcome by sliding windows on a smaller windows but it's also likely to create more false positive. The best method would be to create an END-to-END network which can predict bounding boxes without sliding windows.


