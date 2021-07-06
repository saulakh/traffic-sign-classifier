## Traffic Sign Recognition
________________________________________
The goals / steps of this project are the following:
*	Load the data set
*	Explore, summarize and visualize the data set
*	Design, train and test a model architecture
*	Use the model to make predictions on new images
*	Analyze the softmax probabilities of the new images
*	Summarize the results with a written report
________________________________________

### Data Set Summary & Exploration
##### Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.
I used the numpy library to calculate summary statistics of the traffic signs data set:
*	The size of training set is len(X_train) = 34799
*	The size of the validation set is len(X_valid) = 4410
*	The size of test set is len(X_test) = 12630
*	The shape of a traffic sign image is X_train[0].shape = (32,32,3)
*	The number of unique classes/labels in the data set is len(np.unique(y_train)) = 43

##### Include an exploratory visualization of the dataset.
Here is an exploratory visualization of the data set. It is a bar chart showing how many images are in each class of the training dataset, which are highly unbalanced. I was able to get a 95% validation accuracy in 46 epochs without this step, so I continued without data augmentation for now. I will try augmenting the images to balance the dataset later, and hopefully improve the accuracy.

![image](https://user-images.githubusercontent.com/74683142/122596454-e47b5a00-d037-11eb-88a2-4fa9759332cd.png)

 
### Design and Test a Model Architecture
##### Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 
* I started off with a baseline accuracy of 89% after using the LeNet architecture from the lectures. I tried normalizing the data using (pixel – 128)/128, but this initially reduced the accuracy to 74%. I changed the images to grayscale and began changing other sections of the pipeline to resolve this, mostly through trial and error. 

```
def normalize(image_data):
    
    image_data = np.sum(image_data/3, axis=3, keepdims=True)
    image_data = (image_data - 128)/128
    
    return image_data
```

Here are a few examples of images before preprocessing:

![image](https://user-images.githubusercontent.com/74683142/122596574-12f93500-d038-11eb-8cce-da9be717babd.png) ![image](https://user-images.githubusercontent.com/74683142/122596589-18567f80-d038-11eb-8372-8410def1523d.png) ![image](https://user-images.githubusercontent.com/74683142/122596606-1e4c6080-d038-11eb-9713-be397cc831bb.png)
   
Here are a few examples of images after preprocessing:

![image](https://user-images.githubusercontent.com/74683142/122596622-26a49b80-d038-11eb-9c72-35319d6e7a6e.png) ![image](https://user-images.githubusercontent.com/74683142/122596634-2c9a7c80-d038-11eb-8244-f6717c00d97d.png) ![image](https://user-images.githubusercontent.com/74683142/122596656-32905d80-d038-11eb-917a-bb30a3b1d7db.png)

   
##### Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.
My final model consisted of the following layers, modified from the LeNet architecture in the CNN lab:

| Layer       | Description |
| ----------- | ----------- |
| Input       | 32x32x1 Grayscale image |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x6 |
| RELU      |     |
| Max pooling   | 2x2 stride, outputs 14x14x6        |
| Convolution 5x5	| 1x1 stride, valid padding, outputs 10x10x16 |
| RELU | |	
| Max pooling | 2x2 stride, outputs 5x5x16 |
| Flatten	| Outputs 400 |
| Fully connected | Outputs 120 |
| RELU	| |
| Fully connected |	Outputs 43 |

##### Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
To train the model, I kept the Adam optimizer and 0.001 learning rate from the LeNet architecture. I changed the batch size from 128 to 64, changed the number of epochs to 100, and included an early stop if the validation accuracy reached 95%.

##### Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated.
My final model results were:
*	training set accuracy of 100%
*	validation set accuracy of 95%
*	test set accuracy of 92.8%

This was calculated in cell [8] of the Jupyter notebook:
```
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        training_accuracy = evaluate(X_train, y_train)
        testing_accuracy = evaluate(X_test, y_test)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print("Training Set Accuracy = {:.3f}".format(training_accuracy))
        print("Test Set Accuracy = {:.3f}".format(testing_accuracy))
        print()
        if validation_accuracy >= 0.950:
            break
        
    saver.save(sess, './lenet')
    print("Model saved")
```

##### If an iterative approach was chosen, what was the first architecture that was tried and why was it chosen?
*	I started with the LeNet architecture from the CNN lab, since the project instructions mentioned that would be a good place to start. Using this architecture, I started off with a validation accuracy of 89%.
##### What were some problems with the initial architecture?
*	There was overfitting with the training set, since the training accuracy was much higher than the validation accuracy. From what I understood, this meant I needed to remove one or more layers from the architecture. Another option could have been to add dropout layers in the architecture, reducing overfitting to the training set.
##### How was the architecture adjusted and why was it adjusted?
*	I changed one layer at a time, so I could better understand the impact of each and decide which layers to remove. Taking out a convolutional layer would reduce the overall accuracy, so I took out a fully connected layer. This had a bigger impact on improving the overall accuracy compared to including dropout layers.
##### Which parameters were tuned? How were they adjusted and why?
*	From the LeNet architecture, I adjusted the normalization and grayscale functions, batch size, number of epochs, and removed one fully connected layer. I adjusted the model architecture because of overfitting, and everything else was from trial and error.
##### What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
*	This project was my first time working with neural networks, so I am still trying to better understand what convolution layers do.  It seems like they blur the images, which could help it extract features of signs that stand out the most. From what I understood, I think a dropout layer could help with overfitting because it removes random neurons. This prevents the model from relying too heavily on current training data, allowing it to better read new data in the test set.

### Test a Model on New Images
##### Choose five German traffic signs found on the web and provide them in the report.
I included 10 total images, and here are the first five German traffic signs that I found on the web:

![image](https://user-images.githubusercontent.com/74683142/122599612-a6346980-d03c-11eb-94a7-846cd44fe41b.png) ![image](https://user-images.githubusercontent.com/74683142/122599697-c2d0a180-d03c-11eb-8d5b-418e221eeaaf.png) ![image](https://user-images.githubusercontent.com/74683142/122599713-c8c68280-d03c-11eb-8edc-117765e34a4c.png) ![image](https://user-images.githubusercontent.com/74683142/122599730-ce23cd00-d03c-11eb-945d-c1e4a1cdcc15.png) ![image](https://user-images.githubusercontent.com/74683142/122599755-d4b24480-d03c-11eb-8d82-9e554e7cb9bc.png)

##### Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.

Here are the results of the prediction:
| Image	| Prediction |
| ------|------------|
| Yield	| Yield |
| Priority Road	| Priority Road |
| Stop	| Stop |
| Straight or Left	| Straight or Left |
| No Entry |	No Entry |

The model was able to correctly guess all 5 of these traffic signs, and had a resulting accuracy of 100%. This compares favorably to the accuracy on the test set of 92.8%. I also included 5 more images in the set of web images, which are saved in the folder called new_images.

 
### Websites for Images Used:
[Image 1](https://image1.masterfile.com/getImage/NjAwLTAzMTUyODUzZW4uMDAwMDAwMDA=ABudB4/600-03152853en_Masterfile.jpg) [Image 2](https://image.shutterstock.com/image-photo/international-traffic-sign-priority-road-260nw-1805608567.jpg) [Image 3](https://miro.medium.com/max/400/1*nhvFD7uT718W59UlRYaIWQ.jpeg) [Image 4](https://image.freepik.com/free-photo/blue-three-separate-signs-blur-backgruond_79529-17.jpg) [Image 5](https://lh3.googleusercontent.com/proxy/Zk9LMXTysVbfgmDnABmxFZGlaUYMG8-rvb4W6UgigbzbW6fGVQZ6dir3r4HW9601m82gBINOvZopjqxSNptD6HmYvYgbKE0B5_T_OL4l)
[Image 6](https://cdn1.epicgames.com/ue/product/Screenshot/productimagesvol28-1920x1080-289ecc1e357b866c0d05a243edb59d12.png?resize=1&w=1600) [Image 7](https://st4.depositphotos.com/3441567/22167/i/450/depositphotos_221679306-stock-photo-road-sign-priority-road.jpg) [Image 9](https://s0.geograph.org.uk/geophotos/04/21/68/4216856_77ada79c.jpg) [Image 10](https://cdn1.epicgames.com/ue/product/Screenshot/productimagesvol28-1920x1080-289ecc1e357b866c0d05a243edb59d12.png?resize=1&w=1600)
