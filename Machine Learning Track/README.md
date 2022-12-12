# MLH GWH 2022 December: Machine Learning Track

This Track was performed on Google Colab.

Datasets Used: 
- [Sound Data (DAY 2)](https://github.com/wlifferth/ghw-2022-12)
- [Mini Speech Commands (DAY 3 & 4)](http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip)

Audio File (DAY 2): [congratulations.wav](https://github.com/wlifferth/ghw-2022-12/blob/main/congratulations.wav)

#

Dates Attended:
- Day 1: Dec 4   (Part-1)
- Day 2: Dec 6   (Part-2)
- Day 3: Dec 7   (Part-3)
- Day 4: Dec 8   (Part-3)

# 

#### Day 1: INTRODUCTION
````
We learnt the basics of Python, its different Data Types and String Operations.
````

#### Day 2: SOUND DATA
````
We learnt about the difference between Lists and Numpy Arrays. After this we learnt the importance of Shapes of Data that we are 
given during any Machine Learning modelling process. The use of Reshape() function was explained. In the end, we learnt about 
%timeit module which is used to time small bits of Python code snippets. In short we used this command to find on average how 
long a line-of code will run for.

After this, we used a sample dataset which was based on Sound data through a sample audio file called "congratulations.wav". We 
used two modules librosa and samplerate to understand this data more. Librosa module can be thought of being used for 
"Reading the Data" whereas Samplerate module can be thought of being used for "Writing the Data". We then used matplotlib.pyplot 
module to visually understand this data in the form of a Waveform plot. As we broke up the given data in the form of data values 
and sample rate, we were able to play around with these values and understand what differences in both these parameters will 
bring in the Audio output and in the plots that we make.

Now, we tried to visualize a method where instead of using a huge amount of integers(data) in the sample file if there was a way 
to use just the "minimal" amount of integers(data) for an n'th sample so that we get a similar or close-to output. This method is
termed as Normalization. We then tried to take a fourth of the data samples (1/4) and resampled this data to see what output we 
might get. As a result we saw how there is a significant change in the length or amount of integers(data) being stored now. With
the different types of resampling, we saw how the audio quality changes.

In the end of this session, we learnt about Fourier Transforms and the plot of them. We also tried playing around with how the 
changes in values either by multiplication or division will bring about a change in the Sin-Curves. Superposition of waves is a
concept which is important to understand here as in the sample file, there were multiple waves which were intersecting.
````

#### Day 3: MINI SPEECH COMMANDS
````
We started this session with the basic introduction to Machine Learning. We used tensorflow module to use the different Machine
Learning concepts which we learnt. We downloaded the Mini Speech Commands dataset which we studied and applied Machine Learning
algorithms. Divided this dataset into the training and testing(validation) parts. We can also define here how big we want the 
Output Sequence Length to be. This basically means how big of an output we want for those audio samples. Also understood the 
concept of Batch and Epoch. Batch is used for training the model by a set of "Question & Answer" and check according to the data. 
Based on the Answers that it gets, it will change the model accordingly to understand the data more and become more accurate. 
Epoch is used to control the number of passes or checks through the training dataset. This will use the Learning Algorithm and 
check it through the data based on the given amount of checks. After this we learnt that for an Tensor, there are different 
parameters for shape. In the audio data, as we do not require the Stereo Output we will "squeeze" this out. 

After this we have plotted the Fourier Transform for each individual Label and studied it. But understanding this plot is not as
easy as it seems to be. Hence we tried to plot this data using a Spectrogram. This is a visual representation of the spectrum of
frequencies of a signal as it varies with time. The "Loudness" (volume) is shown here in Yellow color and it will vary with each 
time frame. 

Now we have started the process to create the Machine Learning Model for this dataset using keras library in tensorflow. Understood
that Layers are important for any model. One such Layer is the Normalization Layer. This basically does a "scaling" of the Loudest 
and the Quietest values and makes them all on the same layer. This will make all of the Spectrograms of the same volume. We built 
a simple model for understanding how Layer concept works. We also understood the importance of the Dense() and Flatten() functions 
for interacting with the data. Now for how the model increases its accuracy by changing according to how we get the Answers from 
the Batch, we will be using the Compile() function.
````

#### Day 4: CONVOLUSIONS, USE OF A GPU FOR MORE BETTER COMPUTATIONS, AND PREDICTIONS & ANALYSIS
````
We understood how a GPU can be used with the models. We will fit this model using a given GPU (I used the default GPU available 
under Colab) and plotted it based on the loss and accuracy. Understood the basic concept of Convolusion and how it is basically a 
sliding-window which looks at the things which are next to one another. Convolution Layer is the next important concept we learnt 
under Layers.

We then created a new model which uses the Convolution concept (CNN) and did the appropriate steps of compile and fitting the 
data. With the use of this, we have understood that how the accuracy of the model increases when compared to the simple model. 
This also means that the model is getting better. The best part about this method (CNN) is that they are easy to parallelize. This 
is one important and crucial part of using such methods. One issue with this is that we notice the Dropout condition, which is 
similar to Overfitting. To overcome this, we will add this Dropout() function in the model creation step. This is kept regularly 
between the different layers, but is is usually not kept between the complex layers. In this case when we do the steps of plotting 
the curves after compiling and fitting, we will see that there is a slow steady curve for the accuracy unlike the previous cases. 
For this we will add a Normalization Layer and an additional Convolutional Window to the model. After seeing the plot curves for 
this updated model, we will understand that the accuracy has increased and it has become better. This means that our model is 
becoming better.

Now for the Predictions, we have used Seaborn module to create a Confusion Matrix. Before all of this, we will first use the 
predict() function for our model and pass the validation dataset to see the prediction probabilities for the labels. To make this 
numerical data more understandable, we have used the argmax() function. We similary found the True values for the respective 
labels, so that they can be compared to understand the accuracy and the drawbacks of our model through the Confusion Matrix. When 
we studied the studied the Confusion Matrix, we understood that there are a few places where labelling is wrongly done. To 
overcome this, we used the Class-Weights concept to identify which labels are imbalanced. We stored the class-weight in a 
dictionary for all the respective labels. Then we created this new model and while fitting the data, we included this class-weight 
so that we can accurately fit the respective data to those labels. Then we plotted the curves of this model and found out the true 
and predicted probabilities. Now we created the Confusion Matrix for these values.

From this understand the following points:
1. We were able to map the labels with more importance/priority in a better way. This means that it has improved from the previous 
   model.
2. There were few places where labelling was done wrong. This is a drawback of this approach but it can be improved further using 
   other methods.
````

#### Points to Note:
````
In Machine Learning, we look at the following points for any Model: 
1. Overall Model Performance: We need better performance for any model to get more better accurate results.
2. Real Life Situations: In this case, we may be not able to always rely on performance as we have to value human safety and other 
                         such parameters.
````
