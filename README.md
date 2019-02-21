# Paul's log

This is a problem of image classification. 
A model is needed to classify images into 7 hard classes.

## Basic implementation idea
I wanted to use Keras for high-level prototyping. I used the Keras version bundled in TF 1.10 (see below for why).

This is what I had in mind when I started:
- I wanted to train at least a vanilla DCNN, a DenseNet and transfer learn on Inception-v4.
- I wanted to compare model performance of these three.
- I wanted to try to add auxiliary data to the classifier head from the metadata (age and sex) as well.

##Notes about the dataset:
- It was unbalanced.
- I wrote a dirty Dataset class that had options to balance the dataset via either up- or downsampling.
- I also wanted add an option to train with heterogeneous class weights.
- I removed rows with NaN's and "unknown" values from the metadata. After this, there were a few images that could not be found among the filenames, so those were removed as well. The final image count was around 9950.

## Balancing notes
My Dataset class could upsample under-represented or downsample over-represented classes.
This is of course hacky. I am aware that this approach is only half-valid if I add image augmentation.
So I did.

Overall, I wanted to balance by:
- First, downsample the most populous nevi class.
- Then, upsample slightly the remaining classes.
- Finally, use class weights on the resulting dataset.

This seemed like a graceful, little hacky, not-too-invasive way to go.

If balanced properly, in theory, the multiclass accuracy can be a good indicator of model performance. 
In practice, I wanted to play it safe and use precision-recall and the full confusion matrix in the final evaluation.

Class weighting can be used to attribute more importance to classes that are deemed more important, such as the 'mel' class. 
I assume this represents melanomas. They are aggressive and spread fast, so a false negative for 'mel' could be painful.

## Data generation notes
The first idea was to write a data generator class or method around the keras.preprocessing.image.ImageDataGenerator.flow_from_dataframe method that is compatible with keras.models.Model.fit_generator.
Why? Because the metadata csv seemed like an easy way to map image filenames with class labels and aux. data.
This was a bad idea, because:
- The augmenting data generator was slow single-threaded.
- Multi-threaded data generators duplicate data. This is even more severe on Colab, since there is more cross-CPU talk there than on workstations.
- This can be circumvented by inheriting from the keras.utils.Sequence class, but then it makes no sense to further cling to the flow_from_dataframe method.
- As it turned out, the flow_from_dataframe method is just low-quality code, illogical and buggy.

After this, I went with inheriting from keras.utils.Sequence. See the dataset.Dataset class for details.

On augmentation:
- Inspection of the images revealed that most of the moles were centered quite well.
- I feel that there must be some sort of standardization in the dermoscopic technique that ensures this typical image appearance in practice.
- Thus, it is unlikely that practical test images will be off-center, unless a half-witted dermatologist acquires them.
- I wanted to code translational invariance into the network topologies. See the next section for details.
- For all the above, I thought that it was unnecessary to prescribe translation as part of augmentation. Only random rotation was used, since that appeared as a legit way to generate more data in under-represented classes.
- Pixel intensities were fine, both in terms of color fidelity and saturation.
- The final test set should not be augmented.
- The validation set can be augmented, best with a controlled random seed so that the validation loss does not fluctuate.

## Network design notes
Looking at the images, it was obvious that there were useful features on multiple scales. Thus, we need an architecture that allows for effortless feature propagation from finer to coarser spatial scales.
Vanilla DCNN's are not very good at this, since they bottleneck information flow. Forms of residual learning are better at this. I personally like the DenseNet architecture - this is why I picked it for comparing against the vanilla DCNN.
The Inception-v3 architecture is another form that is also a top-performer. It is a bit too expensive to train on my laptop, so I planned to transfer-learn with it.

I generally prefer fully convolutional architectures. Thus I added global maxpooling heads to all models (networks/basic module).
This ensures some translation invariance, since the layers in the head lose spatial localization information.

## Training notes
I used the Adam optimizer for all training, as it is the "standard" algorithm suggested by the DL community (and the fastest converging default choice in my experience).
I used learning rate scheduling and early stopping and waited until the early stopping hit in all cases.
All training histories were recorded. The models that yielded the highest validation accuracy were kept and saved.

When going for transfer learning with Inception-V3, I noticed that the validation losses were off.
After some research, I found out that the Keras version that I used suffered from [this bug](http://blog.datumbox.com/the-batch-normalization-layer-of-keras-is-broken/).
This can be fixed, but not in the timeframe of this assignment. So I finally decided to skip Inception-V3 and transfer learn on VGG16 instead.
VGG16 does not have any batch normalization that are the source of this problem. A pity...

Overall, I think my results are OK. I am particulary sorry about the stupid batchnorm bug in Keras, since it would have been nice to replicate [these results](https://www.nature.com/articles/nature21056).
These guys used Inception-V3 (guess where I took the idea from...) pre-trained on ImageNet and achieved 74% mean cat. accuracy.
With that said, they trained on ca. 130,000 images, not 10,000, but also had more classes. And they are from Stanford, so their paper must be put in Nature.

## Results and possible improvements
Here are the training curves and confusion matrices for the models that I tested.

I achieved around 60% multi-class accuracy on the test set that had the same class balance as the training set.
This seemed low compared to e.g., [this guy's](https://towardsdatascience.com/classifying-skin-lesions-with-convolutional-neural-networks-fc1302c60d54) results on the same dataset with MobileNet.
He claims he got to 85% validation categorical accuracy. I did check his implementation and with my data generator, it got to around 60%.
Why the difference? I think the key is that he did not balance the validation set.
Looking at the confusion matrix that his model produced, the nevi and vascular lesions were classified with very good accuracy, but other lesions were not.
His test set was not balanced and most of the images were of nevi. With his unbalanced dataset, I got a similar, 84% accuracy.
With this in mind, I regard the 60% that I achieved with a balanced dataset comparable to the claimed 85%.
Either that, or the [bug](http://blog.datumbox.com/the-batch-normalization-layer-of-keras-is-broken/) again.
[This guy](https://towardsdatascience.com/building-a-skin-lesion-classification-web-app-16fd2c422b9d) also achieved similar performance, also with transfer-learnt MobileNet (they must be copying each other).

With that said, the same issue with easily classified nevi and vascular lesion images and a shaky melanoma class appeared in my results as well.
This is not desirable, given that it is the melanomas that should be classified the most accurately.
Further improvements in training might be (should be!) achieved with e.g., Focal Loss, other hard negative mining, or a custom asymmetric loss that e.g., penalizes false negative melanomas. I did not implement this due to lack of time.

Furthermore, due to the malenoma confusion problem, I think this could be a nice project for someone to take. 
Even [this paper](https://www.nature.com/articles/nature21056) reports the same issue with melanoma detection (look at the AUC figures in sensitivity-specificity, they are the lowest for melanomas!).

Trends were obvious when looking at different architectures: vanilla CNN's performed worst, followed by the DenseNet, then the VGG transfer-learnt, and finally, the VGG model with auxiliary inputs gave best results.

Here are the confusion matrices:

![confusion matrices](./results/cm.png?raw=true "confusion matrices")

Here are some ideas to further improve results:
- use Focal Loss;
- implement an asymmetric loss on melanoma images, especially when one is mistaken for a nevus;
- use few-shot learning tricks to single out potential melanoma images;
- this is a wild idea, but one could also try to pre-train autoencoders on each supervised class, then project those features onto a best-separating hyperspace using a top-performer DNN architecture;
- melanomas appear quite diverse in terms of appearance, maybe define multiple subclasses to reduce confusion;
- add more data, especially for under-represented classes;
- what I wanted to try but did not have time is to set up a Spatial Pyramid-style model that has skip connections from low-level layers directly to the classification head. I understand that DenseNet has the same idea, but those shared features are further projected by intermediate layers;
- use a committee of classifiers, all trained class-weighted on different classes?
- wait until they fix the batchnorm Keras bug and/or reimplement in base Tensorflow, transfer learn on Inception-V3 or another top-performer, then random search in parameter space to really milk the problem;
- or a combination or all of the above.

### Vanilla CNN
A Vanilla DCNN implementation can be found in the networks/basic module.
I trained it from scratch. I tested this with both the simple and hybrid FC head and auxiliary data.
The model contained approximately 6 million parameters.
 
## DenseNet
I did my own implementation of this. It was trained from scratch. Not tested with auxiliary inputs.
The model contained approximately 6 million parameters.

## VGG16 transfer-learnt
This was trained in two steps. First, I trained only the head with an aggressive learning rate and waited till convergence.
Then, a lower learning rate was set and the head along with block 5 was finetuned.
Approximately 7.5 million parameters were fine-tuned, including the classification head.

# Code notes
I wrote all code on a Windows laptop using Spyder for prototyping. I linted the code using PyCharm.

Models were trained on the same laptop on a single GPU. I did not use Colab, since it did not give me enough GPU memory. [Others](https://stackoverflow.com/questions/48494853/google-colaboratory-resourceexhaustederror-with-gpu) complained about this as well.

#### How to run
Don't :). If you really want to, start from script.py. A main module is missing, I can add this if you insist. Didn't want to spend time on it.

I ran it in an IPython kernel in Spyder. I have not tested it from the command line. There might be problems with imports.

#### How to get the models
I will upload the trained models [here](https://drive.google.com/open?id=1TpF_sMsmM8bwEl5uWnxcBsZtPFKJO82X).

####Some thoughts on the code:
- It is not meant to be perfect, but is still more than a single-file script.
- I followed Google's pyguide as much as I could.
- Not all modules are of the same code quality. Please take the dataset module as a benchmark of my style and just imagine that the rest is on the same level. This only pertains to style, not to class design and user-friendliness. The class is still bloated, I am aware of that. Given time, I would have encapsulated the optional inputs, aux_class_order, etc. into their own classes.
- I'll be honest, I knew of Python's type hinting, but I have not used it seriously before. I may not know the best practices yet.

#### Git
I have not committed into the git repo that you sent, only zipped it up and sent it. I can of course do it if you insist. Did not want to spend more time.

#Mishaps
A mistake I made was not giving a fixed random seed to utils.setup_generators so that the test/training/validation sets were mixed up after reloading a model. For this reason, I could not visualize the true predictions on unseen data. I could of course retrain all models, but I would not spend more time on that. The results I show here are therefore just for demonstration purposes. You'll see the code with which I made the visualizations.

I evaluated test accuracies right after I trained the models and still had the data generators with the right indices. The trends were the same as seen in the confusion matrices plot, with a negative bias. I.e., an accuracy of around 80% shown in the figure was about 65% tested only on unseen images.
# Time spent
I spent three full evenings on the assignment. Models were training simultaneously while I coded.
Frankly, most of the time was spent tracking down [this bug](http://blog.datumbox.com/the-batch-normalization-layer-of-keras-is-broken/).

I received the assignment on Friday, the 15th of February and I immediately proceeded to not doing anything about it due to sickness. After recovering over the weekend somewhat, I started. It is Thursday, the 20th of February on the day of submission.

# *original Readme*

# Lesion Diagnosis

Automated predictions of disease classification within dermoscopic images.

## Problem Statement

Build and evaluate a deep learning model that classifies a dermoscopic image as one of the following classes:

- [Melanoma](https://dermoscopedia.org/Melanoma)
- [Melanocytic nevus](https://dermoscopedia.org/Benign_Melanocytic_lesions)
- [Basal cell carcinoma](https://dermoscopedia.org/Basal_cell_carcinoma)
- [Actinic keratosis / Bowen’s disease (intraepithelial carcinoma)](https://dermoscopedia.org/Actinic_keratosis_/_Bowen%27s_disease_/_keratoacanthoma_/_squamous_cell_carcinoma)
- [Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)](https://dermoscopedia.org/Solar_lentigines_/_seborrheic_keratoses_/_lichen_planus-like_keratosis)
- [Dermatofibroma](https://dermoscopedia.org/Dermatofibromas)
- [Vascular lesion](https://dermoscopedia.org/Vascular_lesions) 

## Data

As data, use [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T).

For your convenience we also provide the dataset for easy download via these three links:

* [HAM10000_images_part_1.zip](https://storage.googleapis.com/peltarion-ml-assignment/HAM10000/HAM10000_images_part_1.zip)
* [HAM10000_images_part_2.zip](https://storage.googleapis.com/peltarion-ml-assignment/HAM10000/HAM10000_images_part_2.zip)
* [HAM10000_metadata.csv](https://storage.googleapis.com/peltarion-ml-assignment/HAM10000/HAM10000_metadata.csv)

Example images for the different lesion categories:
![Example images of the different lesion categories.](lesions.png)

## Assignment Details

**The assignment shouldn't take more than one or two evenings to finish.**

Feel free to use the existing implementation in this repository as a base, but beware of low code quality and some data scientific problems in this implementation, that you will need to fix.

Here are some ideas on what could be interesting things to consider, you won't have time to go deep into all of them, so choose the areas that you find most interesting to implement and investigate. 

* Define and implement metrics suitable for this problem.
* Try different model architectures / hyper parameter settings and compare their performance.
* There are much more examples of some of the classes in the data set. How does 
that impact the way you approach this problem?
* There are not that many examples to learn from. What alternatives are there to 
train a good model despite not that much data?
* Improve the code quality, e.g. by following 
  [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)
* You're free to split the dataset however you choose but motivate your decisions. 
   
Don't forget to keep track of some of your mishaps as well as the successful 
experiments. The work test is not about building the perfect classifier - we 
are more interested in how you approach the problem. 

If you have access to a GPU you are free to use it; alternatively you can use [Google Colab's GPU runtime](https://colab.research.google.com/), which is currently free of charge.
Make sure to save locally all of the output you will need to prepare the report (in case of Colab, local storage is not persistent).

## Deliverables 

Git commit your results to this repo when you're done with the assignment.
Make sure to include descriptions on what you have done, including your modeling
choices, results, conclusions and visualizations.
Notebooks can be a good way to show and visualize your work and results, but
you are free to use alternative solutions. 

Zip up the repo and send us the file in an e-mail to ml-team@peltarion.com.

If your solution is good you will be invited to Peltarion’s office to present your work for the ML team and have a follow-up discussion.

## Questions

If you run into problems or have questions, don't hesitate to email ml-team@peltarion.com. Asking questions is a good thing.

## Appendix - If you want to use Colab for model training
After navigating to [Colab](https://colab.research.google.com/), start a new Python 3 notebook. In the Runtime menu, select Change runtime type and choose GPU as Hardware accelerator.

You can then run the following list of commands to download the data.

```bash
!wget https://storage.googleapis.com/peltarion-ml-assignment/HAM10000/HAM10000_images_part_1.zip
!wget https://storage.googleapis.com/peltarion-ml-assignment/HAM10000/HAM10000_images_part_2.zip
!wget https://storage.googleapis.com/peltarion-ml-assignment/HAM10000/HAM10000_metadata.csv
!unzip -qq HAM10000_images_part_1.zip -d data
!unzip -qq HAM10000_images_part_2.zip -d data
!mv HAM10000_metadata.csv data/

```

Once you have uploaded the code in the zip file to your Colab notebook environment you can use

```bash
!python main.py
```

to run the training loop.
