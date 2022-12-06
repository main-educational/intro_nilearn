---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{code-cell} python3
# Let's keep our notebook clean, so it's a little more readable!
import warnings
warnings.filterwarnings('ignore')
```

# Predict age from resting state fMRI (rs-fMRI) with [`scikit-learn`](https://scikit-learn.org)

We will integrate what we've learned in the previous sections to extract data from *several* rs-fmri images, and use that data as features in a machine learning model.

The dataset consists of children (ages 3-13) and young adults (ages 18-39). We will use rs-fmri data to try to predict who are adults and who are children.

+++

## Load the data

```{code-cell} python3
:tags: [hide-output]
# change this to the location where you want the data to get downloaded
data_dir = './nilearn_data'

# Now fetch the data
from nilearn import datasets
development_dataset = datasets.fetch_development_fmri(
                                                      data_dir=data_dir,
                                                      reduce_confounds = False
                                                    )

data = development_dataset.func
confounds = development_dataset.confounds
```

How many individual subjects do we have?

```{code-cell} python3
len(data)
```

## Get Y (our target) and assess its distribution

```{code-cell} python3
# Let's load the phenotype data
import pandas as pd

pheno = pd.DataFrame(development_dataset.phenotypic)
pheno.head(40)
```

Looks like there is a column labeling children and adults. Let's capture it in a variable

```{code-cell} python3
y_ageclass = pheno['Child_Adult']
y_ageclass.head()
```

Let's have a look at the distribution of our target variable

```{code-cell} python3
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x=y_ageclass)
pheno.Child_Adult.value_counts()
```

This is very unbalanced -- there seems to be many more children than adults. It is something we can accomodate to a degree when training our model, but it is not within the scope of this tutorial. So let's select an arbitrary subset of the children to match the number of adults. As the 32 adults are at the beginning of the frame, this is easy to do:

```{code-cell} python3
data = data[0:66]
pheno = pheno.head(66)
y_ageclass = pheno['Child_Adult']
```

## Extract features

+++

Here, we are going to use the same techniques we learned in the previous tutorial to extract rs-fmri connectivity features from every subject. Let's reload our atlas, and re-initiate our masker and correlation_measure.

```{code-cell} python3
:tags: [hide-output]

from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure

# load atlas
multiscale = datasets.fetch_atlas_basc_multiscale_2015(data_dir=data_dir)
atlas_filename = multiscale.scale064

# initialize masker (change verbosity)
masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True,
                           memory='nilearn_cache', resampling_target="data",
                           detrend=True, verbose=0)

# initialize correlation measure, set to vectorize
correlation_measure = ConnectivityMeasure(kind='correlation', vectorize=True,
                                         discard_diagonal=True)
```

Okay -- now that we have that taken care of, let's load all of the data!

+++

**NOTE**: On a laptop, this might take a few minutes.

```{code-cell} python3
:tags: [hide-output]

all_features = [] # here is where we will put the data (a container)

for i,sub in enumerate(data):
    # extract the timeseries from the ROIs in the atlas
    time_series = masker.fit_transform(sub, confounds=confounds[i])
    # create a region x region correlation matrix
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]
    # add to our container
    all_features.append(correlation_matrix)
    # keep track of status
    print('finished %s of %s'%(i+1,len(data)))
```

```{code-cell} python3
# Let's save the data to disk
import numpy as np

np.savez_compressed('data/MAIN_BASC064_subsamp_features', a=all_features)
```

In case you do not want to run the full loop on your computer, you can load the output of the loop here!

```{code-cell} python3
feat_file = 'data/MAIN_BASC064_subsamp_features.npz'
X_features = np.load(feat_file)['a']
```

```{code-cell} python3
X_features.shape
```

Okay so we've got our features.

+++

We can visualize our feature matrix

```{code-cell} python3
import matplotlib.pyplot as plt

plt.imshow(X_features, aspect='auto', interpolation='nearest')
plt.colorbar()
plt.title('feature matrix')
plt.xlabel('features')
plt.ylabel('subjects')
```

## Prepare data for machine learning

Here, we will define a "training sample" where we can play around with our models. We will also set aside a "test" sample that we will not touch until the end.

+++

We want to be sure that our training and test sample are matched! We can do that with a "stratified split". Specifically, we will stratify by age class.

```{code-cell} python3
y_ageclass.shape
```

```{code-cell} python3
from sklearn.model_selection import train_test_split

# Split the sample to training/test and
# stratify by age class, and also shuffle the data.

X_train, X_test, y_train, y_test = train_test_split(
                                                    X_features, # x
                                                    y_ageclass, # y
                                                    test_size = 0.2, # 80%/20% split  
                                                    shuffle = True, # shuffle dataset
                                                                    # before splitting
                                                    stratify = y_ageclass, # keep
                                                                           # distribution
                                                                           # of ageclass
                                                                           # consistent
                                                                           # betw. train
                                                                           # & test sets.
                                                    random_state = 123 # same shuffle each
                                                                       # time
                                                                       )

# print the size of our training and test groups
print('training:', len(X_train),
     'testing:', len(X_test))
```

Let's visualize the distributions to be sure they are matched

```{code-cell} python3
fig,(ax1,ax2) = plt.subplots(2)
sns.countplot(x=y_train, ax=ax1, order=['child','adult'])
ax1.set_title('Train')
sns.countplot(x=y_test, ax=ax2, order=['child','adult'])
ax2.set_title('Test')
plt.tight_layout()
```

## Run your first model!

Machine learning can get pretty fancy very quickly. We'll start with a very standard classification model called a Support Vector Classifier (SVC).

While this may seem unambitious, simple models can be very robust. And we don't have enough data to create more complex models.

For more information, see this excellent resource:
https://hal.inria.fr/hal-01824205

+++

First, a quick review of SVM!
![](https://docs.opencv.org/2.4/_images/optimal-hyperplane.png)

+++

Let's fit our first model!

```{code-cell} python3
from sklearn.svm import SVC
l_svc = SVC(kernel='linear', class_weight='balanced') # define the model

l_svc.fit(X_train, y_train) # fit the model
```

Well... that was easy. Let's see how well the model learned the data!

We can judge our model on several criteria:
* Accuracy: The proportion of predictions that were correct overall
* Precision: Accuracy of cases predicted as positive
* Recall: Number of true positives correctly predicted to be positive
* f1 score: A balance between precision and recall

Or, for a more visual explanation...

![](https://upload.wikimedia.org/wikipedia/commons/2/26/Precisionrecall.svg)

```{code-cell} python3
from sklearn.metrics import classification_report, confusion_matrix, precision_score, f1_score

# predict the training data based on the model
y_pred = l_svc.predict(X_train)

# calculate the model accuracy
acc = l_svc.score(X_train, y_train)

# calculate the model precision, recall and f1, all in one convenient report!
cr = classification_report(y_true=y_train,
                      y_pred = y_pred)

# get a table to help us break down these scores
cm = confusion_matrix(y_true=y_train, y_pred = y_pred)
```

Let's view our results and plot them all at once!

```{code-cell} python3
import itertools
from pandas import DataFrame

# print results
print('accuracy:', acc)
print(cr)

# plot confusion matrix
cmdf = DataFrame(cm, index = ['Adult','Child'], columns = ['Adult','Child'])
sns.heatmap(cmdf, cmap = 'RdBu_r')
plt.xlabel('Predicted')
plt.ylabel('Observed')
# label cells in matrix
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j+0.5, i+0.5, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white")
```

![](https://sebastianraschka.com/images/faq/multiclass-metric/conf_mat.png)

+++

HOLY COW! Machine learning is amazing!!! Almost a perfect fit!

...which means there's something wrong. What's the problem here?

```{code-cell} python3
from sklearn.model_selection import cross_val_predict, cross_val_score

# predict
y_pred = cross_val_predict(l_svc, X_train, y_train,
                           groups=y_train, cv=3)
# scores
acc = cross_val_score(l_svc, X_train, y_train,
                     groups=y_train, cv=3)
```

We can look at the accuracy of the predictions for each fold of the cross-validation

```{code-cell} python3
for i in range(len(acc)):
    print('Fold %s -- Acc = %s'%(i, acc[i]))
```

We can also look at the overall accuracy of the model

```{code-cell} python3
from sklearn.metrics import accuracy_score
overall_acc = accuracy_score(y_pred = y_pred, y_true = y_train)
overall_cr = classification_report(y_pred = y_pred, y_true = y_train)
overall_cm = confusion_matrix(y_pred = y_pred, y_true = y_train)
print('Accuracy:',overall_acc)
print(overall_cr)
```

```{code-cell} python3
thresh = overall_cm.max() / 2
cmdf = DataFrame(overall_cm, index = ['Adult','Child'], columns = ['Adult','Child'])
sns.heatmap(cmdf, cmap='copper')
plt.xlabel('Predicted')
plt.ylabel('Observed')
for i, j in itertools.product(range(overall_cm.shape[0]), range(overall_cm.shape[1])):
        plt.text(j+0.5, i+0.5, format(overall_cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white")
```

The model seems to be performing very well. Let's run some null model:

```{code-cell} python3
from sklearn.model_selection import permutation_test_score
score, permutation_score, pvalue = permutation_test_score(
    l_svc, X_train, y_train, cv=3, scoring="accuracy",
    n_jobs=2, n_permutations=100)
print(f'accuracy {score}, average permutation accuracy {permutation_score.mean()}, p value {pvalue}')
```

so, as the classes are balanced, the chance level is close to 50%. The model performs significantly higher than chance.

+++

## Tweak your model

It's very important to learn when and where it's appropriate to "tweak" your model.

Since we have done all of the previous analysis with our training data, it's fine to try different models. But we **absolutely cannot** "test" it on our left-out-data. If we do, we are in great danger of overfitting.

We could try other models, or tweak hyperparameters, but we are probably not powered sufficiently to do so, and would once again risk overfitting.

+++

But as a demonstration, we could see the impact of "scaling" our data. Certain machine learning algorithms perform better when all the input data is transformed to a uniform range of values. This is often between 0 and 1, or mean centered around with unit variance. We can perhaps look at the performance of the model after scaling the data.

```{code-cell} python3
# Scale the training data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train_scl = scaler.transform(X_train)
```

```{code-cell} python3
plt.imshow(X_train, aspect='auto', interpolation='nearest')
plt.colorbar()
plt.title('Training Data')
plt.xlabel('features')
plt.ylabel('subjects')
```

```{code-cell} python3
plt.imshow(X_train_scl, aspect='auto', interpolation='nearest')
plt.colorbar()
plt.title('Scaled Training Data')
plt.xlabel('features')
plt.ylabel('subjects')
```

```{code-cell} python3
# repeat the steps above to re-fit the model
# and assess its performance

# don't forget to switch X_train to X_train_scl

# predict
y_pred = cross_val_predict(l_svc, X_train_scl, y_train,
                           groups=y_train, cv=3)

# get scores
overall_acc = accuracy_score(y_pred = y_pred, y_true = y_train)
overall_cr = classification_report(y_pred = y_pred, y_true = y_train)
overall_cm = confusion_matrix(y_pred = y_pred, y_true = y_train)
print('Accuracy:',overall_acc)
print(overall_cr)

# plot
thresh = overall_cm.max() / 2
cmdf = DataFrame(overall_cm, index = ['Adult','Child'], columns = ['Adult','Child'])
sns.heatmap(cmdf, cmap='copper')
plt.xlabel('Predicted')
plt.ylabel('Observed')
for i, j in itertools.product(range(overall_cm.shape[0]), range(overall_cm.shape[1])):
        plt.text(j+0.5, i+0.5, format(overall_cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white")
```

What do you think about the results of this model compared to the non-transformed model?

+++

**Exercise:** Try fitting a new SVC model and tweak one of the many parameters. Run cross-validation and see how well it goes. Make a new cell and type SVC? to see the possible hyperparameters

```{code-cell} python3
#l_svc = SVC(kernel='linear') # define the model
```

## Can our model classify children from adults in completely un-seen data?
Now that we've fit a model that we think has possibly learned how to decode childhood vs adulthood based on rs-fmri signal, let's put it to the test. We will train our model on all the training data, and try to predict the age of the subjects we left out at the beginning of this section.

+++

Because we performed a transformation on our training data, we will need to transform our testing data using the *same information!*

```{code-cell} python3
# Notice how we use the Scaler that was fit to X_train and apply it to X_test,
# rather than creating a new Scaler for X_test
X_test_scl = scaler.transform(X_test)
```

And now for the moment of truth!

No cross-validation needed here. We simply fit the model with the training data and use it to predict the testing data

I'm so nervous. Let's just do it all in one cell

```{code-cell} python3
l_svc.fit(X_train_scl, y_train) # fit to training data
y_pred = l_svc.predict(X_test_scl) # classify age class using testing data
acc = l_svc.score(X_test_scl, y_test) # get accuracy
cr = classification_report(y_pred=y_pred, y_true=y_test) # get prec., recall & f1
cm = confusion_matrix(y_pred=y_pred, y_true=y_test) # get confusion matrix

# print results
print('accuracy =', acc)
print(cr)

# plot results
thresh = cm.max() / 2
cmdf = DataFrame(cm, index = ['Adult','Child'], columns = ['Adult','Child'])
sns.heatmap(cmdf, cmap='RdBu_r')
plt.xlabel('Predicted')
plt.ylabel('Observed')
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j+0.5, i+0.5, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white")
```

The model generalized very well! We may have found something in this data which does seem to be systematically related to age ... but what?

+++

## Interpreting model feature importances
Interpreting the feature importances of a machine learning model is a real can of worms. This is an area of active research. Unfortunately, it's hard to trust the feature importance of some models.

You can find a whole tutorial on this subject here:
http://gael-varoquaux.info/interpreting_ml_tuto/index.html

For now, we'll just eschew better judgement and take a look at our feature importances.

+++

We can access the feature importances (weights) used by the model

```{code-cell} python3
l_svc.coef_
```

Let's plot these weights to see their distribution better

```{code-cell} python3
plt.bar(range(l_svc.coef_.shape[-1]),l_svc.coef_[0])
plt.title('feature importances')
plt.xlabel('feature')
plt.ylabel('weight')
```

Or perhaps it will be easier to visualize this information as a matrix similar to the one we started with

We can use the correlation measure from before to perform an inverse transform

```{code-cell} python3
correlation_measure.inverse_transform(l_svc.coef_).shape
```

```{code-cell} python3
from nilearn import plotting

feat_exp_matrix = correlation_measure.inverse_transform(l_svc.coef_)[0]

plotting.plot_matrix(feat_exp_matrix, figure=(10, 8),  
                     labels=range(feat_exp_matrix.shape[0]),
                     reorder=False,
                    tri='lower')
```

Let's see if we can throw those features onto an actual brain.

First, we'll need to gather the coordinates of each ROI of our atlas

```{code-cell} python3
coords = plotting.find_parcellation_cut_coords(atlas_filename)
```

And now we can use our feature matrix and the wonders of nilearn to create a connectome map where each node is an ROI, and each connection is weighted by the importance of the feature to the model

```{code-cell} python3
plotting.plot_connectome(feat_exp_matrix, coords, colorbar=True)
```

Whoa!! That's...a lot to process. Maybe let's threshold the edges so that only the most important connections are visualized

```{code-cell} python3
plotting.plot_connectome(feat_exp_matrix, coords, colorbar=True, edge_threshold=0.001)
```

That's definitely an improvement, but it's still a bit hard to see what's going on.
Nilearn has a new feature that lets us view this data interactively!

```{code-cell} python3
plotting.view_connectome(feat_exp_matrix, coords, edge_threshold='90%')
```

You can choose to open the figure in a browser with the following lines:

```{code-cell} python3
# view = plotting.view_connectome(feat_exp_matrix, coords, edge_threshold='90%')
# view.open_in_browser()
```
