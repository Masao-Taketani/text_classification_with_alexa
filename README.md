# text_classification_with_alexa(Chatty Kanna👧🏻)
<p align="center">
  <img src="https://user-images.githubusercontent.com/37681936/48673899-b1c00b00-eb89-11e8-8861-69c980fcc70a.PNG">
</p>

## Overview
This Machine Learning module classifies categories of input text. By using an Amazon Alexa together, it can handle speech.
The diagram of the architecture is as follows.
![architecture](https://user-images.githubusercontent.com/37681936/48673918-fc418780-eb89-11e8-9263-6d41d0570655.PNG)
You can deploy this module in a server or a local PC. It takes JSON requests first, which includes text message, and then process the data to classify categories of the input text. After predicting the category of the input text, it responds with JSON, which contains the predicted category.

## How to train a ML model and deploy it
1. Test `tfidf.py`.
```console
$ python tfidf.py
```

2. Create pickle files for data and dictionary.
```console
$ python make_pkl_for_data_and_dic.py
```
Executing the above command, you will create two pickle files 'data.pkl' and 'dic.pkl' in data/pkl directory.

3. Train a ML model by using Multilayer Perceptron(MLP) algorithms.
```console
$ python train_mlp.py
```
You will save the weights(parameters) of the model 'genre-model.hdf5' in 'data/weights' directory after training.

4. Test the trained model.
```console
$ python prdict_using_mlp.py
```

5. Deploy the model as WebAPI.
```console
$ python webapi.py
```
