# Udacity_NLP_disaster_message_classifer

# 1.	Installations
Clone repository: 
git clone https://github.com/john77R/Udacity_NLP_disaster_message_classifer

# 2.	Current Requirements:
pandas, numpy, sklearn, sqlite3,nltk,pickle,sys,re, Flask,plotly...

# 3.	Project Motivation

  The aim of this project was to deploy a web app which analyzes disaster data from Figure Eight( now https://appen.com/) to build a model for an API that classifies disaster messages. The project includes a web app where in theory an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. The choice of model used in this case is a KNN (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) for multiclass classification. A number of models were trialed and KNN struck a nice balance of good F1 score and small model( Pkl file) size. Grid search CV again from Sklearn was used to find the optimal model.

  To convert our preclassified text data into a more model friendly format a typical NLP pipeline was built using Regular expressions, tokenisation, Lemmatization( Nltk module). Once our text is ready for ML training we build a pipline using;CountVectorizer,TfidfTransformer & a MultiOutputClassifier. We use CVgrid search with the search parameters specifed to train and test our model. See the notebook attached to step through the steps.
  
  The Web app is python using flask with visulaisations in plotly.
  
  Visuals are provided to show the Web App running.


# 4.	File Descriptions
The data source for this project is the disaster message repo from https://appen.com/ which is availble through Udacity.


# 5.	How to Interact with this project.
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and savespwd
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


Alternatively: Clone the repo and use it for your own NLP project. All the code presented here may be reused with minimal refactoring for other NLP or muliticalssification tasks. 

# 6. Files in the repository
Keep the file structure as follows:
app
| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app
data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py
|- InsertDatabaseName.db # database to save clean data to
models
|- train_classifier.py
|- classifier.pkl # saved model
README.md


# 7.	Acknowledgements
*Appen//figureEight
*Udacity.



MIT License
Copyright (c) 2021 John Ringrose
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
