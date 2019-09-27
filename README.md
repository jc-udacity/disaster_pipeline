# Disaster Response Pipeline Project
### Summary
This repository contains all required files to clean messages data, save in a database, trian a classifier and finally expose a web application that predict any new message inputed by a user, and classify against 36 categories.


### Instructions:
1. Clone the repository on your local machine using:
    git clone https://github.com/jc-udacity/disaster_pipeline.git
 
2. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/model.pkl`
        Please note that execution time is pretty long since there is a GridSearchCV step to find the best parameters set.
        In case you want to skip this step, you can use the provided model.pkl file.

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/
    You will see some graphs such as:
![Top 5 categories(top5.png)
