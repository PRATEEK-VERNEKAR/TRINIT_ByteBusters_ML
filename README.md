# TRINIT_ByteBusters_ML 

## Video Link
https://youtu.be/rhhgOq2Lh8Q

https://drive.google.com/file/d/1TcGdGBmZaJvctPvitZOddEsPduOl6VCY/view?usp=drive_link

![Screenshot from 2024-03-10 08-31-37](https://github.com/PRATEEK-VERNEKAR/TRINIT_ByteBusters_ML/assets/107637873/407cb3ef-d19a-4a9e-8368-2f6929e28dd4)

# Project Repository

This repository contains codes and documentation for a project focused on the identification and classification of sexual harassment using machine learning techniques. It also includes a web application and a web scraping tool.

## Codes

### MachineLearning
- **Model1**
  - Contains code for training a DistilBERT model to identify sexual harassment in text.
  - Files:
    - Code for training the DistilBERT model.
    - Code for making predictions using the trained model.

- **Model2**
  - Contains code for classifying the type of sexual harassment.
  - Subfolders:
    - **Model2_part1**: Prediction of whether there is Commenting involved. (84% Accuracy)
    - **Model2_part2**: Prediction of whether there is Staring involved.  (82% Accuracy)
    - **Model2_part3**: Prediction of whether there is Touching involved.  (79.98% Accuracy)

### WebApp
- **FlaskServer**
  - Contains Flask application integrating all four machine learning models.
  
- **NodeServer**
  - Contains code for connecting to MongoDB database and handling API requests.
  - Files:
    - `app.js`: File handling API requests and sending responses.

- **React-Server**
  - Contains CSS and React components.
  - Subfolder:
    - **src**: React components.

### WebScrapping
- Contains Python code used to scrape tweets from Twitter.

## Documentation

### SHAP Results
- The SHAP(SHapley Additive exPlanations) for all 3 sub models of Model-2 with visual diagram

### LIME Results
- The LIME(Local interpretable model-agnostic explanations) for all 3 sub models of Model-2

### Humming Score
- The Humming Score of this model is aroun 85 - 90 %

### Architecture Design
- PDF document detailing the architecture of the project.

### Visualization
- PDF document containing visualizations related to the project.

### Scrapped_tweets
- Directory containing the scraped tweets.

### Data
- Data used to train Model2 as a whole.

### Output Results
- Results of the machine learning models predictions.

## How to Use
1. Clone the repository.
2. Navigate to the MachineLearning Folder to use the BlackBox of three ML models
3. Navigate to the WebApp Folder to use the MERN Stack Application
4. Navigate to the WebScrapping Folder to use the Twitter Tweet Scrapping Data


