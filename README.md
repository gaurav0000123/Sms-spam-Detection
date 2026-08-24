# SMS Spam Detection using NLP

A Machine Learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)**.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- NLP
- TF-IDF Vectorization
- Multinomial Naive Bayes

## Project Workflow
1. Load the SMS dataset
2. Clean and preprocess text
3. Split data into training and testing sets
4. Convert text into numerical features using TF-IDF
5. Train a Multinomial Naive Bayes model
6. Evaluate the model using accuracy and classification metrics
7. Predict a custom SMS message

## Dataset
Download the SMS Spam Collection dataset from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

After downloading, place either:
- `spam.csv` inside the `data` folder, or
- the original `SMSSpamCollection` file inside the `data` folder.

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python sms_spam_detection.py
```

## Author
Gaurav Singh
