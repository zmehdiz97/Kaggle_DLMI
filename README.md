# Deep learning in medical imaging:
# Prostate cancer grade assessment challenge
[Report](https://github.com/zmehdiz97/Kaggle_DLMI/blob/main/DLMI_challenge_report.pdf)

> **Abstract:** *In recent years, deep learning technology has been used for analysing medical images in various fields.
The use of artificial intelligence (AI) in diagnostic medical imaging is undergoing extensive evaluation. AI has shown impressive accuracy and sensitivity in the identification of imaging abnormalities and promises to enhance tissue-based detection and characterisation.
Prostate cancer is one of the most common types of cancer for men. Usually prostate cancer grows slowly and is initially confined to the prostate gland, where it may not cause serious harm. The key to decreasing mortality is developing more precise diagnostics. For this reason, several researches have focused on the use of deep learning to detect prostate cancer. Diagnosis of PCa is based on the grading of prostate tissue biopsies. These tissue samples are examined by a pathologist and scored according to the Gleason grading system.
In this challenge, we will develop models for detecting PCa on images of prostate tissue samples, and estimate severity of the disease using the most extensive multi-center dataset on Gleason grading yet available.The goal of this challenge is to predict the ISUP Grade using only Histopathology images. For that, we will need to deal with the process of Whole Slide Images as huge gigapixel images and deal with the limited number of patients provided in the train set.*
# Requirements
```
pip install -r requirements.txt
```
# Instructions 
1. First uses tiles.py to extract patches and save them before training a model
2. follow steps in train.ipynb or use main.py

# Members
Mehdi Zemni & Hamdi Bel Hadj Hassine
