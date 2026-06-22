# HERE WE HAVE THE PREDICTIONS RESULTS!
# $ python task7/predict.py "data\val\Chili___leaf_curl\download (1).jpg"
# within dataset
Prediction
---------------------
Class      : Chili___leaf_curl
Confidence : 65.27%
(venv) 

# $ python task7/predict.py "data\val\Tomato___Bacterial_spot\00a7c269-3476-4d25-b744-44d6353cd921___GCREC_Bact.Sp 5807.JPG"

Prediction
---------------------
Class      : Tomato___Bacterial_spot
Confidence : 98.73%
(venv) 

# $ python task7/predict.py "LEAFCURL-CHILLI.jpg"
# Uploaded a image outside dataset (correct pred)
Prediction
---------------------
Class      : Chili___leaf_curl
Confidence : 91.13%
(venv) 

# $ python task7/predict.py "healthy-chilli.jpg"
# Uploaded a image outside dataset (correct pred)
Prediction
---------------------
Class      : Chili___healthy
Confidence : 52.94%
(venv) 

# $ python task7/predict.py "tom-healthy-3.jpg"
# image outside dataset (tomato-healthy) but predicting wrong
Prediction
---------------------
Class      : Chili___healthy
Confidence : 88.43%
(venv) 

# $ python task7/predict.py "tom-late-blight1.jpg"

Prediction
---------------------
Class      : Chili___healthy
Confidence : 62.12%
(venv) 

# so far chilly is only being correctly classified.
# need to make improvments!!