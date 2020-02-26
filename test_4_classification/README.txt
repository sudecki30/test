
Hello! Here are the instructions for the case study:

This case study is to be done in autonomy. You have complete freedom on what method/technology to use.

The objective is to train a classifier that will distinguish pictures of original Cypheme certificates from pictures of fake certificates. Example pictures ("real.jpg" and "copy.png") are included.

For each picture, we collect 6 features:
 - blue, green and red components of the orange color (in the two rings around the certificate)
 - blue, green and red components of the white baground reference color of the certificate

The features are already extracted and normalised between 0 and 1.

Labeled training data is provided (label is 0 for a real certificate, 1 for a fake). The objective is to label the evaluation data.

Once you have labeled the data, please send an email to pierre@cypheme.com with object "Case study results" containing:

- a text file named "labels.txt" containing the labels of the evaluation data (0 or 1), in the same order as the given data, with nothing else.
- a text file named "explaination.txt" contaning a short description of the process you followed to label the data.

Good luck!