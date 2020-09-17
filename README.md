# Simple OCR for Handwriting

This is a simple version of OCR (Optical Character Recognition) by using Tensorflow and OpenCV within Python3.

Firstly OCR (Optical Character Recognition) is usually made by combining several image processing techniques such as:<br>
&ensp; De-Skewing,<br>
&ensp; Cleaning,<br>
&ensp; Normalization,<br>
&ensp; Character isolation,<br>
&ensp; etc.<br>
Then it will goes into the pattern match or pattern recognition process to know the pattern of that character.<br><br>

Thus, this notebook will implement Tensorflow 2.0 as CNN within OpenCV2 to make a Simple OCR; whereas:<br>
&ensp; 1. Tensorflow for the core (pattern recognition) which trained using the [dataset from Sueiras](https://github.com/sueiras/handwritting_characters_database)[1]<br>
&ensp; 2. The model from tensorflow will be used to identify each character detected in the result of OpenCV process

Contact me at [LinkedIn](https://www.linkedin.com/in/amadeus-winoto/)
<br>
<hr>

<h3> References </h3>
[1] J. Sueiras, et al.: "Using Synthetic Character Database for Training Deep Learning Models Applied to Offline Handwritten Character Recognition", Proc. Intl. Conf.Intelligent Systems Design and Applications (ISDA), Springer, 2016.
