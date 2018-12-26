# Optical Character Recogintion

## Optical Character Recognition (OCR)
To show the versatility of HMMs, let’s try applying them to another problem; if you’re careful and
you plan ahead, you can probably re-use much of your code from Part 1 to solve this problem. Our
goal is to recognize text in an image – e.g., to recognize that Figure 2 says “It is so ordered.” We’ll
consider a simplified OCR problem in which the font and font size is known ahead of time, but the
basic technique we’ll use is very similar to that used by commercial OCR systems.
Modern OCR is very good at recognizing documents, but rather poor when recognizing isolated
characters. It turns out that the main reason for OCR’s success is that there’s a strong language
model: the algorithm can resolve ambiguities in recognition by using statistical constraints of
English (or whichever language is being processed). These constraints can be incorporated very
naturally using an HMM.
Let’s say we’ve already divided a text string image up into little subimages corresponding to individual letters; a real OCR system has to do this letter segmentation automatically, but here
we’ll assume a fixed-width font so that we know exactly where each letter begins and ends ahead
of time. In particular, we’ll assume each letter fits in a box that’s 16 pixels wide and 25 pixels
tall. We’ll also assume that our documents only have the 26 uppercase latin characters, the 26
lowercase characters, the 10 digits, spaces, and 7 punctuation symbols, (),.-!?’". Suppose we’re
trying to recognize a text string with n characters, so we have n observed variables (the subimage
corresponding to each letter) O1, ..., On and n hidden variables, l1..., ln, which are the letters we
want to recognize. We’re thus interested in P(l1, ..., ln|O1, ..., On). As in part 1, we can rewrite
this using Bayes’ Law, estimate P(Oi|li) and P(li|li−1) from training data, then use probabilistic
inference to estimate the posterior, in order to recognize letters.
What to do. Write a program called ocr.py that is called like this:
./ocr.py train-image-file.png train-text.txt test-image-file.png
The program should load in the train-image-file, which contains images of letters to use for training
(we’ve supplied one for you). It should also load in the train-text file which is simply some text
document that is representative of the language (English, in this case) that will be recognized. (The
training file from Part 1 could be a good choice). Then, it should use the classifier it has learned to
detect the text in test-image-file.png and output the recognized text on the last line of its output.
[djcran@raichu djc-sol]$ ./ocr.py train-image-file.png train-text.txt test-image-file.png
Simple: 1t 1s so orcerec.
Viterbi: It is so ordered.
Final answer:
It is so ordered.
Make sure to include a detailed report at the top of your code that explains how your approach
works, any design decisions you made, other designs you tried, etc.
Hints. We’ve supplied you with skeleton code that takes care of all the image I/O for you, so you
don’t have to worry about any image processing routines. The skeleton code converts the images
into simple Python list-of-lists data structures that represents the characters as a 2-d grid of black
and white dots. You’ll need to define an HMM and estimate its parameters from training data.
The transition and initial state probabilities should be easy to estimate from the text training data.
For the emission probability, we suggest using a simple naive Bayes classifier. The train-imagefile.png file contains a perfect (noise-free) version of each letter. The text strings your program
will encounter will have nearly these same letters, but they may be corrupted with noise. If we
assume that m% of the pixels are noisy, then a naive Bayes classifier could assume that each pixel
of a given noisy image of a letter will match the corresponding pixel in the reference letter with
probability (100 − m)%. This problem is left purposely open-ended; feel free to try other emission
probabilities, Bayes Net structures, and inference algorithms to get the best results
