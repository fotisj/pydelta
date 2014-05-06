Pydelta v 0.1

pydelta is a commandline tool which implements 3 algorithms in the form 
described by Argamon in a paper on John Burrows Delta.
(S. Argamon, "Interpreting Burrows’s Delta: Geometric and Probabilistic 
Foundations," Literary and linguistic computing, vol. 23, iss. 2, pp. 131-147, 2008.)

Delta is a measure to describe the stylistic difference between texts. It is used
in computational stylistics, especially in author attribution. 
This is implementation is for research purposes only, If you want to use 
a reliable implementation with a nice Gui and much more features you should 
have a closer look at the great R tool 'stylo': 
https://sites.google.com/site/computationalstylistics/

Usage of pydelta:
put your files into a subdirectory called 'corpus' under the directory
this script is living in. After the first run, you can use the file
pydelta.ini to set most of the important variables. You can use commandline
parameters to override default settings / settings in the pydelta.ini: 
pydelta.py -O figure.title:"German Novelists around 1800" -O stat.mfwords:2500
(setting a title and the amount of most frequent words used for delta). Use
pydelta -h   
for more information or look into the ini-file to find explanations of all
parameters.
The filenames for the corpus should have the format authorname_title.txt (the first part of the 
authorname is used to color the labels)

Thanks go to
Thorsten Vitt for his help with profiling some critical parts and general improvements
Allan Riddell for advise on matplotlib