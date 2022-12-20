# TB-robot_code-data
TB robot classifiaction model was built based on ResNet50 architecture and trained with our datasets to be used for CV and GCD classification.
This repository is composed of 3 main folders as described in Files introduction.


![Fig 3](https://user-images.githubusercontent.com/120438949/208368176-1fccebe0-250b-4e53-b4f6-4fc655d12c06.png)

Figure 3 | (a) CV and GCD datasets obtained after classification by Process 1, splitting them
into training and validation datasets for further GCD and CV classification in Process 2 and
Process 3, respectively. (b) The outputs from Process 3 are used in this final classification step
to obtain the pseudocapacitive tendency based on percentage confidence rating of the
prediction. (c) Table of processes, inputs and outputs performed/used to obtain these results.

## Files introduction

- CV Classification

-It includes the classification codes/models in Process 3, 4, and 5 as well as the training datasets.

-For the dataset folders, it was the label data into 4 catagoried (50% battery and 100% battery for Process 4) (50% pseudocapacitor and 100% pseudocapacitor for Process 5) according to the classification processes described in the publication (Figure 3).

- GCD Classification

It includes the classification codes/models in Process 2.

- Matching title_signal

This folder provides codes for the matching (title vs. CV/GCD) experiment.
In this part, only Process 2 (CV classification) and Process 3 (GCD classification) involve.


![titlle vs signal_white bg](https://user-images.githubusercontent.com/120438949/208585406-b72b6385-783f-4aea-a054-3a2ab0ff7135.png)
