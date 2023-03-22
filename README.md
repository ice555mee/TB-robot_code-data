# TB-robot_code-data
"TB robot"classifiaction model was built based on supervised machine-learning using ResNet50 architecture and trained with our datasets that composed of CV and GCD images. 



## Table of Contents
- [Branches](#branches)
- [Clone the repository](#Clone-the-repository)
- [Installation and running](#installation-and-running)
- [Files description](#files-description)
	- [Classification structure](#classificatio-structure)
    - [Part 1: CV Classification](#part-1-CV-Classification)
	- [Part 2: GCD Classification](#part-2-GCD-Classification)
	- [Part 3: Matching title_signal](#part-3-Matching-title_signal)
- [Interactive channel](#Interactive-channel)

## Branches

This project has two branches: `main`, `master`, which can be explained here

* `main` contains aggregate codes and datasets
* `master` additionally contains codes that user can use for test running. All models (including file.py, file.h5, file.json) are located in master branch of this repository. 


## Clone the repository
```bash
git clone   https://github.com/ice555mee/TB-robot_code-data.git
```

## Installation/running
You may need to install the libraries/packages according to Python requirments.txt file.

* Getting start on running prediction
  * Codes location 
  	- Codes for prediction test are located in folder named 'Test_prediction' in main branch, where all model files are located in master branch following to the structure below: 
	```
		Main branch
		└── Test_prediction
		  	├── CV_classification.py
		 	├── GCD_classification.py
		  	└── requirements.txt
		
		Master branch
		└──Test_prediction   	
	```


  * Installing/running
  	 * Install these requirements by this following command
  	 
  		```
		pip install -r requirements.txt
		```
	 * Run prediction 
		- Run files named 'CV_classification.py' or 'GCD_classification.py'
		- Before running, edit filepaths for the testing images (line 49) and the directory (line 51), where the user wants to put the results:
	 ![Picture1](https://user-images.githubusercontent.com/120438949/226878251-15ce403e-4435-452c-b46a-f17846f2f700.png)

			The result will show in the terminal with percentage confident of {type of material as Battery or Pseudocapacitor}.

## Files description

* Classification structure

![P5O135_datasets_mar2023](https://user-images.githubusercontent.com/120438949/226878156-5698bba7-d529-4823-a296-f0ae6583fe30.png)

	- (a) CV and GCD datasets obtained after classification by Process 1. Then, they were splitted  into training (manually labelled) and validation datasets for GCD and CV classification in Process 2 and Process 3, respectively. Here, the pobability of being Battery or Pseudocapacitor can be determined by percentage confidence
	- (b) The outputs from CV classification process 3 were manually specifically labelled to use in CV classification process 4 and 5 to obtain the 'Capacitive tendency' based on percentage confidence. 
	- (c) Table of processes, inputs and outputs.

* Part 1: CV Classification
	-	This folder named CV classification in main and master branches containes codes, datasets used in process 3, 4, 5.
	-	For the dataset folders are located in each folder according to each processes. 
	-	Folder structure:

		```
		Main branch
		└── CV classification/CV python
		  	├──CV classification process 3
			├──CV classification process 4
		  	└──CV classification process 5
		```

* Part 2: GCD Classification
	-	It includes the classification codes/models in Process 2 (in main and master branches for more model files).
	-	Folder structure:
	
		```
		Main branch
		└── GCD classification
		  	├──GCD Py
			├──Test_GCD_prediction
		  	└──Theoretical GCD classification
		```


* Part 3: Matching title_signal
	-	This folder provides codes for the matching (title vs. CV/GCD) experiment. In this part, only Process 2 (CV classification) and Process 3 (GCD classification) are involved.
	-	Folder structure:
	
		```
		Main branch
		└── Matching title_signal
		  	├──Matching result
			├──Python  	
		```


## Interactive channel
 Moreover, the interactive website was built based on our developed models: 
 http://supercapacitor-battery-artificialintelligence.vistec.ac.th/. 
 This channel is a quick tool to access the classification for any users without running script or coding required.




