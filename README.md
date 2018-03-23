# Kaggle Planet Satelite Machine Learning Competition

##Repo Structure
- train.csv (the labels for the image data)
- sample\_submission\_v2.csv (an example of what a submission should look like)
- submission.csv (my actual submission)
- submission_additionalepochs.csv (my actual submission part2)
- script_RGBstats.py (my image classifier using RGB statistics and random forest)
- script\_RGBstats\_Edge.py (my image classifier using RGB statistics, edge statistics, and random forest)
- scriptTensorflow.py (my image classifier using tensorflow)
- Keras\_Submission\_Final.ipynb (my final image classifier used to get my highest score using a fine-tuned image classifier and RGB statistics with keras)

##Run your own copy
1. Download or clone repo
2. Download and install [Anaconda distribution](https://www.anaconda.com/download/#macos), because it has most everything you need pre-packaged
3. Open a terminal and run command ```jupyter notebook```
4. It will open a browser window with a file explorer, then open the desired ipynb file.
5. You may need to install additional python packages via pip install to run code