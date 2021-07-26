# teamPoseidon
## Code base for the WRI Wave2Web Hackathon 

### The repository consists of the following items:

1. Script called *weather.py* used for collecting and curating the dataset.
   
2. Script called *prelim_analysis.m* used for running preliminary model training and analysis using MATLAB. This was done using simple regression and the results are not from time series dataset 

3. Jupyter Notebook called *ReservoirEDA.ipynb* which contains the initial analysis of the given dataset.

4. Jupyter Notebook called *dataset_play.ipynb* which contains the initial testing of various models for generating the time series dataset and running basic predictive modelling based on the internet search and literature survey.

5. Jupyter Notebook called *Wave2Web.ipynb* which contains the code used to train models used in the prototype using XGBoost.

6. Script called *dbio.py* used for inserting and updating the mysql database on Linux VM on Azure cloud.

7. CSV file named *final_dataset_csv.csv* which contains the final dataset used for prediction. These are raw values.

8. A PBIX (PowerBi Report) named *reservoir.pbix* to launch the Dashboard which is shown in the demonstration video.

## Dependencies to run the PowerBi Report

Install (R) <https://cran.r-project.org/bin/windows/base/R-4.1.0-win.exe> 

Install *Corrplot* from packages.

In an R console

`install.packages("corrplot")`


## Dependencies to run the code


### Setup the virtual environment

Use the requirements file to install the dependencies 

`pip install -r requirements.txt`

[Optional] Create a conda environment

`conda create --name <YOUR_ENV_NAME> --python=3.7` <BR>
`conda activate <YOUR_ENV_NAME>` <BR>
`pip install -r requirements.txt` <BR>

Other Dependencies

MATLAB is required to run .m script.


