# teamPoseidon
## Code base for the WRI Wave2Web Hackathon 

### Setup
1. install python requirements
2. add weather api key from worldweatheronline.com using `export WEATHER_API_KEY=mykey`

### To run the pipeline

1. `python fetch_live_data.py`
2. `python train.py`
3. `python predict.py `


### The repository consists of the following items:
   
1. A PBIX (PowerBi Report) named [*SWaRM_Dashboard_with_user_guide_Final.pbix*](https://app.powerbi.com/view?r=eyJrIjoiNTYyM2IzMjktYmJjMC00NmU1LWI0OWEtYTY0YTZmYTVlMmNjIiwidCI6IjYwOTU2ODg0LTEwYWQtNDBmYS04NjNkLTRmMzJjMWUzYTM3YSIsImMiOjF9) to launch the Dashboard which is shown in the demonstration video.
   
2. CSV file named [*updated_data_set_full.csv*](https://github.com/goelshivam1210/teamPoseidon/blob/main/data/updated_data_set_full.csv) which contains the final dataset used for prediction. These are raw values.

### Other Miscellaneous scripts

3. Script called [*prelim_analysis.m*](https://github.com/goelshivam1210/teamPoseidon/blob/main/prelim_analysis.m) used for running preliminary model training and analysis using MATLAB. This was done using simple regression and the results are not from time series dataset 

4. Jupyter Notebook called [*ReservoirEDA.ipynb*](https://github.com/goelshivam1210/teamPoseidon/blob/main/ReservoirEDA.ipynb) which contains the initial analysis of the given dataset.

5. Jupyter Notebook called [*dataset_play.ipynb*](https://github.com/goelshivam1210/teamPoseidon/blob/main/ReservoirEDA.ipynb) which contains the initial testing of various models for generating the time series dataset and running basic predictive modelling based on the internet search and literature survey.

6. Jupyter Notebook called [*Wave2Web.ipynb*](https://github.com/goelshivam1210/teamPoseidon/blob/main/wave2web.py) which contains the code used to train models used in the prototype using XGBoost.

7. Script called [*dbio.py*](https://github.com/goelshivam1210/teamPoseidon/blob/main/dbio.py) used for inserting and updating the mysql database on Linux VM on cloud.


## Dependencies to run the PowerBi Report

Install [R](https://cran.r-project.org/bin/windows/base/R-4.1.0-win.exe) 

Install *Corrplot* from packages.

In an R console

`install.packages("corrplot")`

Install Microsoft PowerBi


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

## Note

Please reach out to reachswarm@gmail.com for detailed description about the webapp and the cloud deployment of the SWaRM. We will be happy to discuss the details and provide docker support for the complete deployment of the pipeline. The code in this repository contains the necesarry scripts to train and evaulate the model. However, the complete software pipeline is implemented on a Linux VM hosted on a public cloud platform. 


