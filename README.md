# class6_homework
HOMEWORK SESSION 6_MORE CHARTS AND DATA EXPLORATION
Class6-homework

This homework is a more extense testing of python library graphics (matplotlib and Seaborn), using the Diabetes dataset.  
In addition, it includes an initial analysis of the information, focused mainly on the variable that shows a more significant relationship to the target variable (Y=Diabetes Progression after 1 year), which is BMI (Body Mass Index).
According to the initial display of correlation coefficients and heatmap, the variables with the highest influence/significance vs. the target variable are BMI and S5, however when looking at the graphs (correlation pair plots).
          AGE           BMI            S5                 Y
AGE  1.000000     0.185085    0.270774     0.187889
BMI  0.185085     1.000000    0.446157      0.586450
S5    0.270774       0.446157   1.000000      0.565883
Y      0.187889       0.586450    0.565883     1.000000

I decided to validate if age or gender influence or are influenced by BMI and Y.  According to the data, age seems as an element that has a significant relationship for BMI and to a lesser extent for diabetes progression; its low correlation coefficients may indicate that the relationship is not linear.  

In terms of gender (SEX), while there are some trend differences in BMI and diabetes progression, it does not have a significant influence (i.e., it would not be a solid predictor of diabetes progression).

As for the Body Mass Index, according to its Regression and KDE plot parameters, the relationship is adjusted to a linear regression (this conclusion is initial, without running formal prediction models, only with basic statistics and charts).

LinregressResult(slope=10.233127870100775, intercept=-117.77336656656527, rvalue=0.5864501344746885, pvalue=3.4660064451673995e-42, stderr=0.6737955329480583)

This chart (my first suggestion to review the relationship) is in the folder diabetes.plots that generates the code, under the name of:
kdeplot_bmi_y.png

As additional information, also interesting, it can be observed the KDE (without regression line BMI-diabetes progression), where it is evidenced the distribution of samples and their concentration in the different ranges of the variables. 
KDE_BIVAR_bmi_y.png

To see the general two-way relationship of BMI and diabetes progression, go to the file:
pairgridscatter_bmi_y_.png

To see the differences found in BMI sorted by gender (SEX), you can check the file:  
pairPLOT_KDE.png

Looking a little more at the behavior of the BMI variable, we see it influences in a significant way other dataset measurements such as blood pressure, S5, age.  An overview of this effect can be found in the graph:
jointplotNEW.png
______________________________________
Dataset Dataset used in this homework is part of Scikit-learn datasets (toydatasets)
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html#sklearn.datasets.load_diabetes

Files included in this repository

Code:
HOMEWORK_SESSION6_REP.py

Dataset:
diabetes.csv

Graphics: Folder: diabetesplots Note: Some files in this folder have information not related/applicable to this dataset, since they are created only for testing purposes. In addition, they are provided only for information; the code includes the creation of the same directory during execution.

Install:
Supported Python3 version

Libraries imported for this exercise:
• os • numpy • pandas • matplotlib.pyplot • seaborn • time •from scipy.stats (linregress) •statsmodels.formula.api •from mpl_toolkits.mplot3d (Axes3D)

Code:
The code used in this homework is inside HOMEWORK_SESSION6_REP.py 

How to Run:
To run this homework it is recommended to use Anaconda, which provides support for executing .py files in a friendly-user enviro (Spyder-Jupyter Notebook). In order to run the code, you can:
1.	Download the diabetes.csv file included in the repository and send it to your Desktop, or
2.	You can move the diabetes.csv to another folder; if you do so, you need to edit the line 13 on the .py file, in order to appropriately locate your .csv file path.
diabetes_df=pd.read_csv(filepath_or_buffer='~/Desktop/diabetes.csv', sep=',', header=0)
After making sure you have this defined, you can run the file from your editor or terminal.
License
MIT License
