import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy.stats import linregress
import statsmodels.formula.api as smf
from mpl_toolkits.mplot3d import Axes3D


#creating Dataframe from Files
diabetes_df=pd.read_csv(filepath_or_buffer='~/Desktop/diabetes.csv',
                         sep=',',
                         header=0)
print(diabetes_df)
time.sleep(1)
print(diabetes_df.columns)
time.sleep(1)
print(diabetes_df.dtypes)
time.sleep(1)
print(diabetes_df.shape)
time.sleep(1)
print(diabetes_df.info())
time.sleep(1)
print(diabetes_df.describe())
time.sleep(1)
print(diabetes_df.corr())# -*- coding: utf-8 -*-
time.sleep(1)
#Correlation coefficient only for strong corr variables
columns = ['AGE', 'BMI', 'S5', 'Y']
subset = diabetes_df[columns]
print(subset.corr())
# Hypothetical 1
#See slope (bmi VS DIABETES PROGRESSION)
subset = diabetes_df.dropna(subset=['BMI', 'Y'])
xs = subset['BMI']
ys = subset['Y']
res = linregress(xs, ys)
print(res)
#See slope_now diabetes progression vs. BMI
subset = diabetes_df.dropna(subset=['BMI', 'Y'])
xs = subset['Y']
ys = subset['BMI']
res = linregress(xs, ys)
print(res)

#multiple regression
results = smf.ols('Y ~ BMI', data=diabetes_df).fit()
print(results.params)
#Adding age
results = smf.ols('Y ~ BMI + AGE', data=diabetes_df).fit()
print(results.params)
#Adding S5
results = smf.ols('Y ~ BMI + S5', data=diabetes_df).fit()
print(results.params)


os.makedirs('diabetesplots', exist_ok=True)
plt.matshow(diabetes_df.corr())
plt.xticks(range(len(diabetes_df.columns)), diabetes_df.columns)
plt.yticks(range(len(diabetes_df.columns)), diabetes_df.columns)
plt.colorbar()
plt.savefig(f'diabetesplots/heatmapplot.png', format='png')
plt.clf()
plt.close()

# Basic correlogram
sns.pairplot(diabetes_df)
plt.savefig(f'diabetesplots/pairplot.png', format='png')
plt.clf()
plt.close()

# Plotting line chart
plt.style.use("ggplot")
plt.plot(diabetes_df['BP'], color='blue', marker='o')
plt.title('BLOOD PRESSURE RANGE\nInitial info')
plt.xlabel('Samples')
plt.ylabel('Blood Pressure')
plt.savefig(f'diabetesplots/BP_to_see_plot.png', format='png')
plt.clf()
plt.close()

# Add jittering to age
bmi = diabetes_df['BMI'] + np.random.normal(0, 2.5, size=len(diabetes_df))
dprogression=diabetes_df['Y']
# Make a scatter plot
plt.plot(bmi, dprogression, 'o', markersize=5, alpha=0.2)
plt.xlabel('BODY MASS INDEX')
plt.ylabel('DIABETES PROGRESSION AFTER 1 YEAR')
plt.savefig(f'diabetesplots/other_regexample.png', format='png')
plt.clf()
plt.close()

# Pie
fig, axes = plt.subplots(1, 1, figsize=(8, 8))
axes.pie(diabetes_df['SEX'].value_counts(), labels=diabetes_df['SEX'].value_counts().index.tolist())
axes.set_title('GENDER')
axes.legend()
plt.savefig(f'diabetesplots/PIE_GENDER.png', format='png')
plt.clf()
plt.close()

# Plotting TWO lines chart
fig, ax= plt.subplots()
ax.plot(diabetes_df.index, diabetes_df["BMI"], color='blue')
ax.set_xlabel("Samples")
ax.set_ylabel("Body Mass Index", color='blue')
ax.tick_params('y', colors='blue')
ax2 = ax.twinx()
ax2.plot(diabetes_df.index, diabetes_df['Y'], color='green')
ax2.set_ylabel('Diabetes progression after 1 year', color='green')
ax2.tick_params('y', colors='green')
ax.set_title('Body Mass Index and Diabetes progression')
fig.set_size_inches([8,8])
plt.savefig(f'diabetesplots/BMI_vs_Y_plot.png', format='png')
plt.clf()
plt.close()

# Plotting histogram
plt.hist(diabetes_df['AGE'], bins=10, histtype='bar', rwidth=0.6, color='b')
plt.title('AGE RANGE')
plt.xlabel('AGE')
plt.ylabel('COUNT')
fig.set_size_inches([8,8])
plt.savefig(f'diabetesplots/AGE_hist.png', format='png')
plt.clf()
plt.close()

#Plotting scatter with DataCamp tehcnique
fig, ax= plt.subplots()
ax.scatter(diabetes_df["AGE"], diabetes_df["Y"], c=diabetes_df.index)
ax.set_xlabel("AGE OF PATIENT")
ax.set_ylabel("DIABETES PROGRESSION AFTER 1 YEAR")
ax.set_title('AGE AND DIABETES PROGRESSION -1 YEAR-')
fig.set_size_inches([8,8])
plt.savefig(f'diabetesplots/AGEDC_vs_Y_scatter.png', format='png')
plt.clf()
plt.close()

#Plotting scatter with DataCamp tehcnique
fig, ax= plt.subplots()
ax.scatter(diabetes_df["S2"], diabetes_df["S4"], c=diabetes_df.index)
ax.set_xlabel("ldl")
ax.set_ylabel("tch")
ax.set_title('LDL AND TCH CORRELATION')
fig.set_size_inches([8,8])
plt.savefig(f'diabetesplots/LDL_TCH CORR.png', format='png')
plt.clf()
plt.close()

# Plotting scatterplot
plt.scatter(diabetes_df['AGE'], diabetes_df['Y'], color='b', marker='x', s=10)
plt.title('AGE AND DIABETES PROGRESSION -1 YEAR-')
plt.xlabel('AGE')
plt.ylabel('DIABETES PROGRESSION')
fig.set_size_inches([8,8])
plt.savefig(f'diabetesplots/progression_AGE.png', format='png')
plt.clf()
plt.close()

# Plotting scatterplot
plt.scatter(diabetes_df['AGE'], diabetes_df['SEX'], color='b', marker='v', s=10)
plt.title('AGE AND GENRE CORRELATION')
plt.xlabel('AGE')
plt.ylabel('GENRE')
fig.set_size_inches([8,8])
plt.savefig(f'diabetesplots/CORR_AGE_GENRE.png', format='png')
plt.clf()
plt.close()

# Plotting scatterplot
plt.scatter(diabetes_df['SEX'], diabetes_df['Y'], color='g')
plt.title('GENRE AND DIABETES PROGRESSION -1 YEAR-')
plt.xlabel('SEX')
plt.ylabel('DIABETES PROGRESSION')
fig.set_size_inches([8,8])
plt.savefig(f'diabetesplots/progression_SEX.png', format='png')
plt.clf()
plt.close()

# Plotting scatterplot
plt.scatter(diabetes_df['BMI'], diabetes_df['Y'], color='g')
plt.title('BODY MASS INDEX AND DIABETES PROGRESSION -1 YEAR-')
plt.xlabel('BMI')
plt.ylabel('DIABETES PROGRESSION')
fig.set_size_inches([8,8])
plt.savefig(f'diabetesplots/progression_BMI.png', format='png')
plt.clf()
plt.close()

# Plotting scatterplot
plt.scatter(diabetes_df['BP'], diabetes_df['Y'], color='g')
plt.title('BLOOD PRESSURE AND DIABETES PROGRESSION -1 YEAR-')
plt.xlabel('BP')
plt.ylabel('DIABETES PROGRESSION')
fig.set_size_inches([8,8])
plt.savefig(f'diabetesplots/progression_BP.png', format='png')
plt.clf()
plt.close()

# Plotting scatterplot
plt.scatter(diabetes_df['S1'], diabetes_df['Y'], color='g')
plt.title('TC AND DIABETES PROGRESSION -1 YEAR-')
plt.xlabel('S1')
plt.ylabel('DIABETES PROGRESSION')
fig.set_size_inches([8,8])
plt.savefig(f'diabetesplots/progression_S1.png', format='png')
plt.clf()
plt.close()

# Plotting scatterplot
plt.scatter(diabetes_df['S2'], diabetes_df['Y'], color='g')
plt.title('LDL AND DIABETES PROGRESSION -1 YEAR-')
plt.xlabel('S2')
plt.ylabel('DIABETES PROGRESSION')
fig.set_size_inches([8,8])
plt.savefig(f'diabetesplots/progression_S2.png', format='png')
plt.clf()
plt.close()

# Plotting scatterplot
plt.scatter(diabetes_df['S3'], diabetes_df['Y'], color='g')
plt.title('HDL AND DIABETES PROGRESSION -1 YEAR-')
plt.xlabel('S3')
plt.ylabel('DIABETES PROGRESSION')
fig.set_size_inches([8,8])
plt.savefig(f'diabetesplots/progression_S3.png', format='png')
plt.clf()
plt.close()

# Plotting scatterplot
plt.scatter(diabetes_df['S4'], diabetes_df['Y'], color='g')
plt.title('TCH AND DIABETES PROGRESSION -1 YEAR-')
plt.xlabel('S4')
plt.ylabel('DIABETES PROGRESSION')
fig.set_size_inches([8,8])
plt.savefig(f'diabetesplots/progression_S4.png', format='png')
plt.clf()
plt.close()

# Plotting scatterplot
plt.scatter(diabetes_df['S5'], diabetes_df['Y'], color='g')
plt.title('LTG AND DIABETES PROGRESSION -1 YEAR-')
plt.xlabel('S5')
plt.ylabel('DIABETES PROGRESSION')
fig.set_size_inches([8,8])
plt.savefig(f'diabetesplots/progression_S5.png', format='png')
plt.clf()
plt.close()

# Plotting scatterplot
plt.scatter(diabetes_df['S6'], diabetes_df['Y'], color='g')
plt.title('GLUCOSE AND DIABETES PROGRESSION -1 YEAR-')
plt.xlabel('S6')
plt.ylabel('DIABETES PROGRESSION')
fig.set_size_inches([8,8])
plt.savefig(f'diabetesplots/progression_S6.png', format='png')
plt.clf()
plt.close()

#Distribution plot by AGE
sns.set()
sns.distplot(diabetes_df['AGE'], bins=10, kde=True)
plt.savefig('diabetesplots/distplotage.png', format='png')
plt.clf()
plt.close()

sns.set()
for jointplot_kind in ['reg', 'hex', 'kde', 'scatter']:
    sns.jointplot('BMI', 'Y', data=diabetes_df, kind=jointplot_kind)
plt.savefig(f'diabetesplots/jointplot.png', format='png')
plt.clf()
plt.close()
    
#multiple comparisons
plt.style.use("ggplot")

fig, axes = plt.subplots(1, 1, figsize=(5, 5))
axes.grid(axis='y', alpha=0.5)
axes.scatter(diabetes_df['Y'], diabetes_df['BMI'], color='blue')
axes.scatter(diabetes_df['Y'], diabetes_df['BP'], color= 'green')
axes.scatter(diabetes_df['Y'], diabetes_df['AGE'])
axes.set_title(f'Progression Diabetes Comparisons')
axes.legend()
plt.savefig(f'diabetesplots/comparisons_bmi_bp_age.png', format='png', dpi=300)
plt.clf()
plt.close()

# Create a distplot of the Age
fig, ax = plt.subplots()
sns.distplot(diabetes_df['AGE'], ax=ax,
             hist=True,
             rug=True,
             kde_kws={'shade':True})
ax.set(xlabel="Patients Age")
plt.savefig(f'diabetesplots/Age_hist_special.png', format='png')
plt.clf()
plt.close()

# Create a plot with 1 row and 2 columns that share the y axis label
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharey=True)
# Plot the distribution of 1 bedroom apartments on ax0
sns.distplot(diabetes_df['BMI'], ax=ax0)
ax0.set(xlabel="BMI")

# Plot the distribution of 2 bedroom apartments on ax1
sns.distplot(diabetes_df['AGE'], ax=ax1)
ax1.set(xlabel="AGE")
plt.savefig(f'diabetesplots/BMI_AGE_COMB.png', format='png')
plt.clf()

sns.boxplot(data=diabetes_df,
         x='SEX',
         y='Y')

plt.savefig(f'diabetesplots/boxplot_Y_SEX.png', format='png')
plt.clf()

# Create a pointplot and include the capsize in order to show bars on the confidence interval
sns.barplot(data=diabetes_df,
         y='Y',
         x='SEX',
         capsize=.1)
plt.savefig(f'diabetesplots/barplot_Y_SEX.png', format='png')
plt.clf()

# Create a scatter plot by disabling the regression line BMI
sns.regplot(data=diabetes_df,
            y='Y',
            x="BMI",
            fit_reg=False)
plt.savefig(f'diabetesplots/scatternogres_Y_BMI.png', format='png')
plt.clf()

#Reggresion BMI Diabetes Progression
sns.set_style('whitegrid')
sns.set(color_codes=True)
sns.regplot(data=diabetes_df, color='blue',
         x="BMI",
         y="Y")
sns.despine()
plt.savefig(f'diabetesplots/regression_BMI.png', format='png')
plt.clf()

# Create regression with bins BMI
sns.regplot(data=diabetes_df,
            y='Y',
            x="BMI",
            x_bins=10)
plt.savefig(f'diabetesplots/regBINS_BMI.png', format='png')
plt.clf()

#POLYNOMIAL Reggresion BMI Diabetes Progression
sns.regplot(data=diabetes_df, x='BMI', y='Y', order=2)
sns.despine()
plt.savefig(f'diabetesplots/polynregression_BMI.png', format='png')
plt.clf()

#POLYNOMIAL WITH BINS Reggresion BMI Diabetes Progression
sns.regplot(data=diabetes_df, x='BMI', y='Y', order=2, x_bins=10)
sns.despine()
plt.savefig(f'diabetesplots/polyBINSreg_BMI.png', format='png')
plt.clf()

#Residual plot POLYNOMIAL Reggresion BMI Diabetes Progression
sns.residplot(data=diabetes_df, x='BMI', y='Y', order=2)
plt.savefig(f'diabetesplots/residualpolynregression_BMI.png', format='png')
plt.clf()

#Reggresion AGE BINS Diabetes Progression
sns.set_style('whitegrid')
sns.set(color_codes=True)
sns.regplot(data=diabetes_df, x='AGE', y='Y', x_bins=10)
sns.despine()
plt.savefig(f'diabetesplots/regressionBINS_AGE.png', format='png')
plt.clf()

#Reggresion BMI Diabetes Progression lmplot
for p in ['bright', 'colorblind']:
    sns.set_palette(p)
sns.lmplot(data=diabetes_df,
         x="BMI",
         y="Y")
plt.savefig(f'diabetesplots/regressionlmplot_BMI.png', format='png')
plt.clf()

# Display the residual plot BMI regression
sns.residplot(data=diabetes_df,
          y='Y',
          x="BMI",
          color='g')
plt.savefig(f'diabetesplots/residualplot_BMI.png', format='png')
plt.clf()

#heatmap seaborn
sns.heatmap(diabetes_df.corr())
plt.savefig(f'diabetesplots/heatseaborn.png', format='png')
plt.clf()

# Create a PairGrid with a scatter plot for BMI and Diabetes progression
g = sns.PairGrid(diabetes_df, vars=["BMI", "Y"])
g2 = g.map(plt.scatter)
plt.savefig(f'diabetesplots/pairgrid_bmi_y_.png', format='png')
plt.clf()
plt.close()

# Create the same pairgrid but map a histogram on the diag
g = sns.PairGrid(diabetes_df, vars=["BMI", "Y"])
g2 = g.map_diag(plt.hist)
g3 = g2.map_offdiag(plt.scatter)
plt.savefig(f'diabetesplots/pairgridscatter_bmi_y_.png', format='png')
plt.clf()
plt.close()

# Create the same pairgrid but map a histogram on the diag
g = sns.PairGrid(diabetes_df, vars=["BMI", "Y", "AGE"])
g2 = g.map_diag(plt.hist)
g3 = g2.map_offdiag(plt.scatter)
plt.savefig(f'diabetesplots/pairgridscatter_bmi_y_AGE.png', format='png')
plt.clf()
plt.close()

# Plot the same data but use a different color palette and color code by Region
sns.pairplot(data=diabetes_df,vars=["BMI", "Y"],
        kind='scatter',
        hue='SEX',
        palette='RdBu',
        diag_kws={'alpha':.5})
plt.savefig(f'diabetesplots/pairPLOT_bmi_y_SEX.png', format='png')
plt.clf()

# Hexbin plot with bivariate distribution
sns.jointplot(x='BMI', y='Y', data=diabetes_df, kind='hex', height=7, color='g')
plt.savefig(f'diabetesplots/hex_bmi_y.png', format='png')
plt.clf()
plt.close()

# KDE plot with bivariate distribution
sns.jointplot(x='BMI', y='Y', data=diabetes_df, kind='kde', height=7, color='g')
plt.savefig(f'diabetesplots/KDE_BIVAR_bmi_y.png', format='png')
plt.clf()
plt.close()

# 3D for gender
gender1= diabetes_df[diabetes_df['SEX'] == 1]
gender2= diabetes_df[diabetes_df['SEX'] == 2]
fig = plt.figure()
axes = fig.add_subplot(1, 1, 1, projection='3d')
line1 = axes.scatter(gender1['BMI'], gender1['AGE'], gender1['Y'])
line2 = axes.scatter(gender2['BMI'], gender2['AGE'], gender2['Y'])
axes.legend((line1, line2), ('gender1', 'gender2'))
axes.set_xlabel('BMI')
axes.set_ylabel('AGE')
axes.set_zlabel('DIABETES PROGRESSION')
plt.savefig(f'diabetesplots/3D_BMI_AGE_Y.png', format='png')
plt.clf()
plt.close()


# Build a pairplot with different x and y variables
sns.pairplot(data=diabetes_df,
        x_vars=["BMI", "AGE"],
        y_vars=['Y', 'S6'],
        kind='scatter',
        hue='SEX',
        palette='husl')
plt.savefig(f'diabetesplots/pairPLOT_multiple.png', format='png')
plt.clf()
plt.close()

# Build a pairplot with different x and y variables
sns.pairplot(data=diabetes_df,
        x_vars=["BMI"],
        y_vars=['Y', 'AGE', 'SEX', 'S2', 'S4', 'S6'],
        kind='scatter',
        palette='husl')
plt.savefig(f'diabetesplots/pairPLOT_multiple_6.png', format='png')
plt.clf()
plt.close()

# plot relationships between BMI and diabetes progression
sns.pairplot(data=diabetes_df,
             x_vars=["BMI"],
             y_vars=["Y", "AGE"],
             kind='reg',
             palette='BrBG',
             diag_kind = 'kde',
             hue='SEX')
plt.savefig(f'diabetesplots/pairPLOT_KDE.png', format='png')
plt.clf()
plt.close()

# Build a JointGrid comparing BMI and diabetes progression
sns.set_style("whitegrid")
g = sns.JointGrid(x="BMI",
            y="Y",
            data=diabetes_df)
g.plot(sns.regplot, sns.distplot)

plt.savefig(f'diabetesplots/jointgridbasic.png', format='png')
plt.clf()
plt.close()

# Plot a jointplot showing the residuals
sns.jointplot(x="BMI",
        y="Y",
        kind='resid',
        data=diabetes_df,
        order=2)
plt.savefig(f'diabetesplots/residjointpoly.png', format='png')
plt.clf()
plt.close()

# Create a jointplot of BMI vs. diabetes progression
# Include a kdeplot over the scatter plot
sns.set_style('whitegrid')
g = (sns.jointplot(x="BMI",
             y="Y",
             kind='reg',
             data=diabetes_df,
             marginal_kws=dict(bins=10, rug=True))
    .plot_joint(sns.kdeplot))
fig.set_size_inches([8,8])
plt.savefig(f'diabetesplots/kdeplot_bmi_y.png', format='png')
plt.clf()
plt.close()

# Create a jointplot of GLUCOSE vs. diabetes progression
# Include a kdeplot over the scatter plot
g = (sns.jointplot(x="S6",
             y="Y",
             kind='reg',
             data=diabetes_df,
             marginal_kws=dict(bins=10, rug=True))
    .plot_joint(sns.kdeplot))
plt.savefig(f'diabetesplots/kdeplot_GLU_y.png', format='png')
plt.clf()
plt.close()

# Create a jointplot of S5 vs. diabetes progression
# Include a kdeplot over the scatter plot
g = (sns.jointplot(x="S5",
             y="Y",
             kind='reg',
             data=diabetes_df,
             marginal_kws=dict(bins=10, rug=True))
    .plot_joint(sns.kdeplot))
plt.savefig(f'diabetesplots/kdeplot_S5_y.png', format='png')
plt.clf()
plt.close()

# Create a jointplot of BMI VS.GLUCOSE
# Include a kdeplot over the scatter plot
g = (sns.jointplot(x="BMI",
             y="S6",
             kind='reg',
             data=diabetes_df,
             marginal_kws=dict(bins=10, rug=True))
    .plot_joint(sns.kdeplot))
plt.savefig(f'diabetesplots/kdeplot_BMI_GLU.png', format='png')
plt.clf()
plt.close()

# Create a jointplot of BMI VS.S5
# Include a kdeplot over the scatter plot
g = (sns.jointplot(x="BMI",
             y="S5",
             kind='reg',
             data=diabetes_df,
             marginal_kws=dict(bins=10, rug=True))
    .plot_joint(sns.kdeplot))
plt.savefig(f'diabetesplots/kdeplot_BMI_S5.png', format='png')
plt.clf()
plt.close()

# Create a regression plot using hue
sns.lmplot(data=diabetes_df,
           x="BMI",
           y="Y",
           hue="SEX")
plt.savefig(f'diabetesplots/regressionlmplot_BMI_sex.png', format='png')
plt.clf()

# Create a regression plot using hue
sns.lmplot(data=diabetes_df,
           x="AGE",
           y="Y",
           hue="SEX")
plt.savefig(f'diabetesplots/regressionlmplot_AGE_SEX.png', format='png')
plt.clf()

# Create a regression plot with multiple rows
sns.lmplot(data=diabetes_df,
           x="BMI",
           y="Y",
           row="SEX")
plt.savefig(f'diabetesplots/regressionMULT_SEX.png', format='png')
plt.clf()


# 2 in 1
fig, axes = plt.subplots(4, 1, figsize=(8,8))

# Plotting scatterplot
plt.style.use("ggplot")
axes[1].scatter(diabetes_df['BMI'], diabetes_df['BP'], color='b', marker='x', s=10)
axes[1].set_title('BMI AND AVERAGE BLOOD PRESSURE')
axes[1].set_xlabel('BMI')
axes[1].set_ylabel('BLOOD PRESSURE')
# Plotting scatterplot
plt.style.use("ggplot")
axes[0].scatter(diabetes_df['BMI'], diabetes_df['Y'], color='b', marker='x', s=10)
axes[0].set_title('BMI AND DIABETES PROGRESSION -1 YEAR-')
axes[0].set_xlabel('BMI')
axes[0].set_ylabel('DIABETES PROGRESSION')
# Plotting scatterplot
axes[2].scatter(diabetes_df['BMI'], diabetes_df['S5'], color='b', marker='x', s=10)
axes[2].set_title('BMI AND S5')
axes[2].set_xlabel('BMI')
axes[2].set_ylabel('S5')
# Plotting scatterplot
axes[3].scatter(diabetes_df['BMI'], diabetes_df['AGE'], color='b', marker='x', s=10)
axes[3].set_title('BMI AND AGE')
axes[3].set_xlabel('BMI')
axes[3].set_ylabel('AGE')
plt.tight_layout()
plt.savefig(f'diabetesplots/jointplotNEW.png', format='png')
plt.clf()
plt.close()    


