__author__ = 'Ashish Dutt'
# Importing the required libraries
# Note %matplotlib inline works only for ipython notebook. It will not work for PyCharm. It is used to show the plot distributions
#%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi

sns.set(color_codes=True)
# Reading the data where low_memory=False increases the program efficiency
data= pd.read_csv("gapminder.csv", low_memory=False)

# setting variables that you will be working with to numeric
data['breastcancerper100th']= data['breastcancerper100th'].convert_objects(convert_numeric=True)
data['femaleemployrate']= data['femaleemployrate'].convert_objects(convert_numeric=True)
data['alcconsumption']= data['alcconsumption'].convert_objects(convert_numeric=True)

# Create a copy of the original dataset as sub5 by using the copy() method
sub10=data.copy()
# Since the data is all continuous variables therefore the use the mean() for missing value imputation
sub10.fillna(sub10['breastcancerper100th'].mean(), inplace=True)
sub10.fillna(sub10['femaleemployrate'].mean(), inplace=True)
sub10.fillna(sub10['alcconsumption'].mean(), inplace=True)

# categorize quantitative variable based on customized splits using the cut function
sub10['alco']=pd.qcut(sub10.alcconsumption,6,labels=["0","1-4","5-9","10-14","15-19","20-24"])
sub10['brst']=pd.qcut(sub10.breastcancerper100th,5,labels=["1-20","21-40","41-60","61-80","81-90"])
sub10['emply']=pd.qcut(sub10.femaleemployrate,4,labels=["30-39","40-59","60-79","80-90"])

# Showing the frequency distribution of the categorised quantitative variables
print "\n\nFrequency distribution of the categorized quantitative variables\n"
fd1=sub10['alco'].value_counts(sort=False,dropna=False)
fd2=sub10['brst'].value_counts(sort=False,dropna=False)
fd3=sub10['emply'].value_counts(sort=False,dropna=False)
print "Alcohol Consumption\n",fd1
print "\n------------------------\n"
print "Breast Cancer per 100th\n",fd2
print "\n------------------------\n"
print "Female Employee Rate\n",fd3
print "\n------------------------\n"

print "\nAssociation between Alcohol Consumption and Female Employ Rate"
model1=smf.ols(formula='alcconsumption~C(emply)',data=sub10)
results1=model1.fit()
print(results1.summary())
m1=sub10.groupby('alco').mean()
print "\n Mean Table for Alcohol consumption\n",m1

print "\nAssociation between Breast Cancer and Female Employ Rate"
model2=smf.ols(formula='breastcancerper100th~C(emply)',data=sub10)
results2=model2.fit()
print(results2.summary())
m2=sub10.groupby('emply').mean()
print "\n Mean Table for Female Employee Rate \n",m2

# contingency table of observed counts
ct1=pd.crosstab(sub10['alco'],sub10['emply'])
print "\nContingency table of observed counts\n"
print ct1

# column percentage
colsum=ct1.sum(axis=0)
colpct=ct1/colsum
print "\nColumn percentages\n",colpct

# Chi Square
print("\n Chi Square value, p value, expected counts\n")
cs1=ss.chi2_contingency(ct1)
print cs1

# Pearson Correlation

# Removing the missing values otherwise the pearson correlation will not work
sub10_clean=sub10.dropna()

print("\nAssociation between alcohol consumption and breast cancer per 100th\n")
print(ss.pearsonr(sub10_clean['alcconsumption'], sub10_clean['breastcancerper100th']))

print("\nAssociation between alcohol consumption and female employee\n")
print(ss.pearsonr(sub10_clean['alcconsumption'], sub10_clean['femaleemployrate']))

print("\nAssociation between breast cancer and female employee\n")
print(ss.pearsonr(sub10_clean['breastcancerper100th'], sub10_clean['femaleemployrate']))


'''
# using scatter plot the visulaize quantitative variable.
scat22= sns.regplot(x='alcconsumption', y='breastcancerper100th', data=sub10_clean)
plt.xlabel('Alcohol consumption in liters')
plt.ylabel('Breast cancer per 100th person')
plt.title('Scatterplot for the Association between Alcohol Consumption and Breast Cancer 100th person')
'''