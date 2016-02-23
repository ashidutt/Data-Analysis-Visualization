__author__ = 'Ashish Dutt'
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import scipy.stats


with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import seaborn as sns
    sns.set_context("notebook", font_scale=1.5)
#warnings.simplefilter(action = "ignore", category = UserWarning)

#sns.set(color_codes=True)
# Reading the data where low_memory=False increases the program efficiency
data= pd.read_csv("gapminder.csv", low_memory=False)
# setting variables that you will be working with to numeric
data['breastcancerper100th']= data['breastcancerper100th'].convert_objects(convert_numeric=True)
data['femaleemployrate']= data['femaleemployrate'].convert_objects(convert_numeric=True)
data['alcconsumption']= data['alcconsumption'].convert_objects(convert_numeric=True)

#print "Showing missing data coulmn-wise"
#print data.isnull().sum()
# Create a copy of the original dataset as sub5 by using the copy() method
sub6=data.copy()
sub6_clean=sub6.dropna()

print("\nAssociation between alcohol consumption and breast cancer per 100th\n")
print(scipy.stats.pearsonr(sub6_clean['alcconsumption'], sub6_clean['breastcancerper100th']))

# using scatter plot the visulaize quantitative variable.
scat2= sns.regplot(x='alcconsumption', y='breastcancerper100th', data=sub6_clean)
plt.xlabel('Alcohol consumption in liters')
plt.ylabel('Breast cancer per 100th person')
plt.title('Scatterplot for the Association between Alcohol Consumption and Breast Cancer 100th person')
