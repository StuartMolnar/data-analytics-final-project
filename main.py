import numpy as np
import pandas as pd
import math
from scipy import stats
# Import linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

QOL_DATA = pd.read_csv('Quality of life index by countries 2020.csv')
# had to clean some exceptions in country_info.csv to match the names from quality of life csv
COUNTRY_DATA = pd.read_csv('Country_Info.csv')
#PI_DATA = pd.read_csv('Properties price index by countries 2020.csv')
globalLifeQuality = QOL_DATA
countryData = COUNTRY_DATA
#globalPropertyPrices = PI_DATA

#--------------data cleaning---------------
#goal is to adjust values to be a proportion of the quality of life index

'''
#this section will add rental indexes to the dataframe
#prepare the relevant property index columns
globalPropertyPrices = globalPropertyPrices[['Country', 'Price To Rent Ratio City Centre', 'Price To Rent Ratio Outside Of City Centre']]

#join the dataframes on country column
globalLifeQuality = pd.merge(globalLifeQuality, globalPropertyPrices, how='inner', on=['Country'])
'''

#create a temporary frame to do calculations on
proportionOfImpactFrame = globalLifeQuality.drop(['Country'], axis=1)

#add a new column containing a value proportion modifier = (sum of all values in row / quality of life)
proportionOfImpactFrame['Proportion Modifier']  = proportionOfImpactFrame.apply(lambda row: abs(sum([col for col in row[1:]]))/row[0], axis=1)

#apply: (value / proportion modifier) aka (value / (sum of all values / quality of life))
def applyProportion(list):
    for i in range(len(list[:-1])):
        list[i] = math.trunc((list[i]/list[-1]) * 100) / 100
    return list

headers = proportionOfImpactFrame.columns
proportionOfImpactFrame = pd.DataFrame.from_records(proportionOfImpactFrame.apply(lambda row: applyProportion([col for col in row]), axis=1))
proportionOfImpactFrame.columns = headers

#remove unnecessary columns
proportionOfImpactFrame = proportionOfImpactFrame.drop(['Proportion Modifier'], axis=1)
proportionOfImpactFrame = proportionOfImpactFrame.drop(['Quality of Life Index'], axis=1)

#merge modified data back into main dataframe
proportionOfImpactFrame = pd.merge(globalLifeQuality[['Country', 'Quality of Life Index']], proportionOfImpactFrame, left_index=True, right_index=True)

#add an additional row for continents by referencing the country data spreadsheet
countryData = countryData[['name', 'region']].rename(columns={'name' : 'Country'})
proportionOfImpactFrame = pd.merge(proportionOfImpactFrame, countryData, how='inner', on=['Country'])


#rename the index column as Rank
proportionOfImpactFrame.index.name = 'Rank'

#output with values adjusted to proportion
#proportionOfImpactFrame.to_csv('adjusted.csv')

#--------------data cleaning----------------

#assign the indexes we want to minimize a negative value
globalLifeQuality['Cost of Living Index'] = globalLifeQuality['Cost of Living Index'].apply(lambda x: -x)
globalLifeQuality['Pollution Index'] = globalLifeQuality['Pollution Index'].apply(lambda x: -x)
globalLifeQuality['Traffic Commute Time Index'] = globalLifeQuality['Traffic Commute Time Index'].apply(lambda x: -x)
globalLifeQuality['Property Price to Income Ratio'] = globalLifeQuality['Property Price to Income Ratio'].apply(lambda x: -x)

#add an additional row for continents by referencing the country data spreadsheet
globalLifeQuality = pd.merge(globalLifeQuality, countryData, how='inner', on=['Country'])

#add an additional row to act as the label for logistic regression
#we are saying quality of life >= 150 is 1
globalLifeQuality['label'] = globalLifeQuality['Quality of Life Index'].apply(lambda x: 0 if x < 150 else 1)


#output without values adjusted to proportion
#globalLifeQuality.to_csv('unadjusted.csv')

#--------------calculations------------------

#is mean purchasing power the same in europe and asia
asiaMean = math.trunc(globalLifeQuality['Purchasing Power Index'].loc[globalLifeQuality['region'] == 'Asia'].mean() * 100) / 100

asiaSTD = math.trunc(globalLifeQuality['Purchasing Power Index'].loc[globalLifeQuality['region'] == 'Asia'].std() * 100) / 100
asiaSampleSize = len(globalLifeQuality.loc[globalLifeQuality['region'] == 'Asia'])

europeMean = math.trunc(globalLifeQuality['Purchasing Power Index'].loc[globalLifeQuality['region'] == 'Europe'].mean() * 100) / 100
europeSTD = math.trunc(globalLifeQuality['Purchasing Power Index'].loc[globalLifeQuality['region'] == 'Europe'].std() * 100) / 100
europeSampleSize = len(globalLifeQuality.loc[globalLifeQuality['region'] == 'Europe'])

results = pd.DataFrame({"Region": ["Europe", "Asia"],
                        "Sample Size": [europeSampleSize, asiaSampleSize],
                        "Mean": [europeMean, asiaMean],
                        "STD": [europeSTD, asiaSTD]
                        })



#---------------two-sample t-test--------------------

#our null hypothesis: Europe and Asia have the same average purchasing power
#h0: mu1 = mu2
#h1: mu1 != mu2

sp = math.sqrt((asiaSTD**2 / asiaSampleSize) + (europeSTD**2 / europeSampleSize))
t = (asiaMean - europeMean) / sp
degreesOfFreedom = asiaSampleSize + europeSampleSize - 2
pval = stats.t.sf(np.abs(t), degreesOfFreedom)*2

#pval is greater than 0.05 so we reject h0

#-----------------logistic regression-----------------
#predicting if quality of life > 150
logreg_df = globalLifeQuality.sample(frac = 1, random_state=1)

# split dataset in two parts: feature set and target label 
feature_set = logreg_df.columns[:-1]
feature_set = feature_set.drop(['Country', 'region', 'Quality of Life Index'])
features = logreg_df[feature_set]
target = logreg_df['label']

print(features)

# partition data into training and testing set 
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=1)

# instantiate the model
logreg = LogisticRegression(solver='lbfgs', max_iter=1000)

# fit the model with data
logreg.fit(feature_train,target_train)

# Forecast the target variable for given test dataset
predictions = logreg.predict(feature_test)

# Assess model performance using accuracy measure
cnf_matrix = metrics.confusion_matrix(target_test, predictions)
print (cnf_matrix)

print("Accuracy:",metrics.accuracy_score(target_test, predictions))

print ("F1: ", metrics.f1_score(target_test, predictions, average='binary'))

#------------knn-------------
#use knn to predict same value

# partition data into training and testing set 
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=1)


#qualityOfLifeScores = globalLifeQuality['Quality of Life Index']

# Create a KNN classifier object
model = KNeighborsClassifier(n_neighbors=4)

# Train the model using the training dataset
model.fit(feature_train,target_train)

# Predict the target variable for test dataset
predictions = model.predict(feature_test)
'''
# Calculate model accuracy
print("Accuracy:",accuracy_score(target_test, predictions))
# Calculate model precision
print("Precision:",precision_score(target_test, predictions))
# Calculate model recall
print("Recall:",recall_score(target_test, predictions))
# Calculate model f1 score
print("F1-Score:",f1_score(target_test, predictions))
'''
#-----------neural network--------------
#use a neural network to predict the value
'''
data cleaning goal:
add rental price indexes
'''
'''
#changes all location strings into country strings
globalCrimeIndex['City'] =  globalCrimeIndex["City"].apply(lambda x: x.split(',')[-1][1:])

#rename City column to Country
globalCrimeIndex.rename(columns={"City" : "Country"}, inplace=True)

#join all rows on same country, taking the average of their crime values
globalCrimeIndex = globalCrimeIndex.groupby('Country', as_index=False)['Crime Index'].mean()

#sort by crime index
globalCrimeIndex.sort_values(by='Country', ascending=False)

#truncate values to 2 decimal places
globalCrimeIndex['Crime Index'] = globalCrimeIndex['Crime Index'].apply(lambda x: math.trunc(x * 100) / 100)

#recreate the safety index column based on new values
globalCrimeIndex['Safety Index'] = globalCrimeIndex['Crime Index'].apply(lambda x: 100-x)

#rename the index column as Rank
globalCrimeIndex.index.name = 'Rank'

#add a new continent column using pycountry
def getContinentFromCountry(country):
    country_code = pc.country_name_to_country_alpha2(country, cn_name_format="default")
    continent_name = pc.country_alpha2_to_continent_code(country_code)
    return continent_name

globalCrimeIndex['Continent'] = globalCrimeIndex['Country'].apply(getContinentFromCountry)
'''
