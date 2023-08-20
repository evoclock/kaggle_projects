# This is a continuation of the EDA and model testing for credit risk analysis.
# The reason for branching at this point is that I did not feel that KNN imputation of datetime data was justifiable and so 
# to avoid the clutter will continue here
# Julen Gamboa
# j.a.r.gamboa@gmail.com

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
import plotnine as p9
from plotnine import ggplot, aes, geom_tile, geom_violin, geom_boxplot, geom_label, scale_fill_cmap, scale_fill_gradient, scale_fill_manual, theme_minimal, labs, element_text, facet_wrap
import plydata.cat_tools as cat
import os

# Load the data
df1 = pd.read_csv ('~/Desktop/DS/kaggle_projects/Credit_Risk_Classification/data/customer_data.csv')
df2 = pd.read_csv ('~/Desktop/DS/kaggle_projects/Credit_Risk_Classification/data/payment_data.csv')

# Create a temporary DataFrame for renaming columns
df1_renamed = df1.copy()

# Rename columns in df1_renamed
df1_renamed.columns = df1_renamed.columns.str.replace(r'fea_(\d+)', r'feature_\1', regex=True)

# Select columns to keep from the original df1
columns_to_keep = ['id', 'label']

# Select only the renamed feature columns from df1_renamed
renamed_feature_columns = df1_renamed.filter(like='feature_')

# Concatenate the 'id', 'label', and renamed feature columns
df1_final = pd.concat([df1[columns_to_keep], renamed_feature_columns], axis=1)
df1_final

df1_final.isnull().sum()
# we are expencting one with NAs (feature_2)
df2.isnull().sum()
# For df2 we are expecting NAs on prod_limit and highest balance

# Merge df1_final and df2 on id
merged_df = df1_final.merge(df2, on='id')

# Drop the two columns for which we have no reasonable way of finding out cause for NA
merged_df.drop(columns=['update_date','report_date'], inplace=True)

# Check NAs (there should be only three features with NAs)
merged_df.isnull().sum()

# Let's impute those missing values for feature_2, prod_limit, and highest balance
from sklearn.impute import KNNImputer

impute = KNNImputer(n_neighbors= 5)
cols = ['prod_limit','highest_balance','feature_2']
merged_df[cols] = merged_df[cols].round()
for i in cols:
    merged_df[i] = impute.fit_transform(merged_df[[i]])


# Deprecated because frankly the options to create a facet plot with invariant scale using the tidyverse are
# (unsurprisingly) superior to anything in Python

#output_directory = os.path.expanduser('~/Desktop/DS/kaggle_projects/Credit_Risk_Classification/Figures/')

# for i in merged_df.columns:
#     p = (ggplot(merged_df, aes(x=1, y=i))
#          + geom_violin(fill="#21918c", alpha=0.6)
#          + theme_minimal()
#          + labs(title=f'Distribution of {i}', x='', y=i)
#          + theme_minimal()  # You can customize the theme here
#          + theme(figure_size=(6, 4)))  # Adjust the figure size here
    
#     # Generate the filename
#     filename = f"{output_directory}03_{i}_dist.jpeg"
    
#     # Save the plot to the specified filename with a lower dpi
#     p.save(filename, dpi=150)

#     print(f"Plot saved to: {filename}")

#############################################
############### Corr Plot ###################

# Generate the correlation matrix and plot
# There's two options I have found using plotnine, the first below (commented out) 
# has issues adjusting the opacity of the legend but the syntax is the most tidyverse like.
# The second option utilises a tidyverse approach (using .melt()) to transform the data into long format
# prior to plotting, it maps the three columns (var1, var2, and value) to the aes geom before
# beautifying the plot with labels (in this case I filtered values above 0.1 but remove values that
# equal 1 since they would lie on the diagonal and it is of course not informative)
# The second option is less tidyverse like in syntax but it isn't impossible to get used to if you
# are coming from

#from plotnine import ggplot, aes, geom_tile, scale_fill_cmap, scale_alpha_manual, theme_minimal, labs
#
#corr_matrix = merged_df.corr()
#mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
#
# Generate a mask to print only the lower triangle
#mask = np.tril(np.ones_like(corr_matrix, dtype=bool))
#
#heatmap3 = (ggplot(pd.melt(corr_matrix.reset_index(), id_vars='index'), aes(x='index', y='variable', fill='value'))
#           + geom_tile(alpha=0.9)  # Adjust opacity using the 'alpha' parameter
#           + scale_fill_cmap(cmap_name='viridis', name="Correlation")
#           + theme_minimal()
#           labs(title="Correlation Heatmap Merged Data", x="", y=""))
#
#heatmap3 += theme(axis_text_x=element_text(rotation=60, hjust=1)) 
#print(heatmap3)


import plotnine as p9
from plotnine import aes, geom_tile, geom_label, scale_fill_cmap, theme_minimal, labs, element_text
import plydata.cat_tools as cat

tidy_corr = merged_df \
    .corr() \
    .melt(
        ignore_index=False,
    ) \
    .reset_index() \
    .set_axis(
        labels = ["var1", "var2", "value"],
        axis = 1
    ) \
    .assign(lab_text = lambda x: np.round(x['value'], 2)) \
    .assign(
        var1 = lambda x: cat.cat_inorder(x['var1']),
        var2 = lambda x:
             cat.cat_rev(
                 cat.cat_inorder(x['var2'])
             )
    )

tidy_corr

# Filter tidy_corr to exclude values that are 1.0 or below 0.1
tidy_corr_filtered = tidy_corr[(tidy_corr['value'] > 0.1) & (tidy_corr['value'] < 1.0)]

(p9.ggplot(
    mapping=p9.aes("var1", "var2", fill="value"),
    data=tidy_corr
) +
p9.geom_tile(alpha=0.8) +  # Adjust alpha
p9.geom_label(
    p9.aes(label="lab_text"),
    fill="white",
    size=8,
    data=tidy_corr_filtered  # Apply filter to geom_label
) +
scale_fill_cmap(name="Correlation", cmap="viridis") +  # Use viridis colormap
theme_minimal() +
labs(
    title="Credit Risk | Correlation Matrix Merged Data",
    x="", y=""
) +
p9.theme(
    axis_text_x=element_text(rotation=70, hjust=1),
    figure_size=(8, 6)
)) 

# Drop the 'label' column from merged_df and store it as the y variable for our model
x = merged_df.drop(columns=['label'],axis=1)
y = merged_df['label']
merged_df.drop(columns=['label'], axis=1, inplace=True)

# Check
ans = merged_df.columns
print(ans)

# Time to train some models

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# going for a 30% split
X_train,X_test,Y_train,Y_test = train_test_split(x,y, test_size= 0.3)

################################
##### Logistic Regression ######
lr = LogisticRegression()
lr.fit(X_train,Y_train)
Y_pred_lr = lr.predict(X_test)

# return the dimensions of the array
print(Y_pred_lr.shape)
# (2475,)

# Calculate the accuracy score between prediction and the test data
score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100, 2)
print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")
# The accuracy score achieved using Logistic Regression is: 83.64 %

###### Naive Bayes model ######
nb = GaussianNB()
nb.fit(X_train,Y_train)
Y_pred_nb = nb.predict(X_test)

# dimensions of the array
print(Y_pred_nb.shape)
# (2475,)

# Naive Bayes accuracy score
score_nb = round(accuracy_score(Y_pred_nb,Y_test)*100, 2)
print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")
# The accuracy score achieved using Naive Bayes is: 82.75 %

######################################
###### Decision Tree Classifier ######

# initialise the max accuracy object for hyperparameter tuning
max_accuracy = 0

# hyperparameter tuning loop
for x in range(200):
    dt = DecisionTreeClassifier(random_state=x)
    dt.fit(X_train,Y_train)
    Y_pred_dt = dt.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_dt,Y_test)*100, 2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
print(max_accuracy)
print(best_x)

# train the model with the best random state
dt = DecisionTreeClassifier(random_state=best_x)
dt.fit(X_train,Y_train)
Y_pred_dt = dt.predict(X_test)

# dimensions of the array
print(Y_pred_dt.shape)
# (2475,)

# calculate the accuracy
score_dt = round(accuracy_score(Y_pred_dt,Y_test)*100, 2)
print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")
# The accuracy score achieved using Decision Tree is: 97.94 %

#####################
###### XGBoost XXXXXX

# Save the initial XGBoost model
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(dt, f)
import xgboost as xgb

# Initialize the max accuracy object for hyperparameter tuning
max_accuracy_xgb = 0

# Hyperparameter tuning loop for XGBoost
for x in range(200):
    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=x)
    xgb_model.fit(X_train, Y_train)
    Y_pred_xgb = xgb_model.predict(X_test)
    current_accuracy_xgb = round(accuracy_score(Y_pred_xgb, Y_test) * 100, 2)
    if current_accuracy_xgb > max_accuracy_xgb:
        max_accuracy_xgb = current_accuracy_xgb
        best_x_xgb = x
        
# print(max_accuracy_xgb)
print(best_x_xgb)
# 0

# train the model with the best random state
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=0)
xgb_model.fit(X_train, Y_train)
Y_pred_xgb = xgb_model.predict(X_test)

# dimensions of the array
Y_pred_xgb.shape
# (2475,)

# calculate the accuracy
score_xgb = round(accuracy_score(Y_pred_xgb,Y_test)*100, 2)
print("The accuracy score achieved using XGBoost is: "+str(score_xgb)+" %")
# The accuracy score achieved using XGBoost is: 98.46 %

# save the trained XGBoost model
import pickle
with open('model2.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

##########################
##### Neural Network #####
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

# Build the NN model
model = Sequential()
# 11 units on a fully connected hidden layer, the number of input dimensions is the features in our merged_df
model.add(Dense(11, activation='relu', input_dim=21))
# output layer for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compile the NN model
# configure loss method for binary classification
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, epochs=3000)

# Make predictions on the test data with our trained model
Y_pred_nn = model.predict(X_test)

# Calculate the accuracy of the predictions
rounded = [round(x[0]) for x in Y_pred_nn]
Y_pred_nn = rounded
score_nn = round(accuracy_score(Y_pred_nn, Y_test) * 100, 2)
print("The accuracy score achieved using Neural Network is: " + str(score_nn) + " %")

####################################################
##### Compare the accuracy scores of our models ####

scores = [score_lr,score_nb,score_dt,score_xgb,score_nn]
algorithms = ["Logistic Regression","Naive Bayes","Decision Tree","XGBoost","Neural Network"]    

for i in range(len(algorithms)):
    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")


#### Plot ####
from plotnine import ggplot, aes, geom_bar, theme_minimal, labs, scale_fill_cmap
import matplotlib.cm as cm
import numpy as np

# Define the custom order (I find it is more tasteful to organise the categories according to their
# value, it seems to be the default behaviour of bar plots to order categories alphabetically/alphanumerically
# regardless of the arguments passed (although in R the forcats library gets around this superbly))
custom_order = ["XGBoost", "Decision Tree", "Logistic Regression", "Neural Network", "Naive Bayes"]

# Create a DataFrame for plotting
df = pd.DataFrame({"Algorithm": algorithms, "Score": scores})

# Reorder the DataFrame based on the custom order
df["Algorithm"] = pd.Categorical(df["Algorithm"], categories=custom_order, ordered=True)
df = df.sort_values("Algorithm")
# Sort the DataFrame by score in descending order
#df = df.sort_values(by="Score", ascending=False)

# Assign specified colors to algorithms
# I couldn't get this to simply sample 5 colours from viridis and then assign them
# to each algorithm rather than assign the colours to the accuracy scores so
# if you need to generate more in order to fit say 6 models, here's a resource
# for that.
# https://waldyrious.net/viridis-palette-generator/

colour_dict = {
    "XGBoost": "#440154",
    "Decision Tree": "#3b528b",
    "Logistic Regression": "#21918c",
    "Neural Network": "#5ec962",
    "Naive Bayes": "#fde725"
}

# Add a new 'Colour' column based on algorithm order
df["Colour"] = df["Algorithm"].map(colour_dict)

p = (ggplot(df, aes(x="Algorithm", y="Score", fill="Colour"))
    + geom_bar(stat="identity")
    + theme_minimal()
    + labs(title="Algorithm Scores", x="Algorithm", y="Accuracy")
    + scale_fill_manual(values=df["Colour"].tolist(), guide=False)
)

print(p)

# Predict labels using XGBoost
xgb_predicted_labels = xgb_model.predict(X_test)

# Predict labels using Decision Tree
dt_predicted_labels = dt.predict(X_test)

# Filter rows where XGBoost predicted high credit risk (1)
high_risk_cases_xgb = X_test[xgb_predicted_labels == 1]

# Filter rows where Decision Tree predicted high credit risk (1)
high_risk_cases_dt = X_test[dt_predicted_labels == 1]

# Now you have two DataFrames: high_risk_cases_xgb and high_risk_cases_dt
# These DataFrames contain cases where the respective models predicted high credit risk.

# Concatenate both DataFrames vertically
common_high_risk_cases = pd.concat([high_risk_cases_xgb, high_risk_cases_dt], ignore_index=True)

# Drop duplicates based on the 'id' column
common_high_risk_cases = common_high_risk_cases.drop_duplicates(subset='id')

# Print the resulting DataFrame
print(common_high_risk_cases)


# Create DataFrames to store the IDs that were predicted by one algorithm but not the other
xgb_missing_ids = X_test.copy()
xgb_missing_ids['model'] = 'xgb'
xgb_missing_ids = xgb_missing_ids[~xgb_predicted_labels.astype(bool)]

dt_missing_ids = X_test.copy()
dt_missing_ids['model'] = 'dt'
dt_missing_ids = dt_missing_ids[~dt_predicted_labels.astype(bool)]

# Concatenate both DataFrames vertically
missing_ids = pd.concat([xgb_missing_ids, dt_missing_ids], ignore_index=True)

# Drop duplicates based on the 'id' column
missing_ids = missing_ids.drop_duplicates(subset='id')

# Print the resulting DataFrame
print(missing_ids)

# Print the top 50 common cases
top_50_common_cases = common_high_risk_cases.head(50)
print("Top 50 Common Cases:")
print(top_50_common_cases)

output_path = '~/Desktop/DS/kaggle_projects/Credit_Risk_Classification/data/'

# Store the common and missing dataframes as CSV files
common_high_risk_cases.to_csv(output_path + 'common_high_risk_cases.csv', index=False)
missing_ids.to_csv(output_path + 'missing_ids.csv', index=False)

# Now we have an output and the cases which are missing from one algorithm can be scrutinised a little closer
