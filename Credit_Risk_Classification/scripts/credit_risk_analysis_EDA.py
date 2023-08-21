import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df1 = pd.read_csv ('~/Desktop/DS/kaggle_projects/Credit_Risk_Classification/data/customer_data.csv')
df2 = pd.read_csv ('~/Desktop/DS/kaggle_projects/Credit_Risk_Classification/data/payment_data.csv')

df1.info()

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

for i in df1.columns:
    values = df1[i].value_counts
    print(values)


# count the number of instances NAs are present per column
df1.isnull().sum()
# feature 2 has 149 such values

# Plot a correlogram to see which features might be more informative
df1_corr = df1_final.corr()

# first make the data into long (tidy) format
import plotnine as p9
from plotnine import ggplot, aes, geom_boxplot, theme_minimal, labs, element_text
import plydata.cat_tools as cat

tidy_corr = df1_final \
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

p9.ggplot(
    mapping = p9.aes("var1", "var2", fill = "value"),
    data=tidy_corr
) + \
    p9.geom_tile() + \
    p9.geom_label(
        p9.aes(label = "lab_text"),
        fill = "white",
        size = 8
    ) + \
    p9.scale_fill_distiller() + \
    p9.theme_minimal() + \
    p9.labs(
        title = "Credit Risk | Correlation Matrix",
        x = "", y = ""
    ) + \
    p9.theme(
        axis_text_x= p9.element_text(rotation=45, hjust = 1),
        figure_size=(8,6)
    )


# Seaborn being a native library is of course much less verbose but there are plots for which ggplot is still superior IMO.
# Compare the second set of plots for example
#plt.figure(figsize=(15,10))
#sns.heatmap(df1.corr(), annot=True)
#plt.show()

# for i in df1.columns:
#    print('columns: ', i)
#    sns.set_theme(style="ticks")
#    sns.boxplot(df1[i], saturation=.4)
#    sns.stripplot(data=df1[i], alpha=.4)
#    plt.show()

from plotnine import ggplot, aes, geom_violin, coord_cartesian, theme, theme_minimal, labs
import os

# Generate violin plots for each feature using plotnine

output_directory = os.path.expanduser('~/Desktop/DS/kaggle_projects/Credit_Risk_Classification/Figures/')

for i in df1.columns:
    p = (ggplot(df1_final, aes(x=1, y=i))
         + geom_violin(fill="#21918c", alpha=0.6)
         + theme_minimal()
         + labs(title=f'Distribution of {i}', x='', y=i)
         + theme_minimal()  # You can customize the theme here
         + theme(figure_size=(6, 4)))  # Adjust the figure size here
    
    # Generate the filename
    filename = f"{output_directory}01_{i}_dist.jpeg"
    
    # Save the plot to the specified filename with a lower dpi
    p.save(filename, dpi=150)

    print(f"Plot saved to: {filename}")

# We can see that the distribution of each of these features is all over the place

# Since we have 149 NA values for feature 2, we can impute these using KNN 

from sklearn.impute import KNNImputer
impute = KNNImputer(n_neighbors = 3)
df1['feature_2'] = impute.fit_transform(df1[['feature_2']])

# Check the output
df1.isnull().sum()

# Pre-processing of datetime data on df2
df2['update_date'] = pd.to_datetime(df2['update_date'])
print(df2['update_date'].dtype)

df2

# Check for NAs
df2.isnull().sum()

# Count non-missing values per feature
feature_data_count = df2.count()
print(feature_data_count)

# We will need to handle missing values on the update_date column

import pandas as pd

df2['update_date'] = pd.to_datetime(df2['update_date'])

# Check if there are missing values in the 'update_date' column
missing_values = df2['update_date'].isnull().sum()
if missing_values > 0:
    print(f"There are {missing_values} missing values in the 'update_date' column.")
else:
    df2['year1'] = df2['update_date'].dt.year
    df2['month2'] = df2['update_date'].dt.month
    df2['day3'] = df2['update_date'].dt.day

# Create a new column to store year, month, and day components individually
df2['report_year1'] = df2['update_date'].dt.year
df2['report_month'] = df2['update_date'].dt.month
df2['report_day'] = df2['update_date'].dt.day

# Similar as before, convert 'report_date' to datetime
df2['report_date'] = pd.to_datetime(df2['report_date'])  

# Use .dt accessor to extract year, month, and day
df2['year1'] = df2['report_date'].dt.year
df2['month2'] = df2['report_date'].dt.month
df2['day3'] = df2['report_date'].dt.day

# >>> df2.isnull().sum()
#id                    0
#OVD_t1                0
#OVD_t2                0
#OVD_t3                0
#OVD_sum               0
#pay_normal            0
#prod_code             0
#prod_limit         6118
#update_date          26
#new_balance           0
#highest_balance     409
#report_date        1114
#report_year1         26
#report_month         26
#report_day           26
#year1              1114
#month2             1114
#day3               1114
#dtype: int64

df2

# Impute values as before
impute = KNNImputer(n_neighbors= 3)
cols = ['prod_limit','highest_balance', 'report_year1', 'report_month', 'report_day', 'year1', 'month2', 'day3']
#df2[cols] = df2[cols].round()
for i in cols:
    if i in df2.columns:  # Check if the column exists in the DataFrame
        df2[i] = df2[i].round()  # Round values to integers
        df2[i] = impute.fit_transform(df2[[i]])

df2

cols = ['report_year1','report_month','report_day','year1','month2','day3']

for i in cols:
    df2[i] = df2[i].astype(int)

# Check
df2.info()
df2.isnull().sum()

# Note: I'm not sure that imputation of the missing dates is something I would choose to do personally due to overfitting, convergence, and bias issues
# with all applicable methods. For that reason I would need to know more about the nature of the incidents leading to missing datetime data. On that note, 
# it might just be best to merge on a common field and dropping the datetime columns.
# Somethind else, the was pandas deals with datetime data is cumbersome imo. 
# A simple sed command to replace every "/" and separating each resulting value into discrete columns would probably be more efficient.


# Reload the data and check for missing values once more
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

# check again (there should be zero NAs)
merged_df.isnull().sum()
merged_df

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
)) #+
#p9.theme(legend_title=p9.element_blank())  # Remove legend title if wanted. Remember to remove the parenthesis preceding the + before p9.theme
#)
