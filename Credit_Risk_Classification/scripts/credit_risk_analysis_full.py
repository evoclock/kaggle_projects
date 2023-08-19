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


output_directory = os.path.expanduser('~/Desktop/DS/kaggle_projects/Credit_Risk_Classification/Figures/')

for i in merged_df.columns:
    p = (ggplot(merged_df, aes(x=1, y=i))
         + geom_violin(fill="#21918c", alpha=0.6)
         + theme_minimal()
         + labs(title=f'Distribution of {i}', x='', y=i)
         + theme_minimal()  # You can customize the theme here
         + theme(figure_size=(6, 4)))  # Adjust the figure size here
    
    # Generate the filename
    filename = f"{output_directory}03_{i}_dist.jpeg"
    
    # Save the plot to the specified filename with a lower dpi
    p.save(filename, dpi=150)

    print(f"Plot saved to: {filename}")

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

tidy_merged = merged_df \
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

print(tidy_merged)
tidy_merged.info()

# Create a faceted plot using seaborn's facet_grid
g = sns.FacetGrid(data=tidy_merged, col='var2', col_wrap=4, sharey=False)
g.map_dataframe(sns.boxplot, x='var1', y='value')
g.set_axis_labels('var1', 'value')
g.set_titles(col_template="{col_name}")
g.fig.suptitle("Faceted Box Plot by var2", y=1.02)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()


# Create a DataFrame suitable for plotting
#plot_df = merged_df.melt(id_vars=['id', 'label'], var_name='feature')
#
# Create the faceted plot
#g = sns.FacetGrid(plot_df, col='feature', col_wrap=5, height=4, sharey=False)
#g.map(sns.violinplot, 'value', color='#21918c', alpha=0.6)
#g.set_titles(col_template="{col_name}")
#g.set_axis_labels("", "")
#g.set_xticklabels([])
#g.fig.suptitle("Distribution of Features", y=1.02)
#g.tight_layout()
#
#plt.show()

# Generate the filename
#filename = f"{output_directory}04_facet_plot.jpeg"
#
# Save the plot
#g.savefig(filename, dpi=350)
#print(f"Faceted plot saved to: {filename}")


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