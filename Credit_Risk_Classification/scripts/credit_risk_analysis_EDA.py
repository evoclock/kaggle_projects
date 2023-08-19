import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv ('~/Desktop/DS/kaggle_projects/Credit_Risk_Classification/customer_data.csv')
df2 = pd.read_csv ('~/Desktop/DS/kaggle_projects/Credit_Risk_Classification/payment_data.csv')

df1.info()
df1
df2

for i in df1.columns:
    values = df1[i].value_counts
    print(values)


# count the number of instances NAs are present per column
df1.isnull().sum()
# feature 2 has 149 such values

# Plot a correlogram to see which features might be more informative
df1_corr = df1.corr()

# first make the data into long (tidy) format
import plotnine as p9
import plydata.cat_tools as cat

tidy_corr = df1 \
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


# Seaborn being a native library is of course much less verbose but there are plots for which ggplot is still superior IMO
#plt.figure(figsize=(15,10))
#sns.heatmap(df1.corr(), annot=True)
#plt.show()

for i in df1.columns:
    print('columns: ', i)
    sns.set_theme(style="ticks")
    sns.boxplot(df1[i], saturation=.4)
    sns.stripplot(data=df1[i], alpha=.4)
    plt.show()







