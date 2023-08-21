import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
import plotnine as p9
from plotnine import ggplot, aes, geom_tile, geom_violin, geom_boxplot, geom_label, scale_fill_cmap, scale_fill_gradient, scale_fill_manual, theme_minimal, labs, element_text, facet_wrap
import plydata.cat_tools as cat
import os

df_com = pd.read_csv ('~/Desktop/DS/kaggle_projects/Credit_Risk_Classification/data/common_high_risk_cases.csv')
df_mis = pd.read_csv ('~/Desktop/DS/kaggle_projects/Credit_Risk_Classification/data/missing_ids.csv')
df_mis_xgb = pd.read_csv('~/Desktop/DS/kaggle_projects/Credit_Risk_Classification/data/xgb_missing_ids.csv')
df_mis_dt = pd.read_csv('~/Desktop/DS/kaggle_projects/Credit_Risk_Classification/data/dt_missing_ids.csv')

######
# Plot all high risk cases predicted by both XGBoost and DT models
import plotnine as p9
from plotnine import aes, geom_tile, geom_label, scale_fill_cmap, theme_minimal, labs, element_text
import plydata.cat_tools as cat

tidy_corr = df_com \
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
    title="Correlation Matrix ML predicted high risk cases",
    x="", y=""
) +
p9.theme(
    axis_text_x=element_text(rotation=70, hjust=1),
    figure_size=(8, 6)
)) 


###########
# Now the missing cases for the DT model (i.e. the ones predicted by XGBoost but not DT)

tidy_corr = df_mis_dt \
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
    title="Correlation Matrix XGBoost unique high risk cases",
    x="", y=""
) +
p9.theme(
    axis_text_x=element_text(rotation=70, hjust=1),
    figure_size=(8, 6)
)) 


############
# And for those predicted by DT but not XGBoost
tidy_corr = df_mis_xgb \
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
    title="Correlation Matrix DT unique high risk cases",
    x="", y=""
) +
p9.theme(
    axis_text_x=element_text(rotation=70, hjust=1),
    figure_size=(8, 6)
)) 