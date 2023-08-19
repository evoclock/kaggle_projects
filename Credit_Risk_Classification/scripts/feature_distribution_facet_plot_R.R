library(readr)
library(dplyr)
library(forcats)
library(ggplot2)
library(viridis)


dat = read.csv("./data/plot_df.csv")
dat = dat %>% 
  mutate(feature = fct_reorder(feature, label))

# Get the order of levels after fct_reorder
facet_order = unique(dat$feature)
# Reorder the levels of the factor
dat$feature = factor(dat$feature, levels = facet_order)

ggplot(dat, aes(y = value, x = feature)) + 
  geom_violin(aes(fill = feature), 
              stat = "ydensity", 
              position = "dodge", 
              alpha = 0.5, 
              trim = TRUE, 
              scale = "area") + 
  facet_wrap(~ feature, scales = "free") + 
  theme_minimal() + 
  scale_fill_viridis(discrete = TRUE, option = "D") +
  theme(text = element_text(family = "sans", 
                            face = "plain", 
                            color = "#000000", 
                            size = 15, 
                            hjust = 0.5, 
                            vjust = 0.5)) + 
  guides(fill = guide_legend(title = "feature")) + 
  ggtitle("Distribution per feature") + 
  xlab("Feature") + ylab("value")

