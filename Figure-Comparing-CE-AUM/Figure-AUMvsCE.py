import pandas as pd 
import torch
import matplotlib.pyplot as plt
df=pd.read_csv("./Figure-Comparing-CE-AUM/AUMvsCE.csv")
def extract_float(s):
    import re
    if not isinstance(s, str):
        return s
    match = re.search(r"tensor\(([\d.]+)\)", s)
    return float(match.group(1))if match else None

# Apply the function to every cell
df_1 = df.iloc[:,1:3]
df_1 = df_1.applymap(extract_float)
df_long = df_1.melt(var_name="loss_function", value_name="auc")
summary_df = (
    df_long.groupby("loss_function")
    .agg(
        median_auc=("auc", "median"),
        q1_auc=("auc", lambda x: x.quantile(0.25)),
        q3_auc=("auc", lambda x: x.quantile(0.75))
    )
    .reset_index()
)
from plotnine import (
    ggplot, aes, geom_point, geom_errorbarh, labs,
    theme_bw, theme, element_text, scale_x_continuous
)

AUMvsCE=ggplot(summary_df, aes(x='median_auc', y='loss_function'))+\
    geom_point(size=2)+\
    geom_errorbarh(aes(xmin='q1_auc', xmax='q3_auc'), height=0.2)+\
    labs(
        x='Test AUC, median and quartiles over imbalanced training runs',
        y='Loss function',
    )+\
    scale_x_continuous(limits=(0.90, 0.95))+\
    theme_bw()+\
    theme(
        figure_size=(5, 2.5),
        axis_title=element_text(size=10),
        axis_text=element_text(size=8),
        plot_title=element_text(size=10, weight='bold')
    )


fig=AUMvsCE.draw()
fig.savefig("AUMvsCE.png", dpi=300, bbox_inches='tight')