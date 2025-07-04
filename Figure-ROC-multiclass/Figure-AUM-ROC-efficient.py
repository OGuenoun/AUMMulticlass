import plotnine as p9
import pandas as pd
import matplotlib.pyplot as plt
p9.options.figure_size=(8,4)
roc_efficient_df= pd.read_csv('ROC-efficient-points.csv')
gg_roc_efficient = p9.ggplot()+\
    p9.theme(figure_size=(4,4))+\
    p9.coord_equal()+\
    p9.geom_line(
        p9.aes(
            x="FPR",
            y="TPR",
        ),
        data=roc_efficient_df
    )+\
    p9.geom_point(
        p9.aes(
            x="FPR",
            y="TPR",
        ),
        data=roc_efficient_df
    )
fig = gg_roc_efficient.draw()
fig.savefig("ROC_multiclass_micro_efficient_plot.png", dpi=300, bbox_inches='tight')