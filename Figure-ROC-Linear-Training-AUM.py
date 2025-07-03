import plotnine as p9
import pandas as pd
initial=pd.read_csv("Initial_ROC_data.csv")
final=pd.read_csv("Final_ROC_data.csv")

gg_plot_roc = p9.ggplot()+\
    p9.theme(figure_size=(4,4))+\
    p9.coord_equal()+\
    p9.geom_line(
        data=initial,
        mapping=p9.aes(x="FPR", y="TPR", color='"Initial ROC"'),
        size=1.2
    )+\
    p9.geom_line(
        data=final,
        mapping=p9.aes(x="FPR", y="TPR", color='"Final ROC"'),
        size=1.2
    )+\
    p9.geom_point(
        p9.aes(
            x="FPR",
            y="TPR",
        ),
        data=initial
    )+\
    p9.geom_point(
        p9.aes(
            x="FPR",
            y="TPR",
        ),
        data=final
    )+\
     p9.labs(color="Legend", title="ROC Curves")
fig = gg_plot_roc.draw()
fig.savefig("ROC_Linear_Training_AUM.png", dpi=300, bbox_inches='tight')