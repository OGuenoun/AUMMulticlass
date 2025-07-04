import plotnine as p9
import pandas as pd
initial=pd.read_csv("Initial_ROC_data.csv")
final=pd.read_csv("Final_ROC_data.csv")
auc_initial=initial.iloc[0,-1]
auc_final=final.iloc[0,-1]
label_initial = f'Initial ROC (AUC = {auc_initial:.3f})'
label_final = f'Final ROC (AUC = {auc_final:.3f})'
gg_plot_roc = p9.ggplot()+\
    p9.theme(figure_size=(4,4))+\
    p9.coord_equal()+\
    p9.geom_line(
        data=initial,
        mapping=p9.aes(x="FPR", y="TPR", color=f'"{label_initial}"'),
        size=0.3
    )+\
    p9.geom_line(
        data=final,
        mapping=p9.aes(x="FPR", y="TPR", color=f'"{label_final}"'),
        size=0.3
    )+\
     p9.labs(color="Legend", title="ROC Curves")
fig = gg_plot_roc.draw()
fig.savefig("ROC_Linear_Training_AUM.png", dpi=300, bbox_inches='tight')