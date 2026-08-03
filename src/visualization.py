import matplotlib.pyplot as plt
import seaborn as sns



def confusion_plot(cm):


    labels=[

        "Actual",

        "Edited",

        "AI Generated"

    ]



    fig,ax=plt.subplots(
        figsize=(7,6)
    )


    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        xticklabels=labels,

        yticklabels=labels

    )


    ax.set_xlabel(
        "Predicted"
    )


    ax.set_ylabel(
        "Actual"
    )


    ax.set_title(
        "Confusion Matrix"
    )


    return fig