import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



# =====================================================
# CUSTOM CONFUSION MATRIX
# =====================================================

def confusion_plot(cm):


    labels = [

        "Actual",

        "Edited",

        "AI Generated"

    ]



    fig, ax = plt.subplots(
        figsize=(8, 6)
    )



    # Colors according to actual classes

    row_colors = [

        "#A5D6A7",   # Green - Actual

        "#FFCC80",   # Orange - Edited

        "#EF9A9A"    # Red - AI Generated

    ]



    for i in range(3):

        for j in range(3):


            rect = plt.Rectangle(

                (j, i),

                1,

                1,

                facecolor=row_colors[i],

                edgecolor="black",

                linewidth=2

            )


            ax.add_patch(rect)



            ax.text(

                j + 0.5,

                i + 0.5,

                str(cm[i, j]),

                ha="center",

                va="center",

                fontsize=16,

                fontweight="bold"

            )



    ax.set_xlim(
        0,
        3
    )


    ax.set_ylim(
        3,
        0
    )



    ax.set_xticks(
        np.arange(3)+0.5
    )


    ax.set_yticks(
        np.arange(3)+0.5
    )



    ax.set_xticklabels(

        labels,

        fontsize=12,

        fontweight="bold"

    )



    ax.set_yticklabels(

        labels,

        fontsize=12,

        fontweight="bold"

    )



    # Label colors

    class_colors = [

        "green",

        "orange",

        "red"

    ]



    for tick,color in zip(
        ax.get_xticklabels(),
        class_colors
    ):

        tick.set_color(color)



    for tick,color in zip(
        ax.get_yticklabels(),
        class_colors
    ):

        tick.set_color(color)



    ax.set_xlabel(

        "Predicted Class",

        fontsize=13,

        fontweight="bold"

    )


    ax.set_ylabel(

        "Actual Class",

        fontsize=13,

        fontweight="bold"

    )



    ax.set_title(

        "Confusion Matrix",

        fontsize=16,

        fontweight="bold"

    )



    ax.set_aspect(
        "equal"
    )



    plt.tight_layout()



    return fig






# =====================================================
# PRECISION - RECALL - F1 SCORE GRAPH
# =====================================================


def metrics_plot(report):


    df = pd.DataFrame(
        report
    ).transpose()



    # Select only classes

    df = df.loc[

        [

            "Actual",

            "Edited",

            "AI Generated"

        ]

    ]



    fig, ax = plt.subplots(

        figsize=(8,5)

    )



    x = np.arange(
        len(df)
    )


    width = 0.25



    ax.bar(

        x - width,

        df["precision"],

        width,

        label="Precision"

    )



    ax.bar(

        x,

        df["recall"],

        width,

        label="Recall"

    )



    ax.bar(

        x + width,

        df["f1-score"],

        width,

        label="F1 Score"

    )



    ax.set_xticks(
        x
    )



    ax.set_xticklabels(

        df.index,

        fontsize=11

    )



    ax.set_ylim(

        0,

        1

    )



    ax.set_ylabel(

        "Score"

    )



    ax.set_title(

        "Precision, Recall and F1 Score",

        fontsize=15,

        fontweight="bold"

    )



    ax.legend()



    plt.tight_layout()



    return fig