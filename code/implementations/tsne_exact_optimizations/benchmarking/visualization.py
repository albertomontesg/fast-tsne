import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['figure.figsize'] = (8.1, 5)
mpl.rcParams['figure.dpi'] = 600
mpl.rcParams['font.family'] = 'Roboto'
mpl.rcParams['font.size'] = 15


def plot(x,
         y,
         labels=None,
         title=None,
         ylim=2,
         legend=True,
         ylabel="flops/cycle",
         store=False, store_name=""):

    lw = 2
    marker = "s"
    markersize = 8
    labels_fontsize = 12
    fontsize = 20
    xlabel = "n"

    fig = plt.figure()
    if title:
        fig.suptitle(title, fontweight='bold', fontsize=16, x=0, ha='left')
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=.88, left=0, right=1)

    y = np.array(y)
    if len(y.shape) == 1:
        ax.plot(x, y, linewidth=lw, marker=marker, markersize=markersize)
    else:
        assert type(labels) in (list, tuple)
        assert len(labels) == len(y)

        for i, l in enumerate(labels):
            ax.plot(
                x,
                y[i],
                linewidth=lw,
                marker=marker,
                markersize=markersize,
                label=l)

        if legend:
            ax.legend()
    ax.set_ylim([0, ylim])

    ax.set_xlim([x.min(), x.max()])
    ax.set_xlabel(xlabel, fontsize=labels_fontsize)

    # For runtime plots where on the yaxis there is an exponent, place the text
    # on the axis title
    exponent_text = ''
    ax.ticklabel_format(axis='y', style='sci')
    plt.draw()
    if ax.yaxis.get_offset_text().get_text() != '':
        ax.yaxis.major.formatter._useMathText = True
        plt.draw()
        exponent_text = '[{}]'.format(ax.yaxis.get_offset_text().get_text())
        ax.yaxis.offsetText.set_visible(False)

    ax.set_title(
        '[{}]{}'.format(ylabel, exponent_text),
        fontsize=labels_fontsize,
        position=(0, 1.0),
        ha='left',
        va='bottom')

    ax.yaxis.grid(color='#ffffff', linestyle='-', linewidth=.5)
    ax.set_facecolor('#eeeeee')
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    if store:
        import os
        folder = "plots"
        if not os.path.exists(folder):
            os.makedirs(folder);
        
        fig.savefig(folder + "/" + store_name + ".png", bbox_inches='tight')
        
    return ax
