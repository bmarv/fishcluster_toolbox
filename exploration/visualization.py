import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


def significance_stars(p_value):
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""


def plot_correlogram_with_hue_title_log(
    dataframe, hue_column, title=None, log_scale=False, fig_name=None
):
    palette = sns.color_palette("Set2", len(dataframe[hue_column].unique()))
    hue_colors = dict(zip(dataframe[hue_column].unique(), palette))

    g = sns.PairGrid(dataframe, hue=hue_column, palette="Set2")

    # Diagonal: Density plots
    if log_scale:
        g.map_diag(sns.kdeplot, fill=True, alpha=0.3, log_scale=True)
    else:
        g.map_diag(sns.kdeplot, fill=True, alpha=0.3)

    # Lower: Scatter plots with hue
    g.map_lower(sns.scatterplot, alpha=0.8, s=2.5)

    # Upper: Correlation and significance
    def corrfunc(x, y, **kwargs):
        ax = plt.gca()
        y_pos = 0.85

        # overall correlation
        if not hasattr(ax, "overall_corr_printed"):
            if len(x) >= 2 and len(y) >= 2:
                r_all, p_all = pearsonr(dataframe[x.name], dataframe[y.name])
                ax.text(
                    0.5,
                    y_pos,
                    f"All: {r_all:.5f}{significance_stars(p_all)}",
                    fontsize=10,
                    ha="center",
                    transform=ax.transAxes,
                )
            ax.overall_corr_printed = True

        # per-category correlation
        for category, color in hue_colors.items():
            mask = dataframe[hue_column] == category
            x_cat = x[mask]
            y_cat = y[mask]
            y_pos -= 0.15
            if len(x_cat) >= 2 and len(y_cat) >= 2:
                r_cat, p_cat = pearsonr(x_cat, y_cat)
                ax.text(
                    0.5,
                    y_pos,
                    f"{category}: {r_cat:.5f}{significance_stars(p_cat)}",
                    fontsize=10,
                    ha="center",
                    transform=ax.transAxes,
                    color=color,
                )

    g.map_upper(corrfunc)
    g.add_legend()

    if title:
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(title)

    if log_scale:
        for ax in g.axes.flatten():
            if ax is not None:
                ax.set_xscale("log")
                ax.set_yscale("log")

    if fig_name is not None:
        plt.savefig(fig_name)
