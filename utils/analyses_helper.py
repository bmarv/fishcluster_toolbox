import numpy as np
import matplotlib.pyplot as plt
import glob
import hdf5storage
from utils.utils import get_days

WAVELET = 'wavelet'
clusterStr = 'clusters'


def remove_spines(ax):
    """Remove the spines from the given axis ax."""
    for s in ax.spines.values():
        s.set_visible(False)


def average_fit_plot(
    t, polys, ax=plt, title="", xlabel="x", ylabel="y", alpha=0.5
):
    time = np.linspace(t.min(), t.max(), 100)
    if type(alpha) is float:
        alpha = [alpha] * len(polys)
    colors = get_custom_colors(len(polys))
    for i in range(len(polys)):
        ax.plot(
            time,
            np.polyval(polys[i], time),
            alpha=alpha[i],
            lw=0.7,
            color=colors[i]
        )
    ax.plot(time, np.polyval(polys.mean(axis=0), time), c="k", label="average")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return ax


def get_custom_colors(cnt):
    colors = ['darkblue', 'orange', 'red']
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        'custom_cmap', colors
    )
    indices = np.linspace(0, 1, cnt)
    colors = cmap(indices)
    return colors


def get_regions_for_fish_key(wshedfile, fish_key="", day=""):
    index_fk = np.where([
        (f"{fish_key}_{day}" in file.flatten()[0])
        for file in wshedfile['zValNames'][0]
    ])[0]
    if len(index_fk) == 0:
        print(fish_key, day, " no corresponding regions found!")
        return None
    if index_fk[0] == 0:
        idx_p = [0, wshedfile['zValLens'].flatten().cumsum()[index_fk[-1]]]
    else:
        idx_p = wshedfile['zValLens'].flatten().cumsum()[
            [index_fk[0]-1, index_fk[-1]]
        ]
    return wshedfile['watershedRegions'].flatten()[idx_p[0]:idx_p[1]]


def load_clusters(parameters, fk="", day="", k=None):
    if k is None:
        k = parameters.kmeans
    data_by_day = []
    pfile = glob.glob(
        parameters.projectPath +
        f'/Projections/{fk}*_{day}*_pcaModes_{clusterStr}_{k}.mat'
        )
    pfile.sort()
    for f in pfile:
        data = hdf5storage.loadmat(f)
        data_by_day.append(data)
    return data_by_day


def load_clusters_concat(parameters, fk="", day="", k=None):
    clusters_by_day = load_clusters(parameters, fk, day, k)
    if len(clusters_by_day) == 0:
        return None
    clusters = np.concatenate(
        [x["clusters"] for x in clusters_by_day], axis=1
    ).flatten()
    return clusters


def load_trajectory_data(parameters, fk="", day=""):
    data_by_day = []
    pfile = glob.glob(
        parameters.projectPath+f'/Projections/{fk}*_{day}*_pcaModes.mat'
    )
    pfile.sort()
    for f in pfile:
        data = hdf5storage.loadmat(f)
        data_by_day.append(data)
    return data_by_day


def load_trajectory_data_concat(parameters, fk="", day=""):
    data_by_day = load_trajectory_data(parameters, fk, day)
    if len(data_by_day) == 0:
        return None
    daily_df = 5*(60**2)*8
    positions = np.concatenate([trj["positions"] for trj in data_by_day])
    projections = np.concatenate([trj["projections"] for trj in data_by_day])
    days = sorted(list(set(map(
        lambda d: d.split("_")[0],
        get_days(parameters, prefix=fk.split("_")[0])
    ))))
    df_time_index = np.concatenate([
        trj["df_time_index"].flatten() +
        (daily_df*days.index(trj["day"].flatten()[0].split("_")[0]))
        for trj in data_by_day
    ])
    area = data_by_day[0]["area"]
    return dict(
        positions=positions,
        projections=projections,
        df_time_index=df_time_index,
        area=area
    )


def sparse_scatter_plot(df, ax):
    colors = get_custom_colors(df.columns.size)
    for i, col in enumerate(df.columns):
        x = df.index.to_numpy()
        y = df[col]
        ax.scatter(x=x, y=y, s=0.5, color=colors[i])


def get_polys(df):
    t = df.index
    return np.array([np.polyfit(
        t[np.isfinite(df[c])], df[c][np.isfinite(df[c])], deg=2
    ) for c in df.columns[:45]])
