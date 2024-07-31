import glob
import os
import pandas as pd
from scipy.stats import entropy as entropy_m
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
import hdf5storage
import sys
from tqdm import tqdm
from config import BLOCK1, BLOCK2, HOURS_PER_DAY
from utils.analyses_helper import (
    remove_spines,
    get_regions_for_fish_key,
    load_clusters_concat,
    load_trajectory_data_concat,
    average_fit_plot,
    sparse_scatter_plot,
    get_polys,
)
from utils.utils import (
    get_days,
    get_individuals_keys,
    set_parameters,
    split_into_batches,
)

DIR_PLASTCITY = "plasticity"


def day2date(d):
    return "%s.%s.%s" % (d[4:6], d[6:8], d[:4])


def correlation_fit(x, y, deg=1):
    if x.shape[0] < deg:
        return [], [], np.nan, [np.nan for i in range(deg + 1)]
    abc = np.polyfit(x, y, deg)
    corrcof = pearsonr(x, y)
    time = np.linspace(x.min(), x.max(), 100)
    fit_y = np.polyval(abc, time)
    return time, fit_y, corrcof[0], abc


def plasticity_for_file(file, effects=[]):
    """Plots the plasticity for the given file.
    effects: list of strings, can be "scatter", "sort"
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    xlabel = "hour" if "hour" in file else "day"
    ylabel = "CV" if "cv/cv" in file else "entropy"
    clustering_m = "cv"
    if ylabel == "entropy":
        clustering_m = "kmeans" if "kmeans" in file else "wshed"
    num = os.path.basename(file)[-8:-4]
    df = pd.read_csv(file, index_col=0)
    df = sort_by_entropy_mean(df)
    t = df.index
    title = get_title(file)
    op_weights = np.array([0.7] * df.columns.size)

    if "scatter" in effects:
        sparse_scatter_plot(df[df.columns[[0, -1]]], ax)
    if "sort" in effects:
        op_weights = np.array([0.1] * df.columns.size)
        op_weights[:1] = 1
        op_weights[-1:] = 1
        print(df.columns[:3], df.columns[-3:])
    polys = get_polys(df)
    effect = "_".join(effects)
    average_fit_plot(
        t, polys, ax=ax, title=title,
        xlabel=xlabel, ylabel=ylabel, alpha=op_weights
    )
    f_name = (
        os.path.dirname(file)
        + f"/{clustering_m}_{xlabel}_{ylabel}_{title}_{num}{effect}.pdf"
    )
    remove_spines(ax)
    plt.ylim(0, 2.25)
    plt.legend(frameon=False)
    fig.savefig(f_name)
    plt.close()
    print(f_name)
    return fig


def sort_by_entropy_mean(df):
    means = df.mean(numeric_only=True)
    s_sorted = means.sort_values()
    return df[s_sorted.keys()]


def get_title(file):
    names = ["step length", "turning angle", "distance to wall", "entropy"]
    for n in names:
        if n in file:
            return n
    raise ValueError(
        "file name does not contain any of the following: %s" % names
    )


def plot_index_columns(
    df=None,
    columns=None,
    title=None,
    xlabel="index",
    ylabel="entropy",
    filename=None,
    ax=None,
    forall=True,
    fit_degree=1,
):
    if df is None:
        if filename is None:
            raise ValueError("filename must be provided if df is not provided")
        df = pd.read_csv(filename + ".csv")
    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        axis = fig.subplots()
    else:
        axis = ax

    if columns is None:
        columns = df.columns

    colors_map = plt.cm.get_cmap(lut=len(columns))
    df = df[columns]
    leg_h = []
    for i, col in enumerate(columns):
        time = df.index.to_numpy()
        y = df[col].to_numpy().astype(np.float64)
        f_nan = ~np.isnan(y)
        x, y = time[f_nan], y[f_nan]
        axis.scatter(x=x, y=y, color=colors_map(i), s=1)
        if not forall:
            t, f_y, corr, abc = correlation_fit(x, y, deg=fit_degree)
            (line,) = axis.plot(t, f_y, "-", color=colors_map(i))
            leg_h.append((line, "%s - pearsonr: %0.3f" % (col[13:17], corr)))

    if forall:
        x = (
            np.column_stack((df.index.to_numpy() for i in range(len(columns))))
            .reshape(-1)
            .astype(int)
        )
        y = df.to_numpy().reshape(-1).astype(np.float64)
        is_not_nan = ~np.isnan(y)
        x, y = x[is_not_nan], y[is_not_nan]
        t, f_y, corr, abc = correlation_fit(x, y, deg=fit_degree)
        (line,) = axis.plot(t, f_y, "-", color="k", markersize=12)
        leg_h.append((line, "pearsonr: %0.3f" % (corr)))
    axis.legend(*zip(*leg_h), loc="best")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_xlabel(xlabel)
    remove_spines(axis)
    if ax is not None:
        return None
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)
    return fig


def compute_cluster_entropy(
    parameters,
    get_clusters_func,
    fish_keys,
    n_clusters,
    name="cluster_entropy_for_days",
    by_the_hour=False,
):
    days = get_days(parameters, prefix=fish_keys[31].split("_")[0])
    columns = fish_keys
    index = list(
        range(1, 1 + (len(days) * HOURS_PER_DAY if by_the_hour else len(days)))
    )
    entro = np.empty((len(index), len(columns)))
    entro.fill(np.nan)
    for j, fk in tqdm(
        enumerate(fish_keys), total=len(fish_keys),
        desc="fish-keys", file=sys.stdout
    ):
        for i, d in enumerate(get_days(parameters=parameters, prefix=fk)):
            clusters = get_clusters_func(fk, d)
            if clusters is not None:
                if by_the_hour:
                    time_df = load_trajectory_data_concat(parameters, fk, d)[
                        "df_time_index"
                    ]
                    time_split, cluster_hourly = split_into_batches(
                        time_df, clusters, batch_size=5 * 60 * 60, flt=True
                    )
                    h_of_split = [
                        int(
                            HOURS_PER_DAY * (i) +
                            ((t[0] - time_df[0]) // (5 * (60**2)))
                        )
                        for t in time_split
                    ]
                    entro[h_of_split, j] = [
                        entropy_m(compute_cluster_distribution(c, n_clusters))
                        for c in cluster_hourly
                    ]
                else:
                    dist = compute_cluster_distribution(clusters, n_clusters)
                    entro[i, j] = entropy_m(dist)

    dir_p = f"{parameters.projectPath}/{DIR_PLASTCITY}/{name}"
    os.makedirs(dir_p, exist_ok=True)
    all_vals_df = pd.DataFrame(entro, columns=fish_keys, index=index)
    if not by_the_hour:
        df_test_1 = pd.DataFrame(
            {"date_block1": map(
                day2date, get_days(parameters, prefix="block1")
            )}
        )
        df_test_1.index += 1
        df_test_2 = pd.DataFrame(
            {"date_block2": map(
                day2date, get_days(parameters, prefix="block2")
            )},
            index=all_vals_df.index,
        )
        len_diff_df = len(all_vals_df) - len(df_test_1)
        if len_diff_df > 0:
            nan_values = pd.DataFrame(
                {list(df_test_1.columns)[0]: [np.nan] * len_diff_df}
            )
            df_test_1 = pd.concat([df_test_1, nan_values], ignore_index=True)
            df_test_1.index += 1
        all_vals_df = pd.concat([all_vals_df, df_test_1, df_test_2], axis=1)
    time_str = "hourly" if by_the_hour else "daily"

    filename = f"{dir_p}/{name}_{time_str}_%.3d" % n_clusters
    all_vals_df.to_csv(f"{filename}.csv")
    return all_vals_df


def compute_cluster_distribution(clusters, n):
    uqs, counts = np.unique(clusters, return_counts=True)
    dist = np.zeros(n)[(uqs - 1)] = counts / clusters.shape[0]
    return dist


def compute_coefficient_of_variation(
    parameters,
    fish_keys,
    n_df=50,
    forall=False,
    by_the_hour=True,
    fit_degree=1
):
    """
    Compute the coefficient of variation for the step length, turning angle
    and distance to wall for each fish and each day. The coefficient of
    variation is computed for each hour of the day
    parameters: parameters object
    fish_keys: list of fish keys
    n_df: number of data frames to average over
    forall: interpolate over all data points
    by_the_hour: compute the coefficient of variation for each hourly or daily
    fit_degree: degree of the polynomial fit
    """
    fig = plt.figure(figsize=(10, 10))
    axes = fig.subplots(3, 1)
    fig.suptitle("Coefficient of Variation for %0.1f sec data" % (n_df / 5))
    names = ["step length", "turning angle", "distance to wall"]
    days = get_days(parameters, prefix=BLOCK1)
    size = len(days) * (HOURS_PER_DAY) if by_the_hour else len(days)
    sum_data = [np.full((size, len(fish_keys)), np.nan) for i in range(3)]
    for j, fk in enumerate(fish_keys):
        for di, day in enumerate(
            get_days(parameters=parameters, prefix=fk.split("_")[0])
        ):
            sum_data_fk = load_trajectory_data_concat(
                parameters,
                fk=fk,
                day=day
            )
            if sum_data_fk is None:
                continue
            data = sum_data_fk["projections"]
            time_df = sum_data_fk["df_time_index"].flatten()
            for i in range(3):
                if by_the_hour:
                    _, datas = split_into_batches(
                        time_df, data[:, i] if i != 1 else data[:, i] + np.pi
                    )  # shift for turning angle
                    for h, data_in in enumerate(datas):
                        new_len = data_in.size // n_df
                        data_means = (
                            data_in[: n_df * new_len]
                            .reshape((new_len, n_df))
                            .mean(axis=1)
                        )
                        cv = data_means.std() / data_means.mean()
                        sum_data[i][(di * HOURS_PER_DAY) + (h), j] = cv
                else:
                    data_in = data[:, i] if i != 1 else data[:, i] + np.pi
                    new_len = data_in.size // n_df
                    data_means = (
                        data_in[: n_df * new_len].reshape((
                            new_len, n_df
                        )).mean(axis=1)
                    )
                    cv = data_means.std() / data_means.mean()
                    sum_data[i][di, j] = cv
    filename = f"{parameters.projectPath}/{DIR_PLASTCITY}/cv"
    os.makedirs(filename, exist_ok=True)
    time_str = "hourly" if by_the_hour else "daily"
    for i in range(3):
        df = pd.DataFrame(
            sum_data[i],
            columns=fish_keys,
            index=range(1, size + 1)
        )
        if not by_the_hour:
            df = df.join(
                pd.DataFrame(
                    {
                        f"date_{BLOCK1}": map(
                            day2date, get_days(parameters, prefix=BLOCK1)
                        ),
                        f"date_{BLOCK2}": map(
                            day2date, get_days(parameters, prefix=BLOCK2)
                        ),
                    },
                    index=df.index,
                ),
                how="left",
            )
        _ = plot_index_columns(
            df,
            ax=axes[i],
            columns=fish_keys,
            ylabel="cv",
            xlabel="hour",
            title=names[i],
            forall=forall,
            fit_degree=fit_degree,
        )
        df.to_csv(f"{filename}/cv_{names[i]}_{time_str}_ndf_%.3d.csv" % n_df)

    fig.tight_layout()
    fig.savefig(f"{filename}/cv_{time_str}_ndf_{n_df}.pdf")
    return fig


def for_all_cluster_entropy(parameters, fish_ids, cluster_sizes_list):
    print("hours/day", HOURS_PER_DAY)
    get_clusters_func = lambda fk, d: load_clusters_concat(parameters, fk, d)
    for k in cluster_sizes_list:
        print(f"cluster-size: {k}")
        parameters.kmeans = k
        wshedfile = hdf5storage.loadmat(
            "%s/%s/zVals_wShed_groups_%s.mat"
            % (parameters.projectPath, parameters.method, parameters.kmeans)
        )
        get_clusters_func_wshed = lambda fk, d: get_regions_for_fish_key(
            wshedfile, fk, d
        )
        for flag in [True, False]:
            for func, c_type in zip(
                [get_clusters_func_wshed, get_clusters_func],
                ["wshed", "kmeans"]
            ):
                print(f"\tcluster_method: {c_type}")
                compute_cluster_entropy(
                    parameters,
                    func,
                    fish_ids,
                    parameters.kmeans,
                    name=f"cluster_entropy_{c_type}",
                    by_the_hour=flag,
                )


if __name__ == "__main__":
    parameters = set_parameters()
    fks = get_individuals_keys(parameters)
    cluster_sizes_list = parameters.kmeans_list
    for_all_cluster_entropy(parameters, fks, cluster_sizes_list)
    for flag in [True, False]:
        compute_coefficient_of_variation(
            parameters,
            fks,
            n_df=10,
            forall=True,
            fit_degree=2,
            by_the_hour=flag
        )
        compute_coefficient_of_variation(
            parameters,
            fks,
            n_df=50,
            forall=True,
            fit_degree=2,
            by_the_hour=flag
        )
    files_daily = sorted(
        glob.glob(f"/{parameters.projectPath}/plasticity/*/*daily*.csv")
    )
    files_hourly = sorted(
        glob.glob(f"/{parameters.projectPath}/plasticity/*/*hourly*.csv")
    )
    for f in files_hourly + files_daily:
        for effect in ["sort", "scatter", ""]:
            fig = plasticity_for_file(f, effect)
