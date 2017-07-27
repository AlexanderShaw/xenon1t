import os, sys
import numpy as np
from multihist import Histdd, Hist1d
import matplotlib
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit, minimize

def slicing(df, x_feat, y_feat, bins):
    """Designed to slice in a 2-d space, fit gauss to those slices, and return a binned dataframe"""
    bin_mids = []
    
    sigmas = []
    means = []

    for i in range(1, len(bins)):
        a = df.loc[(df[x_feat] < bins[i]) & (df[x_feat] > bins[i-1])]

        mn = np.mean(a[y_feat])
        sg = np.std(a[y_feat])
        bm = np.mean(a[x_feat])

        if (np.logical_not(np.isnan(mn)) & np.logical_not(np.isnan(sg))):
            means.append(mn)
            sigmas.append(sg)
            bin_mids.append(bm)

    return pd.DataFrame(np.array([bin_mids, means, sigmas]).T, index = range(len(means)), columns = [x_feat,y_feat,'sigma'])



def plot2d_compare(x, y,xlim=None, ylim=None):
    plt.figure(figsize=(8,6))
    plt.scatter(ambe[x], ambe[y], color='r', label='ambe', s=0.2)

    plt.scatter(sim[x], sim[y], color='b', label='sim', s=0.2)

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.xlabel(x)
    plt.ylabel(y)

    plt.legend(loc='upper left')



def line_cut(df, values, less_greater, value):
    if less_greater in ("less","<","less than"):
        return df[df[values] < value]
    elif less_greater in ("greater",">","greater than"):
        return df[df[values] > value]



def four_hist_compare(df_1, df_2, slice_in_x, y, plot_ranges):
    """Will take in two dataframes and plot their histograms in y atop eachother in the 4 ranges of slice_in_x
    specified by plot_ranges"""
    fig = plt.figure(figsize=(10,7))
    # ----------------------------------------
    ax = fig.add_subplot(221)

    slice_df_1 = df_1[(df_1[slice_in_x] > plot_ranges[0][0]) & (df_1[slice_in_x] < plot_ranges[0][1])]
    slice_df_2 = df_2[(df_2[slice_in_x] > plot_ranges[0][0]) & (df_2[slice_in_x] < plot_ranges[0][1])]

    ax.hist(slice_df_1[y].dropna(), bins=50, color='b', alpha=1, label='sim', normed=1)
    ax.hist(slice_df_2[y].dropna(), bins=50, color='r', alpha=0.5, label='ambe', normed=1)

    ax.set_title( "%s to %s, %s" % (plot_ranges[0][0], plot_ranges[0][1], slice_in_x))


    # ----------------------------------------
    ax = fig.add_subplot(222)

    slice_df_1 = df_1[(df_1[slice_in_x] > plot_ranges[1][0]) & (df_1[slice_in_x] < plot_ranges[1][1])]
    slice_df_2 = df_2[(df_2[slice_in_x] > plot_ranges[1][0]) & (df_2[slice_in_x] < plot_ranges[1][1])]

    ax.hist(slice_df_1[y].dropna(), bins=40, color='b', alpha=1, label='sim', normed=1)
    ax.hist(slice_df_2[y].dropna(), bins=40, color='r', alpha=0.5,  label='ambe', normed=1)

    ax.set_title( "%s to %s, %s" % (plot_ranges[1][0], plot_ranges[1][1], slice_in_x))

    ax.legend(loc = 'upper right')

    # ----------------------------------------
    ax = fig.add_subplot(223)

    slice_df_1 = df_1[(df_1[slice_in_x] > plot_ranges[2][0]) & (df_1[slice_in_x] < plot_ranges[2][1])]
    slice_df_2 = df_2[(df_2[slice_in_x] > plot_ranges[2][0]) & (df_2[slice_in_x] < plot_ranges[2][1])]

    ax.hist(slice_df_1[y].dropna(), bins=25, color='b', alpha=1, label='sim', normed=1)
    ax.hist(slice_df_2[y].dropna(), bins=25, color='r', alpha=0.5,  label='ambe', normed=1)

    ax.set_title( "%s to %s, %s" % (plot_ranges[2][0], plot_ranges[2][1], slice_in_x))
    ax.set_xlabel(y)

    # ----------------------------------------
    ax = fig.add_subplot(224)

    slice_df_1 = df_1[(df_1[slice_in_x] > plot_ranges[3][0]) & (df_1[slice_in_x] < plot_ranges[3][1])]
    slice_df_2 = df_2[(df_2[slice_in_x] > plot_ranges[3][0]) & (df_2[slice_in_x] < plot_ranges[3][1])]

    ax.hist(slice_df_1[y].dropna(), bins=35, color='b', alpha=1, label='sim', normed=1)
    ax.hist(slice_df_2[y].dropna(), bins=35, color='r', alpha=0.5, label='ambe', normed=1)

    ax.set_title( "%s to %s, %s" % (plot_ranges[3][0], plot_ranges[3][1], slice_in_x))
    ax.set_xlabel(y)


def describe_dist(df_feat):
    """ Describes the distribution of the normalized feature """
    
    df_ = df_feat.dropna()

    descript = {
    'mean' : np.mean(df_),
    'std' : np.std(df_),
    'Q1' : np.percentile(df_, 25),
    'median' : np.percentile(df_, 50),
    'Q3' : np.percentile(df_, 75)
    }

    descript = pd.DataFrame(descript, index = range(1))

    pd.set_option('display.float_format', '{:.3e}'.format)

    print(descript)