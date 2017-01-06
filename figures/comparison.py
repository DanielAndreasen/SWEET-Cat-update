#!/usr/bin/env python
# -*- coding: utf8 -*-

# My imports
from __future__ import division
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
sns.set_style('ticks')
sns.set_context('paper', font_scale=1.7)
# sns.set_style('white')
# sns.set_context('talk')

"""Make the comparison between previous SWEET-Cat values
and new values from this paper."""


if __name__ == '__main__':
    names = ['star', 'hd', 'ra', 'dec', 'vmag', 'ervmag', 'par', 'erpar',
             'parsource', 'teff', 'erteff', 'logg', 'erlogg', 'logglc',
             'erlogglc', 'vt', 'ervt', 'metal', 'ermetal', 'mass', 'ermass',
             'author', 'source', 'update', 'comment', 'z1', 'z2']

    df = pd.read_csv('SC.csv')
    # df = df[df.convergence]
    sc = pd.read_csv('WEBSITE_online.rdb', delimiter='\t', names=names, na_values=['NULL'])

    # Remove trailing withspaces from the names
    df.linelist = df.linelist.str.strip()
    sc.star = sc.star.str.strip()


    df.rename(columns={'teff': 'teffnew', 'tefferr': 'tefferrnew',
                       'logg': 'loggnew', 'loggerr': 'loggerrnew',
                        'feh': 'fehnew',   'feherr': 'feherrnew',
                         'vt': 'vtnew',     'vterr': 'vterrnew'},
                         inplace=True)

    df2 = pd.merge(left=sc, right=df, left_on='star', right_on='linelist', how='inner')

    columns = ['star',
               'teff', 'erteff', 'teffnew', 'tefferrnew',
               'logg', 'erlogg', 'loggnew', 'loggerrnew',
               'metal', 'ermetal', 'fehnew', 'feherrnew',
               'vt', 'ervt', 'vtnew', 'vterrnew']
    df2 = df2.loc[:, columns]
    df2.loc[:, 'teff':] = df2.loc[:, 'teff':].astype(float)
    df2['teffdiff'] = df2['teff']-df2['teffnew']
    df2['loggdiff'] = df2['logg']-df2['loggnew']
    df2['fehdiff'] = df2['metal']-df2['fehnew']
    df2['vtdiff'] = df2['vt']-df2['vtnew']

    df = df2  # Easier than renaming all below to df2...
    stat = df.describe()

    teffdiff = stat.loc['mean', 'teffdiff']
    loggdiff = stat.loc['mean', 'loggdiff']
    fehdiff = stat.loc['mean', 'fehdiff']
    vtdiff = stat.loc['mean', 'vtdiff']
    tx, ty = stat.loc['mean', 'erteff'], stat.loc['mean', 'tefferrnew']
    gx, gy = stat.loc['mean', 'erlogg'], stat.loc['mean', 'loggerrnew']
    fx, fy = stat.loc['mean', 'ermetal'], stat.loc['mean', 'feherrnew']
    vx, vy = stat.loc['mean', 'ervt'], stat.loc['mean', 'vterrnew']

    plt.figure(figsize=(12, 8))
    gs1 = gridspec.GridSpec(3, 1)
    gs1.update(right=0.48, left=0.05, bottom=0.55, hspace=0.10)
    ax1 = plt.subplot(gs1[0:2, 0])
    ax1.plot(df.teff, df.teffnew, 'o', alpha=0.7)
    ax1.errorbar([4200*1.05], [7100*0.95], xerr=tx, yerr=ty, fmt='-k')
    ax1.plot([4200, 7100], [4200, 7100], '--k', lw=3)
    ax1.set_xticklabels([])
    ax1.set_title('Teff')
    ax2 = plt.subplot(gs1[-1, 0])
    ax2.plot(df.teff, df.teffdiff, 'o', alpha=0.7)
    ax2.hlines([teffdiff], 4200, 7100)
    ax2.set_yticks([-400, -200, 0, 200, 400])
    ax2.set_ylim(-500, 500)
    ax2.grid(True)

    gs2 = gridspec.GridSpec(3, 1)
    gs2.update(right=0.48, left=0.05, top=0.45, hspace=0.05)
    ax3 = plt.subplot(gs2[0:2, 0])
    ax3.plot(df.logg, df.loggnew, 'o', alpha=0.7)
    ax3.errorbar([1.8*1.05], [5.1*0.95], xerr=gx, yerr=gy, fmt='-k')
    ax3.plot([1.8, 5.1], [1.8, 5.1], '--k', lw=3)
    ax3.set_xticklabels([])
    ax3.set_title('log g')
    ax4 = plt.subplot(gs2[-1, 0])
    ax4.plot(df.logg, df.loggdiff, 'o', alpha=0.7)
    ax4.hlines([loggdiff], 1.8, 5.1)
    ax4.set_yticks([-0.6, -0.3, 0.0, 0.3, 0.6])
    ax4.set_ylim(-0.9, 0.9)
    ax4.grid(True)

    gs3 = gridspec.GridSpec(3, 1)
    gs3.update(left=0.55, right=0.98, bottom=0.55, hspace=0.05)
    ax5 = plt.subplot(gs3[0:2, 0])
    ax5.plot(df.metal, df.fehnew, 'o', alpha=0.7)
    ax5.errorbar([-0.8*0.95], [0.55*0.65], xerr=fx, yerr=fy, fmt='-k')
    ax5.plot([-0.8, 0.55], [-0.8, 0.55], '--k', lw=3)
    ax5.set_xticklabels([])
    ax5.set_title('[Fe/H]')
    ax6 = plt.subplot(gs3[-1, 0])
    ax6.plot(df.metal, df.fehdiff, 'o', alpha=0.7)
    ax6.hlines([fehdiff], -0.8, 0.55)
    ax4.set_yticks([-0.6, -0.3, 0.0, 0.3, 0.6])
    ax6.set_ylim(-0.5, 0.5)
    ax6.grid(True)

    gs4 = gridspec.GridSpec(3, 1)
    gs4.update(left=0.55, right=0.98, top=0.45, hspace=0.05)
    ax7 = plt.subplot(gs4[0:2, 0])
    ax7.plot(df.vt, df.vtnew, 'o', alpha=0.7)
    ax7.errorbar([-0.1*0.95], [2.2*0.95], xerr=vx, yerr=vy, fmt='-k')
    ax7.plot([-0.1, 2.2], [-0.1, 2.2], '--k', lw=3)
    ax7.set_xticklabels([])
    ax7.set_title('Microturbulence')
    ax8 = plt.subplot(gs4[-1, 0])
    ax8.plot(df.vt, df.vtdiff, 'o', alpha=0.7)
    ax8.hlines([vtdiff], -0.1, 2.2)
    ax8.set_yticks([-0.8, -0.4, 0.0, 0.4, 0.8])
    ax8.set_ylim(-1.0, 1.0)
    ax8.grid(True)

    plt.savefig('update.pdf')

    # plt.show()
