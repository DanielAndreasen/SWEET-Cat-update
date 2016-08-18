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

"""Make histogram of [Fe/H] of all stars in SC, and only with homogenity=1, and
only with the new additions to see how it may shift"""


if __name__ == '__main__':
    names = ['star', 'hd', 'ra', 'dec', 'vmag', 'ervmag', 'par', 'erpar',
             'parsource', 'teff', 'erteff', 'logg', 'erlogg', 'logglc',
             'erlogglc', 'vt', 'ervt', 'metal', 'ermetal', 'mass', 'ermass',
             'author', 'link', 'source', 'update', 'comment', 'z1']

    new = pd.read_csv('SC.csv')
    new = new[new.convergence]
    new.linelist = new.linelist.str.strip()
    new.set_index('linelist', inplace=True)

    sc = pd.read_csv('WEBSITE_online.rdb', delimiter='\t', names=names, na_values=['NULL'])
    sc.star = sc.star.str.strip()
    sc.set_index('star', inplace=True)

    # Inset the new metallicities
    sc.loc[new.index, 'metal'] = new.feh
    sc.loc[new.index, 'source'] = 1
    sc.source = sc.source.astype(bool)

    sc.dropna(subset=['metal'], inplace=True)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax.set_xlabel('Metallicity, [Fe/H]')
    ax.set_ylabel('# of planet host stars')

    ax1.hist(sc.metal, label='Entire SC')
    ax1.hist(sc.metal[sc.source], label='SC with homogenity=1')
    ax1.hist(new.feh, label='New additions')
    ax1.set_title('All stars')
    ax1.set_yscale('log')

    # Only bright stars
    sc.dropna(subset=['vmag'], inplace=True)
    sc = sc[sc.vmag<12]
    ax2.hist(sc.metal, label='Entire SC')
    ax2.hist(sc.metal[sc.source], label='SC with homogenity=1')
    ax2.hist(new.feh, label='New additions')
    ax2.set_title('Stars brighter than 12 V magnitude')

    plt.tight_layout()
    plt.show()
    # TODO: I have to manually change the spacing, since tight_layout gives strange results.
    # plt.savefig('metallicityDistribution.pdf')
