
import operator
from pprint import pprint
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm
from scipy import stats
import matplotlib as mpl
from pandas import DataFrame
import matplotlib.pyplot as plt

__author__ = 'Dasha Herrmannova'
__email__ = 'dasha.herrmannova@open.ac.uk'


# SUPPORT FUNCTIONS ========================================================== #


def deposit_time_lag(
        deposited: datetime, published: datetime
) -> Dict[str, datetime]:
    """Calculate deposit time lag.
    
    :param deposited: [description]
    :type deposited: [type]
    :param published: [description]
    :type published: [type]
    :return: [description]
    :rtype: [type]
    """
    return {
        'difference_days': deposited - published
    }


def country_deposit_time_lag(df: DataFrame, desc: str) -> Dict[str, List[int]]:
    """[summary]
    
    :param df: [description]
    :type df: DataFrame
    :param desc: [description]
    :type desc: str
    :return: [description]
    :rtype: Dict[str, List[int]]
    """
    country_dtl = {}
    for row in tqdm(df.iterrows(), total=len(df), desc=desc, leave=False):
        country_diffs = {}
        for idx, country_code in enumerate(row[1]['core_country_code']):
            dtl = deposit_time_lag(
                row[1]['core_deposited_date'][idx], row[1]['cr_published']
            )
            if country_code not in country_diffs:
                country_diffs[country_code] = []
            country_diffs[country_code].append(dtl['difference_days'].days)
        for country_code, diffs in country_diffs.items():
            if  country_code not in country_dtl:
                country_dtl[country_code] = []
            country_dtl[country_code].append(min(diffs))        
    return country_dtl


def repository_deposit_time_lag(
        df: DataFrame, desc: str
) -> Dict[str, List[int]]:
    """[summary]
    
    :param df: [description]
    :type df: DataFrame
    :param desc: [description]
    :type desc: str
    :return: [description]
    :rtype: Dict[str, List[int]]
    """
    repository_dtl = {}
    for row in tqdm(df.iterrows(), total=len(df), desc=desc, leave=False):
        repository_diffs = {}
        for idx, id_repository in enumerate(row[1]['core_id_repository']):
            dtl = deposit_time_lag(
                row[1]['core_deposited_date'][idx], row[1]['cr_published']
            )
            # one paper may have been deposited into the same repository 
            # multiple times -- we will only use the first of the dates of 
            # deposit into the same repository
            if id_repository not in repository_diffs:
                repository_diffs[id_repository] = []
            repository_diffs[id_repository].append(dtl['difference_days'].days)
        for id_repository, diffs in repository_diffs.items():
            if id_repository not in repository_dtl:
                repository_dtl[id_repository] = []
            repository_dtl[id_repository].append(min(diffs))
    return repository_dtl


def repository_deposit_time_lag_agg(
        df: DataFrame, desc: str
) -> Dict[str, List[int]]:
    """[summary]
    
    :param df: [description]
    :type df: DataFrame
    :param desc: [description]
    :type desc: str
    :return: [description]
    :rtype: Dict[str, List[int]]
    """
    repository_dtl = {}
    for row in tqdm(df.iterrows(), total=len(df), desc=desc, leave=False):
        dtl = deposit_time_lag(
            sorted(row[1]['core_deposited_date'])[0], row[1]['cr_published']
        )
        repository_diffs = {}
        for idx, id_repository in enumerate(row[1]['core_id_repository']):
            repository_diffs[id_repository] = dtl['difference_days'].days
        for id_repository, diff in repository_diffs.items():
            if id_repository not in repository_dtl:
                repository_dtl[id_repository] = []
            repository_dtl[id_repository].append(diff)
    return repository_dtl


# MAIN PLOTS/STATS =========================================================== #


def overall_dtl(df: DataFrame, output_path: str) -> None:
    """Plot a distribution of deposit time lag across all data
    
    :param df: the input dataset
    :type df: DataFrame
    :param output_path: where to store the figure
    :type output_path: str
    :return: None
    :rtype: None
    """
    print('Overall DTL')
    diff_days_all = list(df['difference_days'].dt.days)
    num_bins = int((max(diff_days_all) - min(diff_days_all) + 1) / 30)
    plt.figure(figsize=(6, 3))
    plt.hist(diff_days_all, num_bins, edgecolor='none')
    ylim = plt.ylim()
    plt.plot((90, 90), (0, ylim[1]+100000), color='#DE3D49', linewidth=2)
    plt.yscale('log')
    plt.ylim((1, ylim[1]+100000))
    plt.xlim((-2600, 2350))
    plt.xlabel('Deposit time lag (days)', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x))
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

    print('\nDTL per year')
    plt.figure(figsize=(15, 2.5))
    for idx, year in enumerate(range(2013, 2018)):
        df_y = df[(df.cr_published.dt.year == year)]
        plt.subplot(1, 5, idx+1)
        plt.hist(
            list(df_y.difference_days.dt.days), 
            np.arange(-500, 2200, 30), 
            edgecolor='none'
        )
        plt.xlim((-500, 2200))
    plt.show()


def dtl_per_country(
        df: DataFrame, countries: List[str], output_path: str = None
) -> None:
    """Plot a distribution of deposit time lag per country
    
    :param df: the input dataset
    :type df: DataFrame
    :param output_path: where to store the figure
    :type output_path: str
    :param countries: for which countries to plot the distribution
    :type countries: List[str]
    :return: None
    :rtype: None
    """
    country_dtl = country_deposit_time_lag(df, 'All years')
    plt.figure(figsize=(6, 0.8 * len(countries)))
    plt.subplots_adjust(hspace=0.15)
    bins = range(-500, 2000, 7)
    for idx, c in enumerate(countries):
        ax = plt.subplot(int('{}{}{}'.format(len(countries), 1, idx + 1)))
        days_diff = country_dtl[c]
        plt.hist(days_diff, bins, edgecolor='none')
        ylim = plt.ylim()
        plt.plot((90, 90), (0, ylim[1]), color='#DE3D49')
        plt.gca().yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x))
        )
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        if c == 'us':
            plt.ylim((ylim[0], 2499))
        else:
            plt.ylim(ylim)
        plt.setp(ax.get_xticklabels(), visible=idx == len(countries) - 1)
        ax.text(
            2050, ylim[1]*0.60, c.upper(), fontsize=14, 
            horizontalalignment='right'
        )
    plt.gca().xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x))
    )
    plt.xlabel('Deposit time lag (days)', fontsize=14)
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    plt.show()


def dtl_per_country_year(
        df: DataFrame, countries: List[str], output_path: Optional[str] = None, 
        limit_dtl_years: Optional[int] = None
) -> None:
    """Plot average deposit time lag per country and year
    
    :param df: the input dataset
    :type df: DataFrame
    :param countries: for which countries to plot the distribution
    :type countries: List[str]
    :param output_path: where to store the figure, defaults to None
    :param output_path: Optional[str], optional
    :param limit_dtl_years: maximum limit to deposit time lag in years, 
                            defaults to None
    :param limit_dtl_years: Optional[int], optional
    :return: None
    :rtype: None
    """
    limit_dtl_years = limit_dtl_years if limit_dtl_years else 0
    max_year = 2019 - limit_dtl_years 

    # Calculate DTL of publications from each year and country
    country_year_dtl = {}
    for year in range(2013, max_year):
        f = df['cr_published'].dt.year == year
        if limit_dtl_years:
            f = (f) & (df['difference_days'].dt.days < (limit_dtl_years * 365))
        df_year = df[f]
        country_year_dtl[year] = country_deposit_time_lag(df_year, str(year))

    # aggregate plot
    print('\nDeposit time lag per country/year')
    plt.figure(figsize=(6, 3))
    years = sorted(country_year_dtl.keys())
    ind = np.arange(len(years))
    markers = ['^', 'D', 'o', 's', 'X']
    lines = ['--', '-.', '-', ':', '--']
    for idx, c in enumerate(countries):
        country_data = []
        for year in years:
            days_diff = country_year_dtl[year][c]
            country_data.append(np.mean(days_diff))
        print('Data for {}: {}'.format(c, [round(x) for x in country_data]))
        plt.plot(
            ind, country_data, label=c, marker=markers[idx%5], 
            linestyle=lines[idx%5], linewidth=2, markersize=5
        )
    plt.xticks(ind, years, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('Mean deposit time lag (days)', fontsize=14)
    plt.legend(
        bbox_to_anchor=(1.01, 0, 0, 0.99), loc=2, ncol=1, borderaxespad=0.,
        fontsize=14
    )
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    plt.show()


def dtl_average_change(df: DataFrame) -> None:
    """Calculate average change in deposit time lag between 2013 and 2017
    
    :param df: the input dataset
    :type df: DataFrame
    :return: None
    :rtype: None
    """
    # Calculate DTL of publications from each year and country
    country_year_dtl = {}
    for year in range(2013, 2019):
        df_year = df[df['cr_published'].dt.year == year]
        country_year_dtl[year] = country_deposit_time_lag(df_year, str(year))

    # calculate average change in DTL between 2013 and 2017
    all_countries = set.intersection(*
        [set(country_year_dtl[y].keys()) for y in [2013, 2017]]
    )
    ranges = []
    for c in all_countries:
        val_2013 = np.mean(country_year_dtl[2013][c])
        val_2017 = np.mean(country_year_dtl[2017][c])
        ranges.append(val_2013 - val_2017)
    ranges_stats = stats.describe(ranges)
    print('Min&max decrease in average DTL between 2013 & 2017: {}'.format(
        ranges_stats.minmax
    ))
    print('Average decrease in DTL between 2013 and 2017: {:.2f}'.format(
        ranges_stats.mean
    ))
    dtl_2017 = [np.mean(v) for v in country_year_dtl[2017].values()]
    dtl_2018 = [np.mean(v) for v in country_year_dtl[2018].values()]
    print('Average DTL in 2017: {:.2f}'.format(np.mean(dtl_2017)))
    print('Average DTL in 2018: {:.2f}'.format(np.mean(dtl_2018)))


def uk_compliance_per_year(df: DataFrame, output_path: str) -> None:
    """Plot showing the proportion of likely compliant and non-compliant
    publications in the UK per year
    
    :param df: the input dataset
    :type df: DataFrame
    :param output_path: where to store the figure
    :type output_path: str
    :return: None
    :rtype: None
    """
    df_gb = df[df.is_gb & df.has_issn]
    print('UK DTL statistics:\n')
    print(df_gb.difference_days.describe())

    print('\nCalculating compliance per year')

    compliant = []
    non_compliant = []
    total = []

    for year in tqdm(range(2013, 2019)):
        df_year = df_gb[df_gb['cr_published'].dt.year == year]
        days_diff_year = list(df_year['difference_days'].dt.days)
        compliant.append(sum([1 for x in days_diff_year if x <= 90]))
        non_compliant.append(sum([1 for x in days_diff_year if x > 90]))
        total.append(len(days_diff_year))
        
    compliant = np.array(compliant, dtype=np.float)
    non_compliant = np.array(non_compliant, dtype=np.float)
    total = np.array(total, dtype=np.float)

    print('\nCompliance per year:')

    ind = np.arange(len(compliant))
    plt.figure(figsize=(8, 3))
    plt.bar(
        ind, compliant / total, color='#4568aa', label='Potentially compliant'
    )
    plt.bar(
        ind, non_compliant / total, bottom=compliant / total, 
        label='Non-compliant', color='#d87c4a'
    )
    plt.yticks(fontsize=14)
    plt.xticks(ind, np.arange(2013, 2019), fontsize=14)
    plt.ylabel('Proportion of publications', fontsize=14)
    plt.legend(
        fontsize=14, bbox_to_anchor=(1.01, 0, 0, 0.99), loc=2, ncol=1, 
        borderaxespad=0.
    )
    plt.gca().yaxis.grid(True)
    plt.gca().xaxis.grid(False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def repository_dtl(df: DataFrame, output_path: str) -> None:
    """Plot deposit time lag per repository and year
    
    :param df: the input dataset
    :type df: DataFrame
    :param output_path: where to store the figure
    :type output_path: str
    :return: None
    :rtype: None
    """
    repository_year_dtl = {}
    repository_year_dtl_agg = {}
    for year in range(2013, 2019):
        df_year = df[df['cr_published'].dt.year == year]
        repository_year_dtl[year] = (
            repository_deposit_time_lag(df_year, str(year))
        )
        repository_year_dtl_agg[year] = (
            repository_deposit_time_lag_agg(df_year, str(year))
        )

    plt.figure(figsize=(6.5, 3))

    colors = {}
    for year in sorted(repository_year_dtl.keys())[:-1]:
        repo_data = []
        repo_data_agg = []
        for r in repository_year_dtl[year]:
            if len(repository_year_dtl[year][r]) < 100:
                continue
            repo_data.append(np.mean(repository_year_dtl[year][r]))
            repo_data_agg.append(np.mean(repository_year_dtl_agg[year][r]))
        repo_data = np.array(repo_data)
        repo_data_agg = np.array(repo_data_agg)
        ind = np.arange(len(repo_data))
        repo_data = np.sort(repo_data)
        repo_data_agg = np.sort(repo_data_agg)
        print('{} -- range ({}, {}), std: {}'.format(
            year, min(repo_data), max(repo_data), np.std(repo_data)
        ))
        colors[year] = plt.plot(
            ind, repo_data, label=year, linestyle='-'
        )[0].get_color()
        plt.plot(
            ind, repo_data_agg, color=colors[year], label=year, linestyle='--'
        )
        
    lines, labels = zip(*[
        (mpl.lines.Line2D([0], [0], color=v, lw=2), k) 
        for k, v in sorted(colors.items())
    ])

    plt.legend(
        lines, labels, 
        fontsize=14, bbox_to_anchor=(1.01, 0, 0, 0.99), loc=2, ncol=1, 
        borderaxespad=0.
    )

    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, pos: '{}'.format(x))
    )

    ylim = plt.ylim()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim((ylim[0], 2100))
    plt.xlabel('Repositories', fontsize=14)
    plt.ylabel('Average deposit time lag', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def repository_compliance(df: DataFrame, output_path: str) -> None:
    """Plot compliance per repository and year for UK publications
    
    :param df: the input dataset
    :type df: DataFrame
    :param output_path: where to store the figure
    :type output_path: str
    :return: None
    :rtype: None
    """
    repository_year_compliance = {}
    repository_year_compliance_agg = {}
    for year in range(2013, 2019):
        df_year = df[
            (df.is_gb) & (df.has_issn) & (df['cr_published'].dt.year == year)
        ]
        repo_dtl = repository_deposit_time_lag(df_year, str(year))
        repo_dtl_agg = repository_deposit_time_lag_agg(df_year, str(year))

        compliance_single = {
            k: [x <= 90 for x in v] for k, v in repo_dtl.items()
        }   
        compliance_agg = {
            k: [x <= 90 for x in v] for k, v in repo_dtl_agg.items()
        }
        
        repository_year_compliance[year] = {
            k: float(sum(v)) / float(len(v)) 
            for k, v in compliance_single.items() if len(v) >= 100
        }
        repository_year_compliance_agg[year] = {
            k: float(sum(v)) / float(len(v)) 
            for k, v in compliance_agg.items() if len(v) >= 100
        }

    plt.figure(figsize=(6.5, 3))

    colors = {}
    for year in sorted(repository_year_compliance.keys())[:-1]:
        repo_data = []
        repo_data_agg = []
        for r in repository_year_compliance[year]:
            repo_data.append(repository_year_compliance[year][r])
            repo_data_agg.append(repository_year_compliance_agg[year][r])
        repo_data = np.array(repo_data)
        repo_data_agg = np.array(repo_data_agg)
        ind = np.arange(len(repo_data))
        repo_data = np.sort(repo_data)[::-1]
        repo_data_agg = np.sort(repo_data_agg)[::-1]
        colors[year] = plt.plot(
            ind, repo_data, label=year, linestyle='-'
        )[0].get_color()
        plt.plot(
            ind, repo_data_agg, color=colors[year], label=year, linestyle='--'
        )
        
    lines, labels = zip(*[
        (mpl.lines.Line2D([0], [0], color=v, lw=2), k) 
        for k, v in sorted(colors.items())
    ])

    plt.legend(
        lines, labels, 
        fontsize=14, bbox_to_anchor=(1.01, 0, 0, 0.99), loc=2, ncol=1, 
        borderaxespad=0.
    )

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Repositories', fontsize=14)
    plt.ylabel('Proportion of compliant', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def repository_dtl_detailed(df: DataFrame, year: int) -> DataFrame:
    """Calculate deposit time lag and number of publications per repository
    and return as a table
    
    :param df: the input dataset
    :type df: DataFrame
    :param output_path: where to store the figure
    :type output_path: str
    :return: table containing deposit time lag and number of publications per 
             repository
    :rtype: DataFrame
    """
    df_year = df[df['cr_published'].dt.year == year]
    repo_dtl = repository_deposit_time_lag(df_year, str(year))
    id_name_map = {
        row[1]['core_id_repository'][idx]: row[1]['core_repository_name'][idx]
        for row in tqdm(df.iterrows(), total=len(df)) 
        for idx in range(len(row[1]['core_id_document']))
    }
    s = sorted(
        [
            (k, id_name_map[k], np.mean(v), len(v)) 
            for k, v in repo_dtl.items() if len(v) >= 1000
        ], 
        key=operator.itemgetter(2)
    )
    columns=['id', 'name', 'average DTL', '#publications']
    dtl_df = DataFrame(s, columns=columns)
    dtl_df.set_index('id', inplace=True)
    return dtl_df


def subject_dtl(df: DataFrame, years: List[int], output_path: str) -> None:
    """Plot average deposit time lag per subject and year
    
    :param df: the input dataset
    :type df: DataFrame
    :param years: which years to compare
    :type years: List[int]
    :param output_path: where to store the figure
    :type output_path: str
    :return: None
    :rtype: None
    """
    plot_data = {}
    for year in years:
        df_year = df[df['cr_published'].dt.year == year]    
        plot_data[year] = {}
        for row in tqdm(df_year.iterrows(), total=len(df_year), desc=str(year)):
            if not len(row[1]['subjects']):
                continue
            for k in row[1]['subjects']:
                if k not in plot_data[year]:
                    plot_data[year][k] = []
                plot_data[year][k].append(row[1]['difference_days'].days)

    plt.figure(figsize=(7, 6))

    labels = []
    colors = ['#d87c4a', '#4568aa']

    for idx, year in enumerate(sorted(plot_data.keys())):
        
        dtl = []
                    
        if idx == 0:
            for k, v in plot_data[year].items():
                skip = False
                for skip_year in plot_data.keys():
                    if len(plot_data[skip_year][k]) < 100:
                        print('Removing {} (less than 100 values)'.format(k))
                        skip = True
                if skip:
                    continue
                dtl.append(np.mean(v))
                labels.append(k)
            dtl = np.array(dtl, dtype=np.float)
            print('\nStatistics for {}'.format(year))
            pprint(stats.describe(dtl)._asdict())
            print('Std: {}'.format(np.std(dtl)))
            labels = np.array(labels)
            sort_indices = np.argsort(dtl)
            dtl = dtl[sort_indices]
            labels = labels[sort_indices]
        else:
            for k in labels:
                dtl.append(np.mean(plot_data[year][k]))
            print('\nStatistics for {}'.format(year))
            pprint(stats.describe(dtl)._asdict())
            print('Std: {}'.format(np.std(dtl)))

        ind = np.arange(len(dtl))
        plt.barh(ind, dtl, color=colors[idx], label=year)

    print('\nDTL per subject')

    plt.yticks(ind, labels)
    plt.ylim((-0.5, len(labels) - 0.5))
    plt.xlabel('Average deposit time lag')
    plt.legend(
        bbox_to_anchor=(0., 1.002, 1., .102), loc=3, ncol=2, borderaxespad=0.
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def ref_panel_compliance(
        df: DataFrame, years: List[int], output_path: str
) -> None:
    """Plot average deposit time lag per REF panel and year
    
    :param df: the input dataset
    :type df: DataFrame
    :param years: which years to compare
    :type years: List[int]
    :param output_path: where to store the figure
    :type output_path: str
    :return: None
    :rtype: None
    """
    panel_compliance = {}
    for year in years:
        df_year = df[
            (df.is_gb) & (df['cr_published'].dt.year == year) & df.has_issn
        ]    
        panel_compliance[year] = {}
        for row in df_year.iterrows():
            for p in row[1]['panels']:
                if p not in panel_compliance[year]:
                    panel_compliance[year][p] = []
                panel_compliance[year][p].append(
                    row[1]['difference_days'].days <= 90
                )

    plt.figure(figsize=(5.5, 2))

    labels = []
    previous = None

    for idx, year in enumerate(sorted(panel_compliance.keys())):
        
        compliant = []
        non_compliant = []
        total = []

        if idx == 0:
            for k, v in panel_compliance[year].items():
                compliant.append(sum(v))
                non_compliant.append(len(v) - sum(v))
                total.append(len(v))
                labels.append(k)

            compliant = np.array(compliant, dtype=np.float)
            non_compliant = np.array(non_compliant, dtype=np.float)
            total = np.array(total, dtype=np.float)
            labels = np.array(labels)

            sort_indices = np.argsort(labels)[::-1]
            compliant = compliant[sort_indices]
            non_compliant = non_compliant[sort_indices]
            total = total[sort_indices]
            labels = labels[sort_indices]
            
            previous = compliant / total
            
        else:
            for k in labels:
                v = panel_compliance[year][k]
                compliant.append(sum(v))
                non_compliant.append(len(v) - sum(v))
                total.append(len(v))
            compliant = np.array(compliant, dtype=np.float)
            non_compliant = np.array(non_compliant, dtype=np.float)
            total = np.array(total, dtype=np.float)
            labels = np.array(labels)

        ind = np.arange(len(compliant))
        
        if idx == 0:
            plt.barh(
                ind, non_compliant / total, left=compliant / total, 
                label='Non-compliant', color='#d87c4a'
            )
        else:
            plt.barh(
                ind, compliant / total, hatch='//', fill=False,
                label='Likely compliant: {}'.format(year)
            )
            plt.barh(
                ind, previous, color='#4568aa', 
                label='Likely compliant: {}'.format(
                    sorted(panel_compliance.keys())[0]
                )
            )

    plt.xticks(fontsize=14)
    plt.yticks(ind, labels, fontsize=14)
    plt.xlabel('Proportion of all publications', fontsize=14)
    plt.legend(
        bbox_to_anchor=(1.03, 0, 0, 0.99), loc=2, ncol=1, borderaxespad=0.,
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
