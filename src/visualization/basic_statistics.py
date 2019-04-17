
import json
import operator
from pprint import pprint
from collections import Counter

import numpy as np
from tqdm import tqdm
from scipy import stats
from pandas import DataFrame
import matplotlib.pyplot as plt

__author__ = 'Dasha Herrmannova'
__email__ = 'dasha.herrmannova@open.ac.uk'


def country_distribution(df: DataFrame, top_x: int, output_path: str) -> None:
    """Plot country distribution for 'top_x' countries.
    
    :param df: the input dataset
    :type df: DataFrame
    :param top_x: how many largest countries to include in the figure
    :type top_x: int
    :param output_path: where to store the figure
    :type output_path: str
    :return: None
    :rtype: None
    """
    country_codes = []
    for row in tqdm(df.core_country_code, total=len(df)):
        country_codes += list(set(row))
    countries, counts = zip(*Counter(country_codes).most_common()[0:top_x])
    countries = [x if x else 'n/a' for x in countries]

    plt.figure(figsize=(6, 2.5))
    ind = np.arange(len(countries))
    plt.bar(ind, counts)
    plt.xticks(ind, countries, fontsize=14, rotation=45)
    plt.yticks(fontsize=14)
    plt.xlim((-0.75, top_x - 0.25))
    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x))
    )
    plt.ylabel('Number of papers', fontsize=14)
    plt.tight_layout()
    plt.gca().yaxis.grid(True)
    plt.gca().xaxis.grid(False)
    plt.savefig(output_path)
    plt.show()


def year_of_publication_distribution(df: DataFrame, output_path: str) -> None:
    """Plot publication year distribution
    
    :param df: the input dataset
    :type df: DataFrame
    :param output_path: where to store the figure
    :type output_path: str
    :return: None
    :rtype: None
    """
    year_filter = df['cr_published'].dt.year <= 2018
    gb = list(df[(year_filter) & (df['is_gb'])]['cr_published'].dt.year)
    nongb = list(df[(year_filter) & (~df['is_gb'])]['cr_published'].dt.year)
    years, counts_gb = zip(*sorted(Counter(gb).items()))
    _, counts_nongb = zip(*sorted(Counter(nongb).items()))
    
    plt.figure(figsize=(6, 2))
    ind = np.arange(len(years))
    plt.bar(ind, counts_gb, label='UK')
    plt.bar(ind, counts_nongb, bottom=counts_gb, label='RoW', color='#33CB43')
    plt.xticks(ind, years, fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x))
    )
    plt.ylabel('Number of papers', fontsize=14)
    plt.legend(
        fontsize=14, bbox_to_anchor=(1.01, 0, 0, 0.99), loc=2, ncol=1, 
        borderaxespad=0.
    )
    plt.gca().yaxis.grid(True)
    plt.gca().xaxis.grid(False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def repository_stats(df: DataFrame) -> DataFrame:
    """Calculate basic repository statistics, mainly:
    - how many publications per repository
    - how many repositories per publication
    - largest repositories
    
    :param df: the input dataset
    :type df: DataFrame
    :return: DataFrame with 10 largest repositories
    :rtype: DataFrame
    """
    id_name_map = {
        row[1]['core_id_repository'][idx]: row[1]['core_repository_name'][idx]
        for row in tqdm(df.iterrows(), total=len(df)) 
        for idx in range(len(row[1]['core_id_document']))
    }
    affs_per_paper = [
        list(set(row)) for row in tqdm(df.core_id_repository, total=len(df))
    ]
    flat = [item for sublist in affs_per_paper for item in sublist]
    print('Number of repositories: {}'.format(len(set(flat))))
    affs, counts = zip(*Counter(flat).most_common())
    aff_count_per_paper = [len(l) for l in affs_per_paper]

    print('\nAffiliations per paper:')
    pprint(stats.describe(aff_count_per_paper)._asdict())

    print('\nPapers per affiliation:')
    pprint(stats.describe(counts)._asdict())

    print('\nNumber of repositories with less than 100 publications: {}'.format(
        sum(x < 100 for x in counts)
    ))
    print('Number of repositories with less than 50 publications: {}'.format(
        sum(x < 50 for x in counts)
    ))

    print('\n10 largest repositories')
    return DataFrame(
        [(counts[x], id_name_map[affs[x]], affs[x]) for x in range(10)], 
        columns=['Papers', 'Name', 'ID']
    )


def subject_distribution(df: DataFrame, output_path: str) -> DataFrame:
    """Plot a distribution of publication subjects
    
    :param df: the input dataset
    :type df: DataFrame
    :param output_path: where to store the figure
    :type output_path: str
    :return: table showing how many multi-disciplinary papers there are
    :rtype: DataFrame
    """
    total = len(df)
    with_subject = sum(len(s) > 0 for s in df.subjects)
    print('{} out of {} papers ({}%) have subject information'.format(
        with_subject, total, float(with_subject) / float(total) * 100
    ))

    subject_count_per_paper = [len(s) for s in df.subjects]
    multiple_subjects = sum(x > 1 for x in subject_count_per_paper)
    print('{} documents were tagged with multiple subjects ({}%)'.format(
        multiple_subjects, 
        float(multiple_subjects) / float(len(subject_count_per_paper)) * 100
    ))

    print('\nSubject distribution:')
    subjects_list = [
        (s, 1.0 / float(len(row))) for row in df.subjects for s in row
    ]
    subjects_totals = {}
    for s, v in subjects_list:
        if s not in subjects_totals:
            subjects_totals[s] = 0
        subjects_totals[s] += v

    labels, values = zip(
        *sorted(subjects_totals.items(), key=operator.itemgetter(1), 
        reverse=True)
    )
    ind = np.arange(len(labels))
    plt.figure(figsize=(7.5, 6.5))
    plt.barh(ind, values[::-1])
    plt.yticks(ind, labels[::-1])
    plt.gca().xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x / 1000))
    )
    plt.xlabel('Number of publications (thousands)')
    plt.ylim(-0.5, len(labels) - 0.5)
    plt.gca().xaxis.grid(True)
    plt.gca().yaxis.grid(False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

    # number of papers with multiple subjects
    print('\nNumber of papers per subject count:')
    paper_count_per_subj_count = [
        (s, c, float(c) / float(multiple_subjects)) 
        for s, c in Counter(subject_count_per_paper).most_common()[2:]
    ]
    return DataFrame(
        paper_count_per_subj_count, 
        columns=['subject_count', 'paper_count', '% of multi-subj. papers']
    )


def ref_uoa_distribution(df: DataFrame, output_path: str) -> None:
    """Plot a distribution of REF main panels
    
    :param df: the input dataset
    :type df: DataFrame
    :param output_path: where to store the figure
    :type output_path: str
    :return: None
    :rtype: None
    """
    panel_list = [
        (panel, 1.0 / float(len(row))) 
        for row in df[df.is_gb].panels 
        for panel in row 
        if len(row)
    ]
    panels_totals = {}
    for panel, v in panel_list:
        if panel not in panels_totals:
            panels_totals[panel] = 0
        panels_totals[panel] += v

    labels, values = zip(*sorted(panels_totals.items()))
    ind = np.arange(len(labels))
    plt.figure(figsize=(4, 2))
    plt.bar(ind, values)
    plt.xticks(ind, labels, fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x))
    )
    plt.xlim(-0.5, len(labels) - 0.5)
    plt.ylabel('# of papers', fontsize=14)
    plt.xlabel('REF 2021 panel', fontsize=14)
    plt.gca().yaxis.grid(True)
    plt.gca().xaxis.grid(False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def crossref_acceptance_date_stats(df: DataFrame) -> DataFrame:
    """Calculate basic statistics of Crossref accepted date field, mainly:
    - how many publications have this date set
    - how does the date compare to published date
    
    :param df: the input dataset
    :type df: DataFrame
    :return: table with dates of acceptance and publication for each paper
             in the dataset
    :rtype: DataFrame
    """
    cr_accepted_data = []
    for row in tqdm(df.iterrows(), total=len(df)):
        cr_accepted_data.append({
            'accepted': (
                row[1]['cr_accepted'] if row[1]['cr_accepted'] else None
            ), 
            'published': row[1]['cr_published']
        })

    cr_accepted_data = DataFrame(cr_accepted_data)

    print('Non-empty date of acceptance: {}'.format(
        sum(cr_accepted_data.accepted.notnull())
    ))
    print('Date of acceptace == date of publication: {}'.format(
        sum(cr_accepted_data.accepted == cr_accepted_data.published)
    ))
    print('Date of acceptace > date of publication: {}'.format(
        sum(cr_accepted_data.accepted > cr_accepted_data.published)
    ))
    
    return cr_accepted_data


def issn_stats(df: DataFrame) -> None:
    """Calculate basic ISSN statistics, mainly:
    - how many publications have an ISSN number set in Crossref overall
    - how many UK publications have na ISSN number
    
    :param df: the input dataset
    :type df: DataFrame
    :return: None
    :rtype: None
    """
    print('Publications without ISSN: {}'.format(sum(~df.has_issn)))
    print('UK publications without ISSN: {}'.format(
        sum(~df.has_issn & df.is_gb)
    ))


def core_crossref_matching_accuracy(df: DataFrame) -> None:
    """Analysis of the accuracy of our method of matching CORE and Crossref data
    
    :param df: the input dataset
    :type df: DataFrame
    :return: None
    :rtype: None
    """
    same = 0
    substring = 0
    different = 0
    missing = 0

    for row in tqdm(df.iterrows(), total=len(df)):
        cr_doi = row[1]['cr_doi'].lower()
        for doi in row[1]['core_doi']:
            if not doi or doi == '\\N':
                missing += 1
            else:
                core_doi = doi.rstrip('.').lower()
                if core_doi == cr_doi:
                    same += 1
                else:
                    different += 1
                    if cr_doi in core_doi:
                        substring += 1

    total = same + different + missing
    print('There are {} links between CrossRef and CORE'.format(total))
    print('{} CORE articles didn\'t have a DOI ({:.2f}%)'.format(
        missing, float(missing) / float(total) * 100
    ))
    print('Of the remaining {} CORE articles:'.format(total - missing))
    print('{} had a matching DOI ({:.2f}%)'.format(
        same, float(same) / float(total - missing) * 100
    ))
    print('{} had a different DOI ({:.2f}%)'.format(
        different, float(different) / float(total - missing) * 100
    ))
    print('{} non-matching DOIs were matched as substring ({:.2f}%)'.format(
        substring, float(substring) / float(different) * 100
    ))
    print('Same + substring DOIs: {} ({:.2f}%)'.format(
        same + substring, float(same + substring) / float(total - missing) * 100
    ))
