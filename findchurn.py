import pandas as pd

from pydriller.metrics.process.code_churn import CodeChurn
from datetime import datetime
import sys


#variable definitions
churnperiod = 6


repo_path = '/home/codespace/grasprepo/'
find_file = "repomap.py"


totime = datetime.now()
month = totime.month - churnperiod
year = totime.year

# Adjust the year and month if necessary
if month <= 0:
    month += 12
    year -= 1

fromtime = totime.replace(year=year, month=month)

print(fromtime, totime) 
metric = CodeChurn(path_to_repo=repo_path, since=fromtime, to=totime)
#print(metric)
repo_truth = {}
files_count = metric.count()
#import sys; sys.exit(0)
files_max = metric.max()
files_avg = metric.avg()
#print('Total code churn for each file: {}'.format(files_count))
#print('Maximum code churn for each file: {}'.format(files_max))
#print('Average code churn for each file: {}'.format(files_avg))
from pydriller.metrics.process.commits_count import CommitsCount
metric = CommitsCount(path_to_repo=repo_path, since=fromtime, to=totime)
files = metric.count()
#print('Files: {}'.format(files))
from pydriller.metrics.process.contributors_count import ContributorsCount
metric = ContributorsCount(path_to_repo=repo_path, since=fromtime, to=totime)
count = metric.count()
minor = metric.count_minor()
#print('Number of contributors per file: {}'.format(count))
#print('Number of "minor" contributors per file: {}'.format(minor))
from pydriller.metrics.process.contributors_experience import ContributorsExperience
metric = ContributorsExperience(path_to_repo=repo_path, since=fromtime, to=totime)
contributor_exp = metric.count()
#print('Files: {}'.format(files))

from pydriller.metrics.process.lines_count import LinesCount
metric = LinesCount(path_to_repo=repo_path, since=fromtime, to=totime)
added_count = metric.count_added()
added_max = metric.max_added()
added_avg = metric.avg_added()
removed_count = metric.count_removed()
removed_max = metric.max_removed()
removed_avg = metric.avg_removed()

    
sortedby_commit = sorted(files.items(), key=lambda x:x[1], reverse=True)


import pprint
pp = pprint.PrettyPrinter(depth=4)


repo_truth = []
file_updated = "Files_updated_from_"+ fromtime.strftime("%m-%d-%Y") + "_to_" + totime.strftime("%m-%d-%Y")
i = 0
for element in sortedby_commit:
    file = element[0]
    try:
        newelement={'filename':file, 'commits' : files[file], 'current_UT_coverage(%)' : 0, 'Linting_issue' : 0, 'KLOC': 0, 'Duplication' : 0,
                            'Total_contributor' : count[file], 'Minor_contributor' : minor[file],
                            'Contributor_exp' : contributor_exp[file], 'Total_churn' : files_count[file], 'Max_churn' : files_max[file], 'Avg_churn' : files_avg[file],
                            'lines_added': added_count[file], 'max_lines_added': added_max[file],
                            'avg_lines_added_per_commit': added_avg[file], 'removed_lines':removed_count[file], 'max_lines_removed' : removed_max[file],
                            'avg_lines_removed_per_commit' : removed_avg[file] }
    except:
        print("move on")
    repo_truth.append(newelement)

    if len(repo_truth) > 20:
        break


#################print the result
print("Printing the result")
#print(repo_truth)
#pp.pprint(repo_truth)
#df = pd.DataFrame(data=repo_truth)

#convert into excel
#df.to_excel("gittruthmlpceTable.xlsx", index=False)
for element in repo_truth:
    if find_file in element['filename']:
        print(element)
        break

