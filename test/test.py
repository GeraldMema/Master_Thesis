from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sb

from pmlb import fetch_data, classification_dataset_names

logit_test_scores = []
gnb_test_scores = []

possible_datasets = [
'adult',
'allhypo',
'analcatdata_authorship',
'analcatdata_japansolvent',
'appendicitis',
'crx',
'ionosphere',
'GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_75_EDM_2_001',
'credit_g',
'confidence',
'colic',
'churn',
'car',
'bupa',
'biomed',
'mfeat_fourier'
]

for classification_dataset in classification_dataset_names:
    if classification_dataset not in possible_datasets:
        continue
    print(classification_dataset)

    X, y = fetch_data(classification_dataset, return_X_y=True)
    print('shape: ', X.shape)

    # continue
    X = X[:5000]
    y = y[:5000]

    train_X, test_X, train_y, test_y = train_test_split(X, y)

    logit = LogisticRegression()
    gnb = GaussianNB()

    try:
        logit.fit(train_X, train_y)
        gnb.fit(train_X, train_y)

        score = logit.score(test_X, test_y)
        logit_test_scores.append(score)
        gnb_test_scores.append(gnb.score(test_X, test_y))
    except:
        print('ERROR')
        continue

    print(score)
    print()



# sb.boxplot(data=[logit_test_scores, gnb_test_scores], notch=True)
# plt.xticks([0, 1], ['LogisticRegression', 'GaussianNB'])
# plt.ylabel('Test Accuracy')
# plt.show()