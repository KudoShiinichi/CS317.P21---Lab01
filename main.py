from metaflow import FlowSpec, step, card, environment, Parameter, current, resources
import pandas as pd
import os
from datetime import datetime

# Sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score

# Sklearn models
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
# from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
# from sklearn.ensemble import VotingClassifier


# Sklearn hyperopt
from hpsklearn import *
# from hyperopt import hp, STATUS_OK, Trials, fmin
from hyperopt import  tpe

# MLflow
import mlflow
from mlflow.models import infer_signature
from mlflow.data.pandas_dataset import PandasDataset

# For drawing plot
import matplotlib.pyplot as plt
plt.style.use("ggplot")  #using style ggplot
from metaflow.cards import VegaChart
from metaflow.cards import Markdown
