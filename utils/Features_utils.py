'''
Contains a set of utilities used in the project to generate features
'''
import pandas as pd
from yellowbrick.model_selection import FeatureImportances
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from yellowbrick.features import RFECV

def featureImportance(X,y,model=GradientBoostingClassifier()):
	'''
	Possible models for coefficient importance:
	from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
	from sklearn.svm import SVC, NuSVC, LinearSVC, SVR, NuSVR, LinearSVR
	from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostRegressor, RandomForestRegressor,GradientBoostingRegressor
	'''
	viz = FeatureImportances(model, size=(1080, 720))
	viz.fit(X, y)
	viz.poof()