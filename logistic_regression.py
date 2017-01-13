import pandas as pd
import numpy as np
from case_study import load_scrub_churn
from case_study import split_df
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from sklearn.preprocessing import StandardScaler

def sms_summary(df):
	y2 = df['churn']
	X2 = df[['trips_in_first_30_days', 'luxury_car_user', 'iphone', 'weekday_pct', "King's Landing", 'Winterfell']].astype(float)
	# X2 = df[['avg_dist', 'avg_rating_by_driver', 'avg_rating_of_driver', 'avg_surge', 'signup_date', 'surge_pct', 'trips_in_first_30_days', 'luxury_car_user', 'weekday_pct', 'iphone', "King's Landing", 'Winterfell']].astype(float)
	# X2 = df[['avg_dist', 'avg_rating_by_driver', 'avg_rating_of_driver', 'avg_surge', 'last_trip_date', 'signup_date', 'surge_pct', 'trips_in_first_30_days', 'luxury_car_user', 'weekday_pct', 'iphone', "King's Landing", 'Winterfell']].astype(float)
	X2 = sm.add_constant(X2)
	normalizer = StandardScaler(with_mean=False)
	X2 = normalizer.fit_transform(X2)
	logit = sm.Logit(y2, X2)
	results = logit.fit()
	summary = results.summary()
	return y2, X2, summary

def logistic_regression(X_train, X_test, y_train, y_test):
	model = LogisticRegression()
	model.fit(X_train, y_train)
	# predict class labels for the test set (Churn = True, Not Churn = False)
	predicted = model.predict(X_test)
	# generate class probabilities
	probs = model.predict_proba(X_test)
	return model, predicted, probs

def evaluation_metrics(y_test, predicted, probs):
	print 'Accuracy: ', metrics.accuracy_score(y_test, predicted)
	print 'Precision: ', metrics.precision_score(y_test, predicted)
	print 'Recall: ', metrics.recall_score(y_test, predicted)
	print 'Area under ROC: ', metrics.roc_auc_score(y_test, probs[:, 1])


def plot_roc(y_test, probs):
	probabilities = probs[:, 1]
	fpr, tpr, thresholds = metrics.roc_curve(y_test, probabilities)
	plt.plot(fpr, tpr, label='sklearn')
	plt.xlabel("False Positive Rate (1 - Specificity)")
	plt.ylabel("True Positive Rate (Sensitivity, Recall)")
	plt.title("ROC plot")
	plt.axis([-.1, 1.1, -.1, 1.1])

if __name__ == '__main__':
	df = load_scrub_churn()
	y2, X2, summary = sms_summary(df)
	print summary
	y2 = y2.values
	# X2 = X2.values
	# X_train, X_test, y_train, y_test = split_df(df)
	X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.3, random_state=42)
	model, predicted, probs = logistic_regression(X_train, X_test, y_train, y_test)
	print ''
	evaluation_metrics(y_test, predicted, probs)

	# examine the coefficients
	model.coef_
	print model.coef_
	# print ''
	# print 'Coefficient for each column:'
	# print pd.DataFrame(zip(df.columns, np.transpose(model.coef_)))

	# evaluate the model using 10-fold cross-validation
	scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=10)
	print ' 10-fold cross-validation scores: ' ,scores
	print 'Mean of scores: ', scores.mean()

	plot_roc(y_test, probs)
	plt.show()
