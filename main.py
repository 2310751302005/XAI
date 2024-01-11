import plotly.graph_objects as go         # To plot the candlestick
import pandas as pd                       # structures and data analysis
import datetime as dt                     #

import shap.datasets
import sklearn.preprocessing
import yfinance as yf                     # Yahoo! Finance market data downloader
import seaborn as sns
import scipy.stats as st
import mplfinance as mpf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#import os
import shap as sh

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

                                          # Introduce algorithms
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression, ElasticNet
from sklearn.svm import SVR               # Compared with SVC, it is the regression form of SVM.
                                          # Integrate algorithms
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

'''
The below code shows how to get data for FB from today
 to last 300 days.
'''
vdays = 1000
#dir_path = os.path.dirname(os.path.realpath(__file__))
#dir_path = os.getcwd()
#print(dir_path)

shap.initjs()
actual_date = dt.date.today()                            # Take the actual date
last_month_date = actual_date-dt.timedelta(days=vdays)
actual_date = actual_date.strftime("%Y-%m-%d")
last_month_date = last_month_date.strftime("%Y-%m-%d")
x, y = shap.datasets.california()
scalar = sklearn.preprocessing.StandardScaler()
scalar.fit(x)
x_std = scalar.transform(x)
'''
Stock data from https://finance.yahoo.com/quote/FB/news?ltr=1
'''
stock='AAPL'                                               # Stock name
data = yf.download(stock, last_month_date, actual_date)  # Getting data from Yahoo Finance
da= pd.DataFrame(data=data)
da.to_csv('file.csv')
df = pd.read_csv('file.csv')

#Stock price with interval of 5min
#data = yf.download(tickers=stock, start=dt.datetime(2021, 2,10), end=dt.datetime(2021, 2, 11), interval="5m")
mpf.plot(data,type='candle',mav=(3,6),volume=False,show_nontrading=True) # mav(3,6) moving average trend for 3 day and 6 day moving average price
x = df[['High', 'Low', 'Open', 'Volume']].values  # x features
y = df['Close'].values                            # y labels
sns.distplot(tuple(y), kde=False, fit=st.norm)
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=28) # Segment the data
ss = StandardScaler()                                 # Standardize the data set
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

#Set the model name.
names = ['LinerRegression',
       'Ridge',
       'Lasso',
       'Random Forrest',
       'Support Vector Regression',
       'ElasticNet',
       'XgBoost']

#Define the model.
# cv is the cross-validation idea here.
models = [LinearRegression(),
         RidgeCV(alphas=(0.001,0.1,1),cv=3),
         LassoCV(alphas=(0.001,0.1,1),cv=5),
         RandomForestRegressor(n_estimators=10),
         SVR(),
         ElasticNet(alpha=0.001,max_iter=10000),
         XGBRegressor()]
# Output the R2 scores of all regression models.

#Define the R2 scoring function.
def R2(name,model,x_train, x_test, y_train, y_test):
        model_fitted = model.fit(x_train,y_train)
        y_pred = model_fitted.predict(x_test)
        score = r2_score(y_test, y_pred)
        #########################
        ##Perform visualization.
        ln_x_test = range(len(x_test))
        y_predict = model.predict(x_test)
        # Set the canvas.
        plt.figure(figsize=(16, 8))
        # Draw with a red solid line.
        plt.plot(ln_x_test, y_test, 'm-o', lw=2, label=u'True values')
        # Draw with a green solid line.
        plt.plot(ln_x_test, y_predict, 'b--+', lw=3, label=u'Predicted value with the SVR algorithm,')
        # Display in a diagram.
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.title(u"Stock price prediction with "+name+"Score("+str(score.mean())+")")
        plt.ylabel('Price ($)')
        plt.show()
        #########################
        return score

#Traverse all models to score.
for name,model in zip(names,models):
        score = R2(name,model,x_train, x_test, y_train, y_test)
        print("{}: {:.6f}, {:.4f}".format(name,score.mean(),score.std()))
        explainer = shap.KernelExplainer(model.predict, x_test)
        shap_values = explainer.shap_values(x_test)
        #
        #shap.plots.violin( shap_values, features=X, feature_names=feat_names, plot_type="layered_violin")

        #
#Build a model.
'''
  'kernel': kernel function
  'C': SVR regularization factor
  'gamma': 'rbf', 'poly' and 'sigmoid' kernel function coefficient, which affects the model performance
'''
parameters = {
   'kernel': ['linear', 'rbf'],
   'C': [0.1, 0.5,0.9,1,5],
   'gamma': [0.001,0.01,0.1,1]
}

#Use grid search and perform cross validation.
model = GridSearchCV(SVR(), param_grid=parameters, cv=3)
model.fit(x_train, y_train)

##Obtain optimal parameters.
print("Optimal parameter list:", model.best_params_)
print("Optimal model:", model.best_estimator_)
print("Optimal R2 value:", model.best_score_)
#explainer = shap.KernelExplainer(model.predict,x_test)
#shap_values = explainer.shap_values(x_test)
#
#shap.force_plot(explainer.expected_value, shap_values[0, :], x_test.iloc[0, :])
#
###Perform visualization.
#ln_x_test = range(len(x_test))
#y_predict = model.predict(x_test)
##Set the canvas.
#plt.figure(figsize=(16,8))
##Draw with a red solid line.
#plt.plot (ln_x_test, y_test, 'm-o', lw=2, label=u'True values')
##Draw with a green solid line.
#plt.plot (ln_x_test, y_predict, 'b--+', lw = 3, label=u'Predicted value with the SVR algorithm,\
#=%.3f' % (model.best_score_))
##Display in a diagram.
#plt.legend(loc ='upper left')
#plt.grid(True)
#plt.title(u"Stock price prediction with SVR")
#plt.ylabel('Price ($)')
#plt.show()