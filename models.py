import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from dateutil.parser import parse
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from mlxtend.regressor import StackingCVRegressor
from lightgbm import log_evaluation, early_stopping
import matplotlib.pyplot as plt


# define the error evaluation method for LightGBM model, which is MSE
def evalerror(pred, df):
    # calculate MSE
    label = df.get_label().copy()
    score = mean_squared_error(label,pred)*0.5
    return ('0.5mse',score,False)

# Process features for train set and test set, including missing value imputation,
# abnormal value imputation, categorial data transformation
# Parameters:
#   train: the whole training set dataframe
#   test: the whole testing set dataframe
# return a training set and a test set which are ready to be splitted into X and y
def data_processing(train,test):
    train_id = train.id.values.copy()
    test_id = test.id.values.copy()
    data = pd.concat([train,test])

    # convert Gender to numeric feature
    data['Gender'] = data['Gender'].map({'M':1,'F':0})
 
    data.fillna(data.median(axis=0),inplace=True)       # fill nan with median
    data.drop('Date', axis=1, inplace=True)             # drop date

    # HBsAg, HBsAb, HBeAg, HBeAb, HBcAb have a missing ratio of 0.769347, hence drop them
    data.drop('HBsAg', axis=1, inplace=True)
    data.drop('HBsAb', axis=1, inplace=True)
    data.drop('HBeAg', axis=1, inplace=True)
    data.drop('HBeAb', axis=1, inplace=True)
    data.drop('HBcAb', axis=1, inplace=True)

    # # abnormal value imputation
    # for index, row in data.iterrows():
    #     if row['*AST'] > 300:
    #         row['*AST'] = data['*AST'].mean()
    #     if row['*ALT'] > 300:
    #         row['*ALT'] = data['*ALT'].mean()
    #     if row['*ALP'] > 300:
    #         row['*ALP'] = data['*ALP'].mean()

    train_processed = data[data.id.isin(train_id)]          # split training set and test set
    test_processed = data[data.id.isin(test_id)]
    train_processed.drop('id', axis=1, inplace=True)        # drop id
    test_processed.drop('id', axis=1, inplace=True) 

    return train_processed,test_processed

# build a LightGBM model and select features according to its feature importance ranking
# Parameters:
#   X_train: dataframe, the features of training set
#   y_train: dataframe, the labels of training set
#   k: number of features that want to keep
# return the names of top k important features
def feature_selection(X_train, y_train, k):
    params = {
        'learning_rate': 0.01,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mse',
        'num_leaves': 60,
        'feature_fraction': 0.7,
        'min_data': 100,
        'min_hessian': 1,
        'verbose': -1,
    }

    kf = KFold(n_splits = 5, shuffle=True, random_state=520)
    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        lgb_train1 = lgb.Dataset(X_train.iloc[train_index],y_train.iloc[train_index],categorical_feature=['Gender'])
        lgb_train2 = lgb.Dataset(X_train.iloc[test_index], y_train.iloc[test_index])

        gbm = lgb.train(params,
                        lgb_train1,
                        num_boost_round=3000,
                        valid_sets=lgb_train2,
                        feval=evalerror,
                        callbacks=[log_evaluation(period=100), early_stopping(stopping_rounds=100)],
                        categorical_feature=['Gender'])
        feat_imp = pd.Series(gbm.feature_importance(), index=X_train.columns.tolist()).sort_values(ascending=False)
        # print(feat_imp)
        print(feat_imp[:k])       # output the importance of top k important features

    return(feat_imp.index[:k])

# load the train and test data, return X_train, y_train, X_test, y_test which are ready for training
def load_data(data_path):
    # read file
    train = pd.read_csv(data_path+'train.csv')
    test = pd.read_csv(data_path+'test.csv')
    y_test = (pd.read_csv(data_path+'answer.csv')['label']).tolist()

    # data processing
    train_processed,test_processed = data_processing(train,test)

    # split into X and y
    predictors = [f for f in test_processed.columns if f not in ['Glu']]     # 就是除了血糖以外的列名，表示参与预测的特征？
    X_train = train_processed[predictors]
    y_train = train_processed['Glu']
    X_test = test_processed[predictors]

    # feature selection (keep top 18 important features)
    features = feature_selection(X_train, y_train,18).tolist()
    if 'Gender' not in features:        # keep the categorial feature
        features.append('Gender')
    X_train =  X_train[features]
    X_test =  X_test[features]

    return X_train, y_train, X_test, y_test

# train LightGBM model
def lgb_train(X_train, y_train, X_test, y_test):
    print('lgb model training...')

    # parameter tuning process using grid search
    # grid = [0.65,0.66,0.67,0.68,0.69,0.7,0.71,0.72,0.73,0.74,0.75]
    grid = [1]          # now the tuning is over so only store one useless value
    errors = []

    for x in grid:      # x is every possible parameter value, now not be used because the tuning is already finished
        params = {
            'learning_rate': 0.155,
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'mse',
            'num_leaves': 40,
            'feature_fraction': 0.66,
            'min_data': 7,      # min data per leaf
            'min_hessian': 1,   # min sum hessian of a leaf
            'verbose': -1,
            'max_depth':16,     # max depth of trees, can prevent overfitting to some extent when dataset is small
        }

        print('Cross validation 5-fold training...')
        t0 = time.time()
        train_preds = np.zeros(X_train.shape[0])
        test_preds = np.zeros((X_test.shape[0], 5))
        kf = KFold(n_splits = 5, shuffle=True, random_state=520)
        for i, (train_index, test_index) in enumerate(kf.split(X_train)):
            print('Round {}...'.format(i))
            lgb_train1 = lgb.Dataset(X_train.iloc[train_index],y_train.iloc[train_index],categorical_feature=['Gender'])
            lgb_train2 = lgb.Dataset(X_train.iloc[test_index], y_train.iloc[test_index])

            gbm = lgb.train(params,
                            lgb_train1,
                            num_boost_round=3000,
                            valid_sets=lgb_train2,
                            feval=evalerror,
                            callbacks=[log_evaluation(period=100), early_stopping(stopping_rounds=100)],
                            categorical_feature=['Gender'])
            train_preds[test_index] += gbm.predict(X_train.iloc[test_index])
            test_preds[:,i] = gbm.predict(X_test)

        score = mean_squared_error(test_preds.mean(axis=1),y_test)*0.5
        errors.append(score)

        print('Score: {}'.format(mean_squared_error(test_preds.mean(axis=1),y_test)*0.5))
        print('Time duration:{}'.format(time.time() - t0))
        print()

    # print the result with all parameter choice in the grid
    print()
    print('Summary:')
    i = 0
    for n in grid:
        print('n = ' + str(n) + ', score = ' + str(round(errors[i],4)))
        i += 1

    # plt.plot(grid, errors)
    # plt.xlabel('Min leaf')
    # plt.ylabel('Mean square error of output')
    # plt.title('Number of Estimators vs. Mean Square Error')
    # plt.show()

    # output the predictions
    submission = pd.DataFrame({'pred':test_preds.mean(axis=1)})
    submission.to_csv('LightGBM.csv',header=None, index=False, float_format='%.4f')

    return mean_squared_error(test_preds.mean(axis=1),y_test)*0.5

# train CatBoost model
def catboo_train(X_train, y_train, X_test, y_test):
    print('CatBoost training...')

    # grid search
    # grid = [0.65,0.66,0.67,0.68,0.69,0.7,0.71,0.72,0.73,0.74,0.75]
    grid = [1]
    errors = []

    for x in grid:
        print('Cross validation 5-fold training...')
        t0 = time.time()
        test_preds = np.zeros((X_test.shape[0], 5))
        kf = KFold(n_splits = 5, shuffle=True, random_state=520)
        for i, (train_index, test_index) in enumerate(kf.split(X_train)):
            print('Round {}...'.format(i))

            catboo = CatBoostRegressor(iterations=731,        
                                        learning_rate=0.029,
                                        depth=7,
                                        l2_leaf_reg = 1.8,
                                        loss_function='RMSE',
                                        eval_metric='RMSE',
                                        random_seed=99,
                                        od_type='Iter',
                                        od_wait=50)
            catboo.fit(X_train.iloc[train_index], y_train.iloc[train_index], plot=False)
            test_preds[:,i] = catboo.predict(X_test)

        print('Score: {}'.format(mean_squared_error(test_preds.mean(axis=1),y_test)*0.5))
        print('Time duration:{}'.format(time.time() - t0))
        print()
        errors.append(mean_squared_error(test_preds.mean(axis=1),y_test)*0.5)

        submission = pd.DataFrame({'pred':test_preds.mean(axis=1)})
        submission.to_csv('CatBoost.csv',header=None, index=False, float_format='%.4f')

    print()
    print('Summary:')
    i = 0
    for n in grid:
        print('n = ' + str(n) + ', score = ' + str(round(errors[i],4)))
        i += 1

    # plt.plot(grid, errors)
    # plt.xlabel('Min leaf')
    # plt.ylabel('Mean square error of output')
    # plt.title('Number of Estimators vs. Mean Square Error')
    # plt.show()

    return mean_squared_error(test_preds.mean(axis=1),y_test)*0.5

# train Random Forest model
def rf_train(X_train, y_train, X_test, y_test):
    print('Random Forest training...')

    # estimators = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300]
    # estimators = [100,120,140,160,180,200,220,240,260,280,300]
    # estimators = [274,275,276,277,278,279,280]
    # estimators = [278]
    # depth = [11]
    estimators = [278]
    depth = [11]
    min_leaf = [67]
    errors = []

    # grid search with 3 parameters
    for n in estimators:
        for d in depth:
            for l in min_leaf:
                print('Cross validation 5-fold training...')
                t0 = time.time()
                test_preds = np.zeros((X_test.shape[0], 5))
                kf = KFold(n_splits = 5, shuffle=True, random_state=520)
                for i, (train_index, test_index) in enumerate(kf.split(X_train)):
                    print('Round {}...'.format(i))

                    rf = RandomForestRegressor(n_estimators=278, max_depth=11, random_state=42, min_samples_leaf=67)
                    rf.fit(X_train.iloc[train_index], y_train.iloc[train_index])
                    test_preds[:,i] = rf.predict(X_test)

                print('Score: {}'.format(mean_squared_error(test_preds.mean(axis=1),y_test)*0.5))
                print('Time duration:{}'.format(time.time() - t0))
                print()

                submission = pd.DataFrame({'pred':test_preds.mean(axis=1)})
                submission.to_csv('RandomForest.csv',header=None, index=False, float_format='%.4f')

                error = mean_squared_error(test_preds.mean(axis=1),y_test)*0.5
                errors.append(error)
    
    print()
    print('Summary:')
    i = 0
    for n in estimators:
        for d in depth:
            for l in min_leaf:
                print('n_estimators = ' + str(n) + ', max_depth = ' + str(d) + ', min_leaf = ' + str(l) +  ', score = ' + str(round(errors[i],4)))
                i += 1

    # plt.plot(min_leaf, errors)
    # plt.xlabel('Min leaf')
    # plt.ylabel('Mean square error of output')
    # plt.title('Number of Estimators vs. Mean Square Error')
    # plt.show()

    return mean_squared_error(test_preds.mean(axis=1),y_test)*0.5

# train Linear Regression model
def lr_train(X_train, y_train, X_test, y_test):
    print('Linear Regression model training...')

    print('Cross validation 5-fold training...')
    t0 = time.time()
    test_preds = np.zeros((X_test.shape[0], 5))
    kf = KFold(n_splits = 5, shuffle=True, random_state=520)
    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        print('Round {}...'.format(i))

        lr = LinearRegression()
        lr.fit(X_train.iloc[train_index], y_train.iloc[train_index])
        test_preds[:,i] = lr.predict(X_test)

    print('Score: {}'.format(mean_squared_error(test_preds.mean(axis=1),y_test)*0.5))
    print('Time duration:{}'.format(time.time() - t0))
    print(lr.coef_)
    print(lr.intercept_)
    print()

    submission = pd.DataFrame({'pred':test_preds.mean(axis=1)})
    submission.to_csv('LinearRegression.csv',header=None, index=False, float_format='%.4f')

    return mean_squared_error(test_preds.mean(axis=1),y_test)*0.5

# train XGBoost model
def xgb_train(X_train, y_train, X_test, y_test):
    print('xgb model training...')
    # grid = [100,200,300,400,500,600,700]
    grid = [1]
    # learning rate 0.14
    # max depth 10
    # n estimators 91
    # gamma 0
    # colsample_bytree 0.86
    # subsample 1
    errors = []


    for x in grid:
        print('Cross validation 5-fold training...')
        t0 = time.time()
        test_preds = np.zeros((X_test.shape[0], 5))
        kf = KFold(n_splits = 5, shuffle=True, random_state=520)
        for i, (train_index, test_index) in enumerate(kf.split(X_train)):
            print('Round {}...'.format(i))
            xgb_model = xgb.XGBRegressor(
                learning_rate=0.14,
                max_depth=10,
                n_estimators=91,
                booster='gbtree',       # gbtree or gb linear, that is, tree model or linear model
                gamma=0,                # The minimum "loss reduction" required for further splitting at a leaf
                colsample_bytree=0.86,
                subsample=1,
                reg_alpha = 1.77,       # default 0, controls the L1 regularization parameter. The larger the less likely to overfit
                reg_lambda = 0.42       # default 1, controls the L2 regularization parameter. The larger the less likely to overfit
            )
            xgb_model.fit(X_train.iloc[train_index], y_train.iloc[train_index])
            test_preds[:,i] = xgb_model.predict(X_test)

        print('Score: {}'.format(mean_squared_error(test_preds.mean(axis=1),y_test)*0.5))
        print('Time duration:{}'.format(time.time() - t0))
        print()
        errors.append(mean_squared_error(test_preds.mean(axis=1),y_test)*0.5)

    submission = pd.DataFrame({'pred':test_preds.mean(axis=1)})
    submission.to_csv('XGBoost.csv',header=None, index=False, float_format='%.4f')

    print()
    print('Summary:')
    i = 0
    for n in grid:
        print('n = ' + str(n) + ', score = ' + str(round(errors[i],4)))
        i += 1

    # plt.plot(grid, errors)
    # plt.xlabel('Min leaf')
    # plt.ylabel('Mean square error of output')
    # plt.title('Number of Estimators vs. Mean Square Error')
    # plt.show()

    return mean_squared_error(test_preds.mean(axis=1),y_test)*0.5

# train stacking model
def stacking_train(X_train, y_train, X_test, y_test):
    # too many combination of different models,
    # hence no grid search, just random walk

    gbm = xgb.XGBRegressor(
                learning_rate=0.3,#0.14,
                max_depth=10,
                n_estimators=800,#700,
                booster='gbtree',       # gbtree or gb linear， 也就是是树模型还是线性模型，默认树
                gamma=0,                # 叶节点上进一步分裂所需要的最小“损失减少”
                colsample_bytree=0.86,
                subsample=1,
                reg_alpha = 1.8,#1.77,       # 默认为0，控制L1正则化参数，参数越大模型越不容易过拟合
                reg_lambda = 1#0.42       # 默认为1，控制L2正则化参数，参数越大模型越不容易过拟合
            )
    catboo = CatBoostRegressor(iterations=1100,#1100,        # 这个从100-1000的曲线非常漂亮，缺图的话可以画一画
                                        learning_rate=0.029,
                                        depth=8,#7,
                                        l2_leaf_reg = 2.1,#1.8,
                                        loss_function='RMSE',
                                        eval_metric='RMSE',
                                        random_seed=99,
                                        od_type='Iter',
                                        od_wait=50)
    rf = RandomForestRegressor(n_estimators=500, max_depth=11, random_state=42, min_samples_leaf=67)
    lr = LinearRegression()
    stack = StackingCVRegressor(regressors=(gbm, catboo, rf),
                            meta_regressor=lr,
                            random_state=42)

    t0 = time.time()
    stack.fit(X_train, y_train)
    test_pred = stack.predict(X_test)
    print('Score: {}'.format(mean_squared_error(test_pred,y_test)*0.5))
    print('Time duration:{}'.format(time.time() - t0))
    print()
    
    submission = pd.DataFrame({'pred':test_pred})
    submission.to_csv('Stacking.csv',header=None, index=False, float_format='%.4f')

    return mean_squared_error(test_pred,y_test)*0.5

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data('data/')
    rf_train(X_train, y_train, X_test, y_test)

    models = ['lgb', 'catboost', 'random forest', 'linear regression', 'xgb', 'stacking']
    scores = []
    scores.append(lgb_train(X_train, y_train, X_test, y_test))
    scores.append(catboo_train(X_train, y_train, X_test, y_test))
    scores.append(rf_train(X_train, y_train, X_test, y_test))
    scores.append(lr_train(X_train, y_train, X_test, y_test))
    scores.append(xgb_train(X_train, y_train, X_test, y_test))
    scores.append(stacking_train(X_train, y_train, X_test, y_test))

    print('--------------------------------------------')
    print('The MSE for every model is:')
    for i in range(len(models)):
        print(models[i] + ': ' + str(round(scores[i],4)))