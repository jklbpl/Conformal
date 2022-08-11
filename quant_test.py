import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from nonconformist.cp import IcpRegressor
from nonconformist.nc import NcFactory
from nonconformist.nc import SignErrorErrFunc
from nonconformist.nc import AbsErrorErrFunc


data_data, data_target = fetch_california_housing(return_X_y= True)
#data_data, data_target = load_diabetes(return_X_y= True)

df_features = pd.DataFrame(data_data)
df_target = pd.DataFrame(data_target)
idx_size = df_target.size

np.random.seed(2)

idx = np.random.permutation(len(data_data))

# test = 10%, test(test+calib) = 90% (80%+20%)
test_size = int(idx_size  * 0.1)
train_size = idx_size  - test_size
calib_size = int(train_size * 0.2)
train_size = train_size - calib_size

idx_train, idx_cal, idx_test = idx[:train_size], idx[train_size:train_size + calib_size], idx[train_size + calib_size:]


print('Test size: {}'.format(test_size))
print('Calibration size: {}'.format(calib_size))
print('Train size: {}'.format(train_size))

model = RandomForestRegressor()	# Create the underlying model
nc = NcFactory.create_nc(model,err_func = AbsErrorErrFunc())	# Create a default nonconformity function
icp = IcpRegressor(nc)			# Create an inductive conformal regressor

# Fit the ICP using the proper training set
icp.fit(data_data[idx_train, :], data_target[idx_train])

# Calibrate the ICP using the calibration set
icp.calibrate(data_data[idx_cal, :], data_target[idx_cal])

predictions_cal = model.predict(data_data[idx_cal,:])
predictions_test = model.predict(data_data[idx_test,:])
cal_array = np.arange(0, 300, 1)

signif = 0.03
signif_arr = np.arange(0.01, 1, 0.01)

#functions for conformal prediction abserror + abs_err_inv
def abserror(prediction_set, y):
  return np.abs(prediction_set - y)


def abs_err_inv(cal_score, significance, sign_is_nondef, method ):
  if sign_is_nondef is False:
    cal_scores_sorted =  np.sort(cal_score)[::-1]

    border_2 = 1-(significance)
    quantile_significance = np.quantile(cal_scores_sorted, border_2, method=method)
    cal_scores_sorted_bool = cal_scores_sorted >= quantile_significance
    for i in range(len(cal_scores_sorted)):
      if cal_scores_sorted_bool[i]:
        number = i
    return cal_scores_sorted[number]

  if sign_is_nondef is True:
    nc = np.sort(cal_score)[::-1]
    border_1 = int(np.floor(significance * (nc.size + 1))) - 1
    border = min(max(border_1, 0), nc.size - 1)

    return nc[border]


# this function tests how good each np.quantile method works with abs_err_inv function
# (in a nutshell - np.floor and np.quantile)
def correlat(cal_score_corr, max_num, step):
  if cal_score_corr is not None:
    cal_score = cal_score_corr
  if cal_score_corr is None:
    cal_score = np.arange(0, max_num, step)

  signn2 = np.arange(0.01, 1, 0.01)
  #interpol = ['higher','lower','nearest','midpoint','linear'] # - for older versions of numpy, the best was 'higher'
  method_q = ['inverted_cdf','averaged_inverted_cdf','closest_observation','interpolated_inverted_cdf',
              'hazen','weibull','median_unbiased','normal_unbiased']
  dict_data = []

  for i in (method_q):
    for j in signn2:

      border1 = abs_err_inv(cal_score, significance=j, sign_is_nondef=True, method= None)
      border2 = abs_err_inv(cal_score, significance=j, sign_is_nondef=False, method = i)

      if border1 != border2:
        dict_data.append({'Method': i,
                      'b1=?b2':border1 == border2, 'border':border1})
      else:
        dict_data.append({'Method': i,
                          'b1=?b2':border1 == border2,
                          'border1': border1,
                          'border2': border2})
  err_df=pd.DataFrame(dict_data)

  return err_df



cal_scores1 =abserror(predictions_cal, data_target[idx_cal])
df = correlat(cal_score_corr=cal_scores1, max_num=2000, step =1)
method_q = ['inverted_cdf','averaged_inverted_cdf','closest_observation','interpolated_inverted_cdf',
              'hazen','weibull','median_unbiased','normal_unbiased']

dff = df[df['Method'] == 'method_q']

for j in method_q:
  dff = df[df['Method'] == j]
  counter_true = 0
  counter_false = 0
  for i in dff.index:
    if dff['b1=?b2'][i] == True:
      counter_true = counter_true + 1
    else:
      counter_false = counter_false + 1

  print("Method:{}\n True results:{}%\n False results:{}%\n".format(j,(counter_true/len(dff))*100,(counter_false/len(dff))*100))
