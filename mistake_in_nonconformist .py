from sklearn.datasets import load_diabetes
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from nonconformist.cp import IcpRegressor
from nonconformist.nc import NcFactory
from nonconformist.nc import SignErrorErrFunc


diabetes = load_diabetes()
idx = np.random.permutation(diabetes.target.size)

# Divide the data into proper training set, calibration set and test set
idx_train, idx_cal, idx_test = idx[:310], idx[310:399], idx[399:]

model = RandomForestRegressor()	# Create the underlying model
nc = NcFactory.create_nc(model,err_func=SignErrorErrFunc())	# Create a default nonconformity function
icp = IcpRegressor(nc)			# Create an inductive conformal regressor

# Fit the ICP using the proper training set
icp.fit(diabetes.data[idx_train, :], diabetes.target[idx_train])

# Calibrate the ICP using the calibration set
icp.calibrate(diabetes.data[idx_cal, :], diabetes.target[idx_cal])
# Produce predictions for the test set, with confidence 95%
prediction = icp.predict(diabetes.data[idx_test, :], significance=0.05)

# Produce predictions for the test set, with confidence 95%
prediction = icp.predict(diabetes.data[idx_test, :], significance=0.05)
print(prediction[:5])


#printing predictions without significance
prediction_nondef = icp.predict(diabetes.data[idx_test, :])
print(prediction_nondef[:,:,4][:5])

# comparing the results

# lower bound is always correct, but the upper bound is always wrong
comparison = (prediction == prediction_nondef[:,:,4])
low_bound_corr_frac = sum(comparison[:,0]) / len(idx_test) * 100
up_bound_corr_frac = sum(comparison[:,1]) / len(idx_test) * 100
print('Lower bound correctly predicted: {}%'.format(np.round(low_bound_corr_frac, 2)))
print('Upper bound correctly predicted: {}%'.format(np.round(up_bound_corr_frac, 2)))


