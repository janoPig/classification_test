from rils_rols.rils_rols import RILSROLSRegressor
from HROCH import SymbolicRegressor
from pmlb import fetch_data
from sklearn.model_selection import train_test_split

df = fetch_data('586_fri_c3_1000_25')
X = df.drop('target', axis=1)
y = df['target']

TEST_SIZE = 0.25
RANDOM_STATE = 123

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

rr_params = {
    'random_state' : 123,
    'max_seconds' : 10,
    'max_fit_calls' : 1000000000,
    'sample_size' : 1.0,
    'verbose' : False,
}

print('RILS-ROLS DataFrame')
est = RILSROLSRegressor(**rr_params)
est.fit(X_train, y_train)
print(est.score(X_test.to_numpy(), y_test.to_numpy()))
print(est.score(X_test, y_test))

print('RILS-ROLS Numpy')
est = RILSROLSRegressor(**rr_params)
est.fit(X_train.to_numpy(), y_train.to_numpy())
print(est.score(X_test.to_numpy(), y_test.to_numpy()))
print(est.score(X_test, y_test))

print('RILS-ROLS Mixed1')
est = RILSROLSRegressor(**rr_params)
est.fit(X_train, y_train.to_numpy())
print(est.score(X_test.to_numpy(), y_test.to_numpy()))
print(est.score(X_test, y_test))

print('RILS-ROLS Mixed2')
est = RILSROLSRegressor(**rr_params)
est.fit(X_train.to_numpy(), y_train)
print(est.score(X_test.to_numpy(), y_test.to_numpy()))
print(est.score(X_test, y_test))

hr_params = {
    'num_threads' : 1,
    'random_state' : 123,
    'time_limit' : 10,
    'iter_limit' : 0,
}
print('HROCH DataFrame')
est = SymbolicRegressor(**hr_params)
est.fit(X_train, y_train)
print(est.score(X_test.to_numpy(), y_test.to_numpy()))
print(est.score(X_test, y_test))

print('HROCH Numpy')
est = SymbolicRegressor(**hr_params)
est.fit(X_train.to_numpy(), y_train.to_numpy())
print(est.score(X_test.to_numpy(), y_test.to_numpy()))
print(est.score(X_test, y_test))

print('HROCH Mixed1')
est = SymbolicRegressor(**hr_params)
est.fit(X_train, y_train.to_numpy())
print(est.score(X_test.to_numpy(), y_test.to_numpy()))
print(est.score(X_test, y_test))

print('HROCH Mixed2')
est = SymbolicRegressor(**hr_params)
est.fit(X_train.to_numpy(), y_train)
print(est.score(X_test.to_numpy(), y_test.to_numpy()))
print(est.score(X_test, y_test))






