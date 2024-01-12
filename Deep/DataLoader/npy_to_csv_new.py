import numpy as np

x_train = "/home/dp7972/Desktop/DAYOU/Dayou/Data/newDataset/data/Delicious1_x_train.npy"
y_train = "/home/dp7972/Desktop/DAYOU/Dayou/Data/newDataset/data/Delicious1_pi_train.npy"
x_test = "/home/dp7972/Desktop/DAYOU/Dayou/Data/newDataset/data/Delicious1_x_test.npy"
y_test = "/home/dp7972/Desktop/DAYOU/Dayou/Data/newDataset/data/Delicious1_pi_test.npy"

x_tr = np.load(x_train)
y_tr = np.load(y_train)

x_ts = np.load(x_test)
y_ts = np.load(y_test)

np.savetxt("/home/dp7972/Desktop/DAYOU/Dayou/Data/Delicious_x_train.csv", x_tr, delimiter=',')
np.savetxt("/home/dp7972/Desktop/DAYOU/Dayou/Data/Delicious_y_train.csv", y_tr, delimiter=',')
np.savetxt("/home/dp7972/Desktop/DAYOU/Dayou/Data/Delicious_x_test.csv", x_ts, delimiter=',')
np.savetxt("/home/dp7972/Desktop/DAYOU/Dayou/Data/Delicious_y_test.csv", y_ts, delimiter=',')
