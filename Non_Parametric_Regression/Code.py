import math
import matplotlib.pyplot as plt
import numpy as np

# read data into memory
data_set_train = np.genfromtxt("hw03_data_set_train.csv", delimiter = ",")
data_set_test = np.genfromtxt("hw03_data_set_test.csv", delimiter = ",")

# get x and y values
x_train = data_set_train[:, 0]
y_train = data_set_train[:, 1]
x_test = data_set_test[:, 0]
y_test = data_set_test[:, 1]

# set drawing parameters
minimum_value = 0.00
maximum_value = 2.00
x_interval = np.arange(start = minimum_value, stop = maximum_value + 0.002, step = 0.002)

def plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat):
    fig = plt.figure(figsize = (8, 4))
    plt.plot(x_train, y_train, "b.", markersize = 10)
    plt.plot(x_test, y_test, "r.", markersize = 10)
    plt.plot(x_interval, y_interval_hat, "k-")
    plt.xlim([-0.05, 2.05])
    plt.xlabel("Time (sec)")
    plt.ylabel("Signal (millivolt)")
    plt.legend(["training", "test"])
    plt.show()
    return(fig)



# STEP 3
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def regressogram(x_query, x_train, y_train, left_borders, right_borders):
    # your implementation starts below
    
    
    ''' the commented solution is assuming origin=0, I fixed this.
        def binnumber(data,bin_distance):
        if (data/bin_distance==int(data/bin_distance)):
            return(int(data/bin_distance))
            
        return ((int(data/bin_distance))+1)
    def myfunction(data,bin_distance):
        sums=0
        count_bin=0
        for i in range(len(x_train)):
            if (binnumber(x_train[i],bin_distance)==binnumber(data,bin_distance)):
                count_bin=count_bin+1
                sums=sums+y_train[i]
        if (count_bin==0):
            return 
        else:
            
            return sums/count_bin
        '''
    def find_bin_index(data, left_borders, right_borders):
        for i in range(len(left_borders)):
            if left_borders[i] <= data <= right_borders[i]:
                return i
        return -1  # returns -1 if the data point is outside the bin ranges
    def myfunction(data):
        sums=0
        count_bin=0
        for i in range(len(x_train)):
            if (find_bin_index(x_train[i],left_borders,right_borders)==find_bin_index(data,left_borders,right_borders)):
                count_bin=count_bin+1
                sums=sums+y_train[i]
        if (count_bin==0):
            return 
        else:
            
            return sums/count_bin
    y_hat=[] #To be converted by numpy
    for x in x_query:
        y_hat.append(myfunction(x))
    y_hat=np.array(y_hat)
            
                
        
    

    # your implementation ends above
    return(y_hat)
    
bin_width = 0.10
left_borders = np.arange(start = minimum_value, stop = maximum_value, step = bin_width)
right_borders = np.arange(start = minimum_value + bin_width, stop = maximum_value + bin_width, step = bin_width)

y_interval_hat = regressogram(x_interval, x_train, y_train, left_borders, right_borders)
fig = plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat)
fig.savefig("regressogram.pdf", bbox_inches = "tight")

y_test_hat = regressogram(x_test, x_train, y_train, left_borders, right_borders)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Regressogram => RMSE is {} when h is {}".format(rmse, bin_width))



# STEP 4
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def running_mean_smoother(x_query, x_train, y_train, bin_width):
    # your implementation starts below
    
    def classifier(data_test,data_train):
        ##if (((data_test-data_train)/bin_width)>=(-1/2) and ((data_test-data_train)/bin_width)<(1/2)):
          ##  return 1
        if (data_test-0.5*bin_width<data_train):
            
            if (data_train<=data_test+0.5*bin_width):

                return 1
        return 0
    
    def my_function(data):
        sums=0
        
        
        samegroupcount=0
        for i in range(len(x_train)):
            
            
            sums=sums+y_train[i]*classifier(data,x_train[i])
            samegroupcount=samegroupcount+classifier(data,x_train[i])
        
        return sums/samegroupcount
    y_hat=[]
    for x in x_query:
        y_hat.append(my_function(x))
    y_hat=np.array(y_hat)
    # your implementation ends above
    return(y_hat)

bin_width = 0.10

y_interval_hat = running_mean_smoother(x_interval, x_train, y_train, bin_width)
fig = plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat)
fig.savefig("running_mean_smoother.pdf", bbox_inches = "tight")

y_test_hat = running_mean_smoother(x_test, x_train, y_train, bin_width)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Running Mean Smoother => RMSE is {} when h is {}".format(rmse, bin_width))



# STEP 5
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def kernel_smoother(x_query, x_train, y_train, bin_width):
    # your implementation starts below
    def gaussian(u):
        return (1/(2*np.pi)**(1/2))*np.exp(-(u**2)/2)
    def myfunction(data):
        nominator=0
        denominator=0
        for i in range (len(x_train)):
            nominator=nominator+((y_train[i])*gaussian((data-x_train[i])/bin_width))
            denominator=denominator+gaussian((data-x_train[i])/bin_width)
        
        
        return nominator/denominator
    y_hat=[]
    for x in x_query:
        y_hat.append(myfunction(x))
        
    y_hat=np.array(y_hat)
    # your implementation ends above
    return(y_hat)

bin_width = 0.02

y_interval_hat = kernel_smoother(x_interval, x_train, y_train, bin_width)
fig = plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat)
fig.savefig("kernel_smoother.pdf", bbox_inches = "tight")

y_test_hat = kernel_smoother(x_test, x_train, y_train, bin_width)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Kernel Smoother => RMSE is {} when h is {}".format(rmse, bin_width))
