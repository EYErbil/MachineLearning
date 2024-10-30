import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import pandas as pd



X_train = np.genfromtxt(fname = "hw02_data_points.csv", delimiter = ",", dtype = float)
y_train = np.genfromtxt(fname = "hw02_class_labels.csv", delimiter = ",", dtype = int)



# STEP 3
# assuming that there are K classes
# should return a numpy array with shape (K,)
def estimate_prior_probabilities(y):
    # your implementation starts below
    class_priors=[]
    for i in range(len(np.unique(y))):   #how many classes are there?
        class_priors.append(np.count_nonzero(y==i+1)/len(y))  #append everysingle one of them.
    class_priors=np.array(class_priors)
        
        
    # your implementation ends above
    return(class_priors)

class_priors = estimate_prior_probabilities(y_train)
print(class_priors)



# STEP 4
# assuming that there are K classes and D features
# should return a numpy array with shape (K, D)
def estimate_class_means(X, y):
    # your implementation starts below
    sample_means=[]

    for i in range(len(np.unique(y))): #number of classes
        Filtered=X[y==i+1]
        b=[]

        for l in range(len(X[0])): #number of columns!
            b.append(np.mean(Filtered[:,l]))
            
            
               
        sample_means.append(b)
    sample_means=np.array(sample_means)
    
    # your implementation ends above
    return(sample_means)

sample_means = estimate_class_means(X_train, y_train)
print(sample_means)



# STEP 5
# assuming that there are K classes and D features
# should return a numpy array with shape (K, D, D)
def estimate_class_covariances(X, y):
    # your implementation starts below
    means=estimate_class_means(X, y)
    sample_covariances=[]

    
    #what I need is the class filter first!
    for i in (np.unique(y)):
        summation=0
        Filtered=X[y==i]        
        for data in Filtered: #♦only the classes I want
            summation=summation+(np.outer(data-means[i-1],data-means[i-1]))
            
            
        summation=summation/len(Filtered)
        sample_covariances.append(summation)
    sample_covariances=np.array(sample_covariances)
            
            
            
    
    
    # your implementation ends above
    return(sample_covariances)

sample_covariances = estimate_class_covariances(X_train, y_train)
print(sample_covariances)



# STEP 6
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def calculate_score_values(X, class_means, class_covariances, class_priors):
    # your implementation starts below
    score_values=[]
    for data in X:
        score_vector=[]
        for class_d in range(len(class_priors)): #class priors will give me a len K vector
            
            determinant=class_covariances[class_d]
            determinant=np.linalg.det(determinant)
            inverse=np.linalg.inv(class_covariances[class_d])
            mean=class_means[class_d]
            what=data-mean
            what_t=what.reshape(1,-1)
        
            score=((-(len(X[0]))/2)*(np.log(2*np.pi)))-(1/2)*((np.log(determinant)))+np.log(class_priors[class_d])-(1/2)*(what_t@inverse@what)
            score=float(score)
            
            score_vector.append(score)
        
        score_values.append(score_vector)
    score_values=np.array(score_values)    
        
    # your implementation ends above
    return(score_values)

scores_train = calculate_score_values(X_train, sample_means,
                                      sample_covariances, class_priors)
print(scores_train)



# STEP 7
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, scores):
    # your implementation starts below
    
    class_predicted=np.argmax(scores,axis=1)
    class_predicted=class_predicted+1 #python is 0 index based, lets add +1 to everything!
    # I got this part from my Homework1, same thing really
    confusion_matrix= np.zeros((len(np.unique(y_truth)),len(np.unique(y_truth))),dtype=int)
    for i in range(len(np.unique(y_truth))):
        for l in range(len(np.unique(y_truth))):
            confusion_matrix[i,l]=np.sum((y_truth==l+1)&(class_predicted==i+1))
    #this part is from my homework 1 too.
    
    
    
    # your implementation ends above
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)



def draw_classification_result(X, y, class_means, class_covariances, class_priors):
    class_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#6a3d9a"])
    K = np.max(y)

    x1_interval = np.linspace(-75, +75, 151)
    x2_interval = np.linspace(-75, +75, 151)
    x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
    X_grid = np.vstack((x1_grid.flatten(), x2_grid.flatten())).T
    scores_grid = calculate_score_values(X_grid, class_means, class_covariances, class_priors)

    score_values = np.zeros((len(x1_interval), len(x2_interval), K))
    for c in range(K):
        score_values[:,:,c] = scores_grid[:, c].reshape((len(x1_interval), len(x2_interval)))

    L = np.argmax(score_values, axis = 2)

    fig = plt.figure(figsize = (6, 6))
    for c in range(K):
        plt.plot(x1_grid[L == c], x2_grid[L == c], "s", markersize = 2, markerfacecolor = class_colors[c], alpha = 0.25, markeredgecolor = class_colors[c])
    for c in range(K):
        plt.plot(X[y == (c + 1), 0], X[y == (c + 1), 1], ".", markersize = 4, markerfacecolor = class_colors[c], markeredgecolor = class_colors[c])
    plt.xlim((-75, 75))
    plt.ylim((-75, 75))
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.show()
    return(fig)
    
fig = draw_classification_result(X_train, y_train, sample_means, sample_covariances, class_priors)
fig.savefig("hw02_result_different_covariances.pdf", bbox_inches = "tight")



# STEP 8
# assuming that there are K classes and D features
# should return a numpy array with shape (K, D, D)
def estimate_shared_class_covariance(X, y):
    # your implementation starts below
    mean_Vec=[]
    for i in range (len(X[0])-1):
        mean=np.mean(X[:,i])
        mean_Vec.append(mean)
    mean_Vec=np.array(mean_Vec)
    sample_covariance=0
    sample_covariances=[]
    for l in X:
        sample_covariance=sample_covariance+np.outer(l-mean_Vec,l-mean_Vec)
    sample_covariance=sample_covariance/len(X)
    for z in range(len(np.unique(y))):
        sample_covariances.append(sample_covariance)
    sample_covariances=np.array(sample_covariances)
        
    # your implementation ends above
    return(sample_covariances)

sample_covariances = estimate_shared_class_covariance(X_train, y_train)
print(sample_covariances)

scores_train = calculate_score_values(X_train, sample_means,
                                      sample_covariances, class_priors)
print(scores_train)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)

fig = draw_classification_result(X_train, y_train, sample_means, sample_covariances, class_priors)
fig.savefig("hw02_result_shared_covariance.pdf", bbox_inches = "tight")
