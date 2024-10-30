import numpy as np
import pandas as pd



X = np.genfromtxt("hw01_data_points.csv", delimiter = ",", dtype = str)
y = np.genfromtxt("hw01_class_labels.csv", delimiter = ",", dtype = int)





# STEP 3
# first 50000 data points should be included to train
# remaining 43925 data points should be included to test
# should return X_train, y_train, X_test, and y_test
def train_test_split(X, y):
    # your implementation starts below
    X_train=X[0:50000]
    y_train=y[0:50000]
    X_test=X[50000:93926]
    y_test=y[50000:93926]
    
    
    # your implementation ends above
    return(X_train, y_train, X_test, y_test)

X_train, y_train, X_test, y_test = train_test_split(X, y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



# STEP 4
# assuming that there are K classes
# should return a numpy array with shape (K,)
def estimate_prior_probabilities(y):
    # your implementation starts below
    class_priors=[]
    ##masked the arrays, then took their sizes to see how many of these events occured
    for i in range(len(np.unique(y))): ##how many classes are there?
        class_priors.append(y[y==i+1].size/y.size) ##this will ensure that we will have every class
        
    class_priors=np.array(class_priors)

    
    # your implementation ends above
    return(class_priors)

class_priors = estimate_prior_probabilities(y_train)
print(class_priors)



# STEP 5
# assuming that there are K classes and D features
# should return four numpy arrays with shape (K, D)
def estimate_nucleotide_probabilities(X, y):
    # your implementation starts below
    ##pacd--> chance of adenine for class c at location d. Woah!
    #our size is 50000, first I should filter the classes, filter y==1's and 2's
    #then apply another filter, but this time on X, as first position=A, second position=A etc.
    pAcd=[]
    pCcd=[]
    pGcd=[]
    pTcd=[]
    num_rows,num_columns=X.shape
    nucleotid=['A','C','G','T']
    for z in nucleotid:
        
        for i in range(len(np.unique(y))):
            Filtered_Class_array=X[y==i+1]
            Class_splicer=[]
            for l in range(num_columns):
                ##now I am looking at a specific class, and specific location, perfect!

                count=np.sum((Filtered_Class_array[:,l]==z))
                Class_splicer.append(count/len(Filtered_Class_array))
                 ##since this is a boolean array
            if z=='A':
                pAcd.append(Class_splicer)
            if z=='C':
                pCcd.append(Class_splicer)
            if z=='G':
                pGcd.append(Class_splicer)
            if z=='T':
                pTcd.append(Class_splicer)
    pAcd=np.array(pAcd)
    pCcd=np.array(pCcd)
    pGcd=np.array(pGcd)
    pTcd=np.array(pTcd)

    
                
            
                
            
        
   ##thoses numpy arrays have only the wanted classes in them
    


    
    # your implementation ends above
    return(pAcd, pCcd, pGcd, pTcd)

pAcd, pCcd, pGcd, pTcd = estimate_nucleotide_probabilities(X_train, y_train)
print(pAcd)
print(pCcd)
print(pGcd)
print(pTcd)



# STEP 6
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def calculate_score_values(X, pAcd, pCcd, pGcd, pTcd, class_priors):
    # your implementation starts below
    score_values=[] #I will append b's here, N values!

    for l in range (len(X)):
        b=[] #this will have K values, the scores of classes!
        for z in range(len(class_priors)):
            #now I have a fixed class
            
            sum1=np.log(class_priors[z])
            for i in range(len(X[l])):
                if X[l][i]=='A':
                    sum1=sum1+np.log(pAcd[z][i])
                    
                elif X[l][i]=='G':
                    sum1=sum1+np.log(pGcd[z][i])

                
                elif X[l][i]=='C':
                    sum1=sum1+np.log(pCcd[z][i])

                    
                elif X[l][i]=='T':
                    sum1=sum1+np.log(pTcd[z][i])
            b.append(sum1)
        score_values.append(b)
    score_values=np.array(score_values)
                    
                
                    
               
                      
                
                    
            
            
        
    # your implementation ends above
    return(score_values)

scores_train = calculate_score_values(X_train, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_train)

scores_test = calculate_score_values(X_test, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_test)



# STEP 7
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, scores):
    # your implementation starts below
    
    #I will use np.argmax(scores,axis=1)
    class_predicted=np.argmax(scores,axis=1)
    class_predicted=class_predicted+1 #python is 0 index based, lets add +1 to everything!
    ##class predicted is what we predict, we will use that.
    confusion_matrix= np.zeros((len(np.unique(y_truth)),len(np.unique(y_truth))),dtype=int)
    for i in range(len(np.unique(y_truth))):
        for l in range(len(np.unique(y_truth))):
            confusion_matrix[i,l]=np.sum((y_truth==l+1)&(class_predicted==i+1))
     ##VOILA!
        
    
    
    
        
    
    # your implementation ends above
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)

confusion_test = calculate_confusion_matrix(y_test, scores_test)
print(confusion_test)
