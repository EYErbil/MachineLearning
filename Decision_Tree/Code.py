import matplotlib.pyplot as plt
import numpy as np


# read data into memory
data_set_train = np.genfromtxt("hw04_data_set_train.csv", delimiter = ",")
data_set_test = np.genfromtxt("hw04_data_set_test.csv", delimiter = ",")

# get X and y values
X_train = data_set_train[:, 0:1]
y_train = data_set_train[:, 1]
X_test = data_set_test[:, 0:1]
y_test = data_set_test[:, 1]

# set drawing parameters
minimum_value = 0.00
maximum_value = 2.00
step_size = 0.002
X_interval = np.arange(start = minimum_value, stop = maximum_value + step_size, step = step_size)
X_interval = X_interval.reshape(len(X_interval), 1)

def plot_figure(X_train, y_train, X_test, y_test, X_interval, y_interval_hat):
    fig = plt.figure(figsize = (8, 4))
    plt.plot(X_train[:, 0], y_train, "b.", markersize = 10)
    plt.plot(X_test[:, 0], y_test, "r.", markersize = 10)
    plt.plot(X_interval[:, 0], y_interval_hat, "k-")
    plt.xlim([-0.05, 2.05])
    plt.xlabel("Time (sec)")
    plt.ylabel("Signal (millivolt)")
    plt.legend(["training", "test"])
    plt.show()
    return(fig)
## MY REFERENCE WOULD BE MEHMET GONEN'S TREE LAB-06, I TOOK MY CODE MOSTLY FROM THERE.

# STEP 2
# should return necessary data structures for trained tree
def decision_tree_regression_train(X_train, y_train, P):
    # create necessary data structures
    node_indices = {}
    is_terminal = {}
    need_split = {}

    node_features = {}
    node_splits = {}
    node_means = {}
    # your implementation starts below
    node_indices = {1: np.arange(0, len(y_train))}  
    is_terminal = {1: False}  
    need_split = {1: True}  
    while True:
        split_nodes = [key for key, value in need_split.items() if value]
        
        
        if not split_nodes:
            break
        
        for split_node in split_nodes:
            
            data_indices = node_indices[split_node]
            need_split[split_node] = False
            
            
            if len(data_indices) < P or len(np.unique(y_train[data_indices])) == 1:
                is_terminal[split_node] = True
                node_means[split_node] = np.mean(y_train[data_indices])
                continue  # No need to split this node
            
            is_terminal[split_node] = False
            
            best_score = float('inf')
            best_feature = None
            best_split = None
            
            for d in range(X_train.shape[1]):
                unique_values = np.sort(np.unique(X_train[data_indices, d]))
                split_positions = (unique_values[1:] + unique_values[:-1]) / 2  
                
                for s in split_positions:
                    left_indices = data_indices[X_train[data_indices, d] > s]
                    right_indices = data_indices[X_train[data_indices, d] <= s]
                    
                    left_mse = np.var(y_train[left_indices]) * len(left_indices)
                    right_mse = np.var(y_train[right_indices]) * len(right_indices)
                    total_mse = (left_mse + right_mse) / len(data_indices)
                    
                    if total_mse < best_score:
                        best_score = total_mse
                        best_feature = d
                        best_split = s
            
            node_features[split_node] = best_feature
            node_splits[split_node] = best_split
            
            left_indices = data_indices[X_train[data_indices, best_feature] > best_split]
            right_indices = data_indices[X_train[data_indices, best_feature] <= best_split]
            
            if len(left_indices) > P:
                node_indices[2 * split_node] = left_indices
                is_terminal[2 * split_node] = False
                need_split[2 * split_node] = True
            else:
                is_terminal[2 * split_node] = True
                node_means[2 * split_node] = np.mean(y_train[left_indices])
            
            if len(right_indices) > P:
                node_indices[2 * split_node + 1] = right_indices
                is_terminal[2 * split_node + 1] = False
                need_split[2 * split_node + 1] = True
            else:
                is_terminal[2 * split_node + 1] = True
                node_means[2 * split_node + 1] = np.mean(y_train[right_indices])
    # your implementation ends above
    return(is_terminal, node_features, node_splits, node_means)

# STEP 3
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def decision_tree_regression_test(X_query, is_terminal, node_features, node_splits, node_means):
    # your implementation starts below
    y_hat = np.zeros(X_query.shape[0])

    for i in range(X_query.shape[0]):
        node =1
        while not is_terminal[node]:
            feature=node_features[node]
            split=node_splits[node]
            if X_query[i,feature]>split:
                node=2*node
            else:
                node=node*2+1
        y_hat[i]=node_means[node]
    # your implementation ends above
    return(y_hat)

# STEP 4
# assuming that there are T terminal node
# should print T rule sets as described
def extract_rule_sets(is_terminal, node_features, node_splits, node_means):
    # your implementation starts below
    terminal_nodes = [key for key, value in is_terminal.items() if value]
    
    for terminal_node in terminal_nodes:
        index = terminal_node
        rules = np.array([])  # Array to store the rules
        
        while index > 1:
            parent = int(index // 2)  
            
            if index % 2 == 0:  # Left child
                rules = np.append(rules, 
                                  "x{:d} > {:.2f}".format(
                                      node_features[parent] + 1, 
                                      node_splits[parent]
                                  ))
            else:  
                rules = np.append(rules,
                                  "x{:d} <= {:.2f}".format(
                                      node_features[parent] + 1, 
                                      node_splits[parent]
                                  ))
            
            
            index = parent
        
        
        rules = np.flip(rules)
        
        node_mean = node_means[terminal_node]
        print("Node {:02}: {} => {}".format(terminal_node, rules, node_mean))
    return 0;
    # your implementation ends above

P = 20
is_terminal, node_features, node_splits, node_means = decision_tree_regression_train(X_train, y_train, P)
y_interval_hat = decision_tree_regression_test(X_interval, is_terminal, node_features, node_splits, node_means)
fig = plot_figure(X_train, y_train, X_test, y_test, X_interval, y_interval_hat)
fig.savefig("decision_tree_regression_{}.pdf".format(P), bbox_inches = "tight")

y_train_hat = decision_tree_regression_test(X_train, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_train - y_train_hat)**2))
print("RMSE on training set is {} when P is {}".format(rmse, P))

y_test_hat = decision_tree_regression_test(X_test, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("RMSE on test set is {} when P is {}".format(rmse, P))

P = 50
is_terminal, node_features, node_splits, node_means = decision_tree_regression_train(X_train, y_train, P)
y_interval_hat = decision_tree_regression_test(X_interval, is_terminal, node_features, node_splits, node_means)
fig = plot_figure(X_train, y_train, X_test, y_test, X_interval, y_interval_hat)
fig.savefig("decision_tree_regression_{}.pdf".format(P), bbox_inches = "tight")

y_train_hat = decision_tree_regression_test(X_train, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_train - y_train_hat)**2))
print("RMSE on training set is {} when P is {}".format(rmse, P))

y_test_hat = decision_tree_regression_test(X_test, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("RMSE on test set is {} when P is {}".format(rmse, P))

extract_rule_sets(is_terminal, node_features, node_splits, node_means)
