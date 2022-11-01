from sklearn.datasets import load_diabetes
import numpy as np

def get_batch_diabetes(data, batch_size):
    # the numpy function choice(length, number)
    # selects at random "batch_size" integers from 
    # the range [0, length-1] corresponding to the
    # row indices.
    rows    = np.random.choice(data.shape[0], batch_size)
    # this returns a np array of a list (of size batchsize) of rown numbers, then you can just do data[rows] to get the data at those rows
    batch_x = data[rows]
    # batch_x.T[-1] = np.random.uniform(0, 1, batch_size)
    return batch_x

examples_x, examples_y = load_diabetes(return_X_y=True)

train_frac, test_frac = 0.8, 0.2
train_ind = round(examples_x.shape[0] * train_frac)
examples_x_train, examples_x_test = examples_x[0:train_ind, :], examples_x[train_ind:,:]
examples_y_train, examples_y_test = examples_y[0:train_ind], examples_y[train_ind:]

inference_set_x = examples_x_test
def get_data_set():
    """this creates a list of size n_examples of np arrays, each of size 2x2 
    Generators (functions that use yield instead of return) are useful because they only get the subset of data that you want as opposed to loading the whole set to memory, and only give you the subset once, when called 

    To use this, do import data_loader_two_by_two as dat; train_generator, evaluation_generator = get_data_set(); new_train_example=next(train_generator())
    """
    def training_set_x():
        #start with an infinite loop, so that you can keep calling next (i.e. set = train_set(); set.next() ) until you run out of training examples
        while True:
            batch_x = get_batch_diabetes(examples_x_train, batch_size=20)
            yield batch_x
            
    def evaluation_set_x():
        #start with an infinite loop, so that you can keep calling next (i.e. set = train_set(); set.next() ) until you run out of training examples
        while True:
            batch_x = get_batch_diabetes(examples_x_train, batch_size=20)
            yield batch_x


    def training_set_y():
        #start with an infinite loop, so that you can keep calling next (i.e. set = train_set(); set.next() ) until you run out of training examples
        while True:
            batch_x = get_batch_diabetes(examples_x_train, batch_size=20)
            yield batch_x
            
    def evaluation_set_y():
        #start with an infinite loop, so that you can keep calling next (i.e. set = train_set(); set.next() ) until you run out of training examples
        while True:
            batch_x = get_batch_diabetes(examples_x_train, batch_size=20)
            yield batch_x
            
    return training_set_x, training_set_y, evaluation_set_x, evaluation_set_y









# if __name__ == '__main__':
# 	get_data_set()



