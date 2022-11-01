import numpy as np
from numba import njit


# def initialize_stuff_with_numba:
    # return weights, w_grad,

class Dense(object):
    
    def __init__(self,
    previous_layer,
    N_outputs,
    activation,
    dropout_rate
    ):
        """Define the number of input and output NODES for a particular layer. A layer could then be made by calling 
        layer= Dense(N_inputs,N_outputs). For example, a single layer NN with no hidden layers could be done with
        model_1=[Dense(N_inputs,N_outputs)]
        all these things are first initialized randomly here, and then they pass to the later functions of this class

        previous layer is the N_inpus
        activation is activation class (such as tanh) that well be specified in run_framework.py
        """
        self.previous_layer=previous_layer
        self.N_inputs = self.previous_layer.y.size
        self.N_outputs = int(N_outputs)
        self.activation=activation
        self.learning_rate=int(1e-2)
        #remember that the dropout rate is on average that value (approaches this value as number of nodes approaches infinity
        self.dropout_rate=dropout_rate


        #the size of the weights is a matrix of ( N_inputs + 1) X (N_outputs) (and the inputs) is w[N_inputs]+b which is the rows, and the outputs will have [N_outputs]
        rows = self.N_inputs+1#the +1 because there will be a bias vector 
        columns = self.N_outputs
        self.weights = np.random.sample(size=(rows,columns) )
        #random sample returns a random unifrom between0 and 1
        self.w_grad = np.zeros((self.N_inputs+1, self.N_outputs))

        #Define set of inputs coming in to the network
        self.x = np.zeros((1,self.N_inputs+1))
        self.y=np.zeros((1,self.N_outputs))
        
        self.i_dropout=np.zeros(shape=self.x.size, dtype=bool)

        #initiate empty list of regularizers
        self.regularizers=[]

    def add_regularizer(self, new_regularizer):
        self.regularizers.append(new_regularizer)
            
    
    def reset(self):
        #make sure we reset the right numbers of outputs and inputs
        self.x = np.zeros((1, self.N_inputs))
        self.y = np.zeros((1,self.N_outputs))
        self.dLoss_dx = np.zeros((1, self.N_inputs))
        self.dLoss_dy=np.zeros((1, self.N_outputs))

    # @njit doesnt work here
    def forward_propagate_layer(self, inputs, evaluating=False):
        """propagate the inputs forward through the NN

        Args:
            inputs : (INPUT TO THE LAYER) vector of values of size [1,N_input]
            Evaluating: a boolean, on whether your calling this function in evaluation or training
        Returns:
            y : of size [1, N_out]
        """
        #if in evaluation mode, we want to use the whole NN, i.e. we dont want to use dtopout
        if evaluating:
            dropout_rate=0
        else:
            dropout_rate=self.dropout_rate
        
        #generate a list of false booleans of the size of the inputs (nodes) of he layer
        # self.i_dropout=np.zeros(shape=self.x.size, dtype=bool)
        #sample a random uniform between 0 and 1
        unif = np.random.uniform(size=self.x.size)
        #if unif < dropout_rate, set dropout at that index (node) to be true
        self.i_dropout[np.where(unif < dropout_rate)] = True
        #now set every node (index of x) where i_dropout=True to 0
        self.x[:, self.i_dropout] = 0
        #set everywhere where i_dropout is not true (the np.logical_not just flips the booleans) to a multiplicative factor, so that the aggregate result will have the same mean/variance 
        self.x[:, np.logical_not(self.i_dropout)] *= 1/(1-dropout_rate)
        
        
        
        #inputs are the inputs to the layer
        #make a 1X1 bias matrix of ones
        bias=np.ones((1,1))
        #stack the bias on top of the inputs by adding it as a new column (axis 1)
        self.x = np.concatenate((inputs, bias), axis=1)
        #matrix-multiply the selt of augmented inputs x with the weights
        self.y_intermediate = self.x @ self.weights
        #the shapes of what being multiplied is the following:
        #[1, N_in +1] X [N_in +1, N_out] = [1, N_out]

        #Perform activation on the output for the final output
        self.y = self.activation.calc(self.y_intermediate)
        #here y is really yprime in the back_propagate_layer() layer , ie yprime=activation_function(y)
        return self.y

    
    def back_propagate_layer(self, dLoss_dy):
        """
        Args:
            dLoss_dy ([type]): the derivative of loss wrt y for ONE LAYER. 
            dL/dx = dL/dy * dy/dx
            EG if its a linear layer, y=mx+b and and L=(y-x)^2 with no activation function then: 
            dL/dx = dL/dy * m (and we dont need to simplify since dL/dy is an input) 

        If there is an activation function f such that y = f(y') then 
        dLoss/dx = dLoss/dy dy/dy' dy'/dx
                        = dLoss/dy d [f(y')]/dy' dy'/dx

        Returns:
            dLoss/dx of the current layer
        """
        # $y' = \vec{x} \cdot \vec{w}^T $
        # i.e. yprime = self.x @ self.weights.transpose()
        # so $dy'/dx = x$ i.e. dyprime_dx = self.weights
        # and $dyprime/dw = \vec{w}^T$ ie dyprime_dw=self.weights.transpose()
        
        #dy/dy'=d [f(y')]/dy'
        dy_dyprime = self.activation.calc_deriv(self.y)

        #y = f(y') = f(x @ weights) so 
        # dy/dw = dy/dy' dy'/dw  = dy/dy' * x
        dy_dw = self.x.transpose() @  dy_dyprime

        #dL/dw = dL/dy * dy/dw
        dLoss_dw = dLoss_dy * dy_dw

        #update the weights by subtracting the gradient
        self.weights = self.weights - ( dLoss_dw * self.learning_rate)

        # ADD REGULARIZAERS TO THE WEIGHTS
        for regularizer in self.regularizers:
            #self is a copy of the whole class, so e.g.
            # layer = Layer(),  update(layer) from another script is the same as update(self) here
            self.weights = regularizer.update(self)

        # $L = (y-x)^2 = ( f(y') - x)^2 $ so
        # dL/dx = dL/d[f(y')] d [f(y')]/dy' dy'/dx
        dLoss_dx = (dLoss_dy * dy_dyprime) @ self.weights.transpose()

        #now remove dropout nodes
        dLoss_dx[:,self.i_dropout] = 0
        
        #return everything except the last column, which is the bias term, which is always 1 (const) and not backpropagated
        return dLoss_dx[:, :-1]



#############################################################################################

class GenericLayer(object):
    
    def __ init__(self, previous_layer):
        self.previous_layer=previous_layer
        self.size = self.previous_layer.y.size
        #size is len(inputs of this layer) = len(outputs of previous layer)
        #intitialize the weights and reset he gradients, whenever first calling a GenericLayer
        self.reset()
    
    def reset(self):
        self.x = np.zeros((1,self.size))
        self.y =  np.zeros((1,self.size))
        self.dLoss_dx =  np.zeros((1,self.size))
        self.dLoss_dy =  np.zeros((1,self.size))
        
    def forward_propagate_layer(self, **kwargs):
        #kwargs so that any other arguments can be passed
        self.x  += self.previous_layer.y
        self.y = self.x
        
    self.backward_propagate_layer(self):
        #take the dLoss/dy that has been accumolated from forward pass
        self.dLoss_dx = self.dLoss_dy
        #make sure the previous layer's dLoss/dy gets incremented by this layers dLoss/dx
        self.previous_layer.dLoss_dy += self.dLoss_dx