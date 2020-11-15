import numpy as np

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden, and output layers
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # initialize the weights
        self.weights_input_to_hidden = np.random.normal(0.0,
                                        self.input_nodes**-0.5,
                                        (self.input_nodes, self.hidden_nodes)

        self.weights_hidden_to_output = np.random_normal(0.0,
                                        self.hidden_nodes**-0.5,
                                        (self.hidden_nodes, self.output_nodes))

        self.lr = learning_rate

        #### TODO: Set self.activation_function to your implemented sigmoid
        ####       function
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        self.activation_function = lambda x: 1/(1+np.exp(-x))

    def train(self, features, targets):
        ''' Train the network on batch of features and targets.

            Arguments
            ---------

            features: 2D array, each row is one data record, each column is a
                        feature
            targets: 1D array of target values

        '''
        n_records = features.shape[0] # note .shape(num_rows, num_cols)
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):

            # Implement the forward pass function below
            final_outputs, hidden_outputs = self.forward_pass_train(X)

            # Implement the backpropagation function below
            delta_weights_i_h, delta_weights_h_o = \
                                        self.backpropagation(final_outputs,
                                        hidden_outputs, X, y,
                                        delta_weights_i_h, delta_weights_h_o)

        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def forward_pass_train(self, X):
        ''' Implement forward pass here

            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer = Replace these values with your calculations.

        # Signals into hidden layer
        hidden_inputs = X.dot(weights_input_to_hidden)

        # Signals from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # TODO: Output layer - Replace these values with your calculations.

        # Signals into final output layer
        final_inputs = hidden_outputs.dot(weights_hidden_to_output)

        # Signals from final output layer
        final_outputs = final_inputs

        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y,
                        delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation

        Arguments
        ---------
        final_outputs: output from forward pass
        y: target (i.e. label) batch
        delta_weights_i_h: change in weights from inputs to hidden layers
        delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO: Output error - Replace this value with your calculations

        # Output layer error is the difference between desired target and
        # actual output
        error = y - final_outputs

        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = self.weights_hidden_to_output*error

        # TODO: Backpropagated error terms - Replace these values with your
        #       calculations
        output_error_term = error

        hidden_error_term = hidden_error*hidden_outputs*(1-hidden_outputs)

        # Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term*X
        # Alternative is hidden_error_term[None, :]*x[:, None]

        # Weight_step (hidden to output)
        delta_weights_h_o += output_error_term*hidden_outputs

        return delta_weights_i_h, delta_weights_h_o
