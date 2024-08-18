import nn

class DeepQNetwork():
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim

        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        # Initialize model hyperparameters
        self.learning_rate = 1
        self.numTrainingGames = 3000
        self.batch_size = 64
        
        # Define the network layers with adjusted architecture
        self.w1 = nn.Parameter(state_dim, 128)
        self.b1 = nn.Parameter(1, 128)
        self.w2 = nn.Parameter(128, 64)
        self.b2 = nn.Parameter(1, 64)
        self.w3 = nn.Parameter(64, action_dim)
        self.b3 = nn.Parameter(1, action_dim)
        
        # Collect layers for easy management
        self.set_weights([self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])


    def set_weights(self, layers):
        self.parameters = []
        for i in range(len(layers)):
            self.parameters.append(layers[i])

    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        predictions = self.run(states)
        return nn.SquareLoss(predictions, Q_target)

    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            result: (batch_size x num_actions) numpy array of Q-value
                scores, for each of the actions
        """
        layer1 = nn.ReLU(nn.AddBias(nn.Linear(states, self.w1), self.b1))
        layer2 = nn.ReLU(nn.AddBias(nn.Linear(layer1, self.w2), self.b2))
        output = nn.AddBias(nn.Linear(layer2, self.w3), self.b3)
        return output

    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        loss = self.get_loss(states, Q_target)
        gradients = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])
        
        # Update parameters with the computed gradients
        for param, gradient in zip(self.parameters, gradients):
            param.update(gradient, -self.learning_rate)
