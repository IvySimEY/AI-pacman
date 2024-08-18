import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(x, self.get_weights())


    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        return 1 if nn.as_scalar(self.run(x)) >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        batch_size = 1
        for _ in range(1000):
            misclassified = 0
            for x, y in dataset.iterate_once(batch_size):
                if self.get_prediction(x) != nn.as_scalar(y):
                    self.w.update(x, nn.as_scalar(y))
                    misclassified += 1
            if misclassified == 0:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Increase the number of neurons
        num_neurons = 50  # Increased from 10 to 50

        # Initialize weights and biases for the first hidden layer
        self.w1 = nn.Parameter(1, num_neurons)
        self.b1 = nn.Parameter(1, num_neurons)
        
        # Initialize weights and biases for the second hidden layer
        self.w2 = nn.Parameter(num_neurons, num_neurons)
        self.b2 = nn.Parameter(1, num_neurons)
        
        # Initialize weights and biases for the output layer
        self.w3 = nn.Parameter(num_neurons, 1)
        self.b3 = nn.Parameter(1, 1)


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        layer_1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        layer_2 = nn.ReLU(nn.AddBias(nn.Linear(layer_1, self.w2), self.b2))
        output = nn.AddBias(nn.Linear(layer_2, self.w3), self.b3)
        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        learning_rate = 0.01  # Consider using a learning rate schedule
        epoch = 0
        while True:
            total_loss = 0
            num_batches = 0
            for x, y in dataset.iterate_once(50):  # Adjusted batch size to 50
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])
                self.w1.update(gradients[0], -learning_rate)
                self.b1.update(gradients[1], -learning_rate)
                self.w2.update(gradients[2], -learning_rate)
                self.b2.update(gradients[3], -learning_rate)
                self.w3.update(gradients[4], -learning_rate)
                self.b3.update(gradients[5], -learning_rate)
                total_loss += nn.as_scalar(loss)
                num_batches += 1

            average_loss = total_loss / num_batches
            if average_loss < 0.02:
                print(f"Converged at epoch {epoch + 1}")
                break
            epoch += 1
            if epoch % 10 == 0:  # Print loss every 10 epochs to monitor progress
                print(f"Epoch {epoch}, Average Loss: {average_loss}")


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        # Increase complexity: larger layer sizes and one additional layer
        self.w1 = nn.Parameter(784, 256)  # Increased from 128 to 256
        self.b1 = nn.Parameter(1, 256)
        self.w2 = nn.Parameter(256, 128)  # Increased from 64 to 128
        self.b2 = nn.Parameter(1, 128)
        self.w3 = nn.Parameter(128, 64)   # New layer
        self.b3 = nn.Parameter(1, 64)
        self.w4 = nn.Parameter(64, 10)    # Output layer
        self.b4 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        hidden1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        hidden2 = nn.ReLU(nn.AddBias(nn.Linear(hidden1, self.w2), self.b2))
        hidden3 = nn.ReLU(nn.AddBias(nn.Linear(hidden2, self.w3), self.b3))
        output = nn.AddBias(nn.Linear(hidden3, self.w4), self.b4)
        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        logits = self.run(x)
        return nn.SoftmaxLoss(logits, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        base_learning_rate = 0.1
        for epoch in range(20):
            learning_rate = base_learning_rate / (1 + 0.1 * epoch)  # Decaying learning rate
            for x, y in dataset.iterate_once(100):
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4])
                self.w1.update(gradients[0], -learning_rate)
                self.b1.update(gradients[1], -learning_rate)
                self.w2.update(gradients[2], -learning_rate)
                self.b2.update(gradients[3], -learning_rate)
                self.w3.update(gradients[4], -learning_rate)
                self.b3.update(gradients[5], -learning_rate)
                self.w4.update(gradients[6], -learning_rate)
                self.b4.update(gradients[7], -learning_rate)

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        hidden_size = 128  # Choose a reasonable size for the hidden layer

        # Parameters for initial transformation
        self.w_initial = nn.Parameter(self.num_chars, hidden_size)

        # Parameters for the recurrent transformation
        self.w = nn.Parameter(self.num_chars, hidden_size)
        self.w_hidden = nn.Parameter(hidden_size, hidden_size)
        self.b = nn.Parameter(1, hidden_size)

        # Output layer parameters
        self.w_output = nn.Parameter(hidden_size, len(self.languages))
        self.b_output = nn.Parameter(1, len(self.languages))

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        # Start with an initial hidden state h
        h = nn.Linear(xs[0], self.w_initial)  # Only for the first character
        
        # Process each character in the word
        for i in range(1, len(xs)):
            h = nn.Add(nn.Linear(xs[i], self.w), nn.Linear(h, self.w_hidden))
            h = nn.ReLU(nn.AddBias(h, self.b))  # Apply ReLU after combining with bias
        
        # Compute the final scores for each language
        output = nn.AddBias(nn.Linear(h, self.w_output), self.b_output)
        return output

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        logits = self.run(xs)
        return nn.SoftmaxLoss(logits, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        initial_learning_rate = 0.1
        decay_rate = 0.99  # A decay rate to slightly reduce the learning rate each epoch
        min_learning_rate = 0.001  # A floor for the learning rate

        for epoch in range(20):  # You can adjust this based on when you reach >85% validation accuracy
            learning_rate = max(initial_learning_rate * (decay_rate ** epoch), min_learning_rate)
            for xs, y in dataset.iterate_once(150):  # A batch size between 100 and 200 as suggested
                loss = self.get_loss(xs, y)
                gradients = nn.gradients(loss, [self.w_initial, self.w, self.w_hidden, self.b, self.w_output, self.b_output])
                for param, gradient in zip([self.w_initial, self.w, self.w_hidden, self.b, self.w_output, self.b_output], gradients):
                    param.update(gradient, -learning_rate)
                
            val_accuracy = dataset.get_validation_accuracy()
            if val_accuracy > 0.85:  # Stop after reaching >85% accuracy as suggested
                print(f"Stopping early at epoch {epoch} with validation accuracy: {val_accuracy}")
                break
