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
        "*** YOUR CODE HERE question 1 ***"
        score = nn.DotProduct(self.w, x)

        return score


    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE question 1 ***"
        dProduct = nn.as_scalar(self.run(x))

        if dProduct < 0:
            return -1
        else:
            return 1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE question 1 ***"
        while True:
            trainingComplete = True
            data = dataset.iterate_once(1)

            for feature, label in data:

                if nn.as_scalar(label) != self.get_prediction(feature):
                    self.w.update(feature, nn.as_scalar(label))
                    trainingComplete = False

            if trainingComplete:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        # Use nn.SquareLoss as your loss
        # ReLU: rectified linear unit, f(x) = 0 if x < 0 and x otherwise
        #functions to use:
            #nn.Linear(features, weights)
                #returns a node with shape (batch_size x output_features)

            #nn.ReLU(x),
                #x: a Node with shape (batch_size x num_features)
                #returns a Node with the same shape as x, but no negative entries

            #nn.AddBias(features, bias)
                #features: a Node with shape (batch_size x num_features)
                #bias: a Node with shape (1 x num_features)
                #returns a Node with shape (batch_size x num_features)

            #nn.Linear (again)
        "*** YOUR CODE HERE question 2 ***"
        self.learnRate = -.002 #best so far -.002
        self.batch_size = 1 #best so far 1
        d = 1
        h = 1000 #best so far 1000
        #error(10000, -.002) = 0.011310
        #error(1000, -.002) = 0.009555
        #error(100, -.002) = 0.012887

        self.m0 = nn.Parameter(d, h)
        self.b0 = nn.Parameter(1, h)
        self.m1 = nn.Parameter(h, 1)
        self.b1 = nn.Parameter(1, 1)

        # Ignore below, used for extra layer, needs tuning.
        self.m2 = nn.Parameter(1, 1)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE question 2 ***"
        xm0 = nn.Linear(x, self.m0)
        bias0 = nn.AddBias(xm0, self.b0)
        relu0 = nn.ReLU(bias0)
        xm1 = nn.Linear(relu0, self.m1)
        bias1 = nn.AddBias(xm1, self.b1)
        ### extra layer below ###
        # relu1 = nn.ReLU(bias1)
        # xm2 = nn.Linear(relu1, self.m2)
        # final = nn.AddBias(xm2, self.b2)
        # training another layer makes a smoother curve but needs more tuning...

        return bias1

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE question 2 ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE question 2 ***"
        while True:
            for feature, label in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(feature, label)
                weightList = [self.m0, self.b0, self.m1, self.m2, self.b2]
                grad_wrt_m0, grad_wrt_b0, grad_wrt_m1, grad_wrt_m2, grad_wrt_b2 = nn.gradients(loss, weightList)

                self.m0.update(grad_wrt_m0, self.learnRate)
                self.b0.update(grad_wrt_b0, self.learnRate)
                self.m1.update(grad_wrt_m1, self.learnRate)
                # self.m2.update(grad_wrt_m2, self.learnRate)
                # self.b2.update(grad_wrt_b2, self.learnRate)

            if nn.as_scalar(self.get_loss(feature, label)) <= .02:
                break

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
        "*** YOUR CODE HERE question 3 ***"
        self.learnRate = -.009 # -.009 best
        self.batch_size = 20 #20 best
        d = 784
        h = 100 #100 best

        #validation accuracy(-.009, 20, d=784, h=100) = 97.2 e=13;  97.4 e=20; passed BEST

        self.m0 = nn.Parameter(d, h)
        self.b0 = nn.Parameter(1, h)
        self.m1 = nn.Parameter(h, d)
        self.b1 = nn.Parameter(1, d)
        self.m2 = nn.Parameter(d, 10)
        self.b2 = nn.Parameter(1, 10)

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
        "*** YOUR CODE HERE question 3 ***"
        xm0 = nn.Linear(x, self.m0) # 1x784 * 784x10 = 1x10
        bias0 = nn.AddBias(xm0, self.b0) #1x10 + 1x10 = 1x10
        relu0 = nn.ReLU(bias0) #1x10 relu = 1x10
        xm1 = nn.Linear(relu0, self.m1) #1x10 * 10x784 = 1x784
        bias1 = nn.AddBias(xm1, self.b1) #xm1, b1 1x784 + 1x784 = 1x784
        xm2 = nn.Linear(bias1, self.m2) # 1x784 * 784x10 = 1x10
        final = nn.AddBias(xm2, self.b2) #1x10 + 1x10 = 1x10

        return final

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
        "*** YOUR CODE HERE question 3 ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE question 3 ***"
        # epoch = 1

        while True:
            for feature, label in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(feature, label)
                weightList = [self.m0, self.b0, self.m1, self.m2, self.b2]
                grad_wrt_m0, grad_wrt_b0, grad_wrt_m1, grad_wrt_m2, grad_wrt_b2 = nn.gradients(loss, weightList)

                self.m0.update(grad_wrt_m0, self.learnRate)
                self.b0.update(grad_wrt_b0, self.learnRate)
                self.m1.update(grad_wrt_m1, self.learnRate)
                self.m2.update(grad_wrt_m2, self.learnRate)
                self.b2.update(grad_wrt_b2, self.learnRate)

            # print("EPOCH:", epoch)
            # print("ACC:", dataset.get_validation_accuracy())
            # epoch += 1

            if dataset.get_validation_accuracy() >= .972:
                return
            #
            # if epoch == 13:
            #     break

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
        "*** YOUR CODE HERE question 4 ***"
        self.batch_size = 5
        self.learn_rate = -.005
        self.h = 200
        self.d = len(self.languages)

        self.w = nn.Parameter(self.num_chars, self.h)
        self.w_hidden1 = nn.Parameter(self.h, self.h)
        self.w_hidden2 = nn.Parameter(self.h, self.d)

        #batch = 10, learn = -.005, h = 5: 72%
        #batch = 10, learn = -.005, h=100: 75%
        #batch = 10, learn = -.002, h=50: 75%
        #batch = 10, learn = -.009, h=50: 77%
        #batch = 15, learn = -.005, h=50: 75
        #batch = 10, learn = -.005, h=40: 77%
        #batch = 20, learn = -.005, h=50: 76%
        #batch = 10, learn = -.005, h=10: 74%

        #batch = 15, learn = -.005, h=50: 77%
        #batch = 5, learn= -.005, h=50: 79%
        #batch = 5, learn= -.009, h=50: 74%
        #batch = 5, learn= -.005, h=50: 79% (second run): 78.4% best
        #batch = 5, learn= -.004, h=50: 76

        #batch = 10, learn = -.005, h=50: 78% (second run): 78% best, 3: 77%
        #batch = 10, learn = -.005, h=200: 76%
        #batch = 5, learn = -.005, h=200: abort, too long
        #** added ReLU to for loop in run function **
        #batch = 5, learn = -.005, h=200: 83% e=?, passed!
        #batch = 10, learn = -.005, h=50: (rerun best with ReLU): 81%
        #batch = 10, learn = -.005, h=200: 79%
        #batch = 5, learn = -.005, h=200: 86% e=8, passed!
        #batch = 5, learn = -.009, h=200: 83%, 82, 86
        #batch = 5, learn = -.002, h=200: 78%
        #batch = 5, learn = -.005, h=400: 83%

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
        "*** YOUR CODE HERE question 4 ***"
        # h1=finitial(x0),
        # print("XS", xs[0]) #1x47
        # h1 = nn.Linear(xs[0]) #output vector
        #Next, we’ll combine the output of the previous step with the next letter
        #in the word, generating a vector summary of the the first two letters of the word.
        # h2 = nn.Linear(h1, xs[1])
        #This pattern continues for all letters in the input word, where the hidden
        #state at each step summarizes all the letters the network has processed thus far:
        z = nn.Linear(xs[0], self.w)
        for x in xs:
            z = nn.Add(nn.Linear(x, self.w), nn.Linear(z, self.w_hidden1))
            z = nn.ReLU(z) #adding nonlinearity resulted in a huge performance increase

        return nn.Linear(z, self.w_hidden2)

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
        "*** YOUR CODE HERE question 4 ***"
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE question 4 ***"
        epoch = 0
        while True:
            epoch += 1

            for feature, label in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(feature,label)
                weightList = [self.w, self.w_hidden1, self.w_hidden2]
                grad_wrt_w, grad_wrt_hidden1, grad_wrt_hidden2 = nn.gradients(loss, weightList)

                self.w.update(grad_wrt_w, self.learn_rate)
                self.w_hidden1.update(grad_wrt_hidden1, self.learn_rate)
                self.w_hidden2.update(grad_wrt_hidden2, self.learn_rate)

            if dataset.get_validation_accuracy() >= .85:
                return

            if epoch == 50:
                break
