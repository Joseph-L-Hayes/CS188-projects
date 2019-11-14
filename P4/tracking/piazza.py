def sample(self):
    "*** YOUR CODE HERE question 0 ***"
    #the probability that a key is sampled is proportional to its corresponding value.
    self.normalize()
    wild = random.random()
    sum = 0

    for key in self:
        sum += self[key]
        if sum >= wild:
            return key

def initializeUniformly(self, gameState):

    self.particles = []
    particleSize = int(self.numParticles / len(self.legalPositions))
    for pos in self.legalPositions:
        for i in range(particleSize):
            self.particles += [pos]


def observeUpdate(self, observation, gameState):
    
    "*** YOUR CODE HERE question 6 ***"
    particleBeliefs = DiscreteDistribution()
    pacmanPos = gameState.getPacmanPosition()
    jailPos = self.getJailPosition()
    # where the weight of a particle is the probability of the observation given Pacman’s
    #position and that particle location. Then, we resample from this weighted distribution to construct our new list of particles.

    for partPos in self.particles:
        prob = self.getObservationProb(observation, pacmanPos, partPos, jailPos)
        particleBeliefs[partPos] += prob

    if particleBeliefs.total() == 0:
        self.initializeUniformly(gameState)
    else:
        particleBeliefs.normalize()
        self.beliefs = particleBeliefs

        for i in range(len(particleBeliefs)):
            newDist = particleBeliefs.sample()
            self.particles[i] = newDist
