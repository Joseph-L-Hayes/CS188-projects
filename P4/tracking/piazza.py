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


"""Question q6
===========
dict_items([((8, 4), 0.125), ((3, 4), 0.125), ((7, 4), 0.125), ((5, 4), 0.125), ((1, 4), 0.125), ((2, 4), 0.125), ((5, 2), 0.125), ((9, 4), 0.125), ((1, 1), 0.0)])
dict_items([((5, 4), 0.222222222222), ((5, 2), 0.0), ((1, 4), 0.0555555555556), ((2, 4), 0.111111111111), ((7, 4), 0.222222222222), ((9, 4), 0.0555555555556), ((3, 4), 0.222222222222), ((1, 1), 0.0), ((8, 4), 0.111111111111)])
*** q6) Distribution deviated at move 1 by 0.0540 (squared norm) from the correct answer.
      key:     student                  reference
   (5, 4):     0.125                    0.222222222222
   (5, 2):     0.125                    0.0
   (1, 4):     0.125                    0.0555555555556
   (7, 4):     0.125                    0.222222222222
   (1, 1):     0.0                      0.0
   (9, 4):     0.125                    0.0555555555556
   (3, 4):     0.125                    0.222222222222
   (2, 4):     0.125                    0.111111111111
   (8, 4):     0.125                    0.111111111111
*** q6) Particle filter observe test: 51 inference errors.
*** FAIL: test_cases/q6/1-ParticleUpdate.test
dict_items([((9, 9), 0.07142857142857142), ((1, 3), 0.07142857142857142), ((9, 3), 0.07142857142857142), ((2, 3), 0.07142857142857142), ((2, 9), 0.07142857142857142), ((1, 9), 0.07142857142857142), ((1, 4), 0.07142857142857142), ((1, 8), 0.07142857142857142), ((2, 4), 0.07142857142857142), ((2, 8), 0.07142857142857142), ((5, 6), 0.07142857142857142), ((8, 8), 0.07142857142857142), ((8, 9), 0.07142857142857142), ((9, 8), 0.07142857142857142), ((1, 1), 0.0)])
dict_items([((1, 3), 0.208469055375), ((2, 9), 0.0260586319218), ((5, 6), 0.0), ((2, 8), 0.00325732899023), ((9, 8), 0.0260586319218), ((9, 3), 0.208469055375), ((9, 9), 0.208469055375), ((1, 4), 0.0260586319218), ((2, 4), 0.00325732899023), ((8, 9), 0.0260586319218), ((1, 8), 0.0260586319218), ((8, 8), 0.00325732899023), ((2, 3), 0.0260586319218), ((1, 9), 0.208469055375), ((1, 1), 0.0)])
*** q6) Distribution deviated at move 5 by 0.1065 (squared norm) from the correct answer.
      key:     student                  reference
   (1, 3):     0.07142857142857142      0.208469055375
   (2, 9):     0.07142857142857142      0.0260586319218
   (5, 6):     0.07142857142857142      0.0
   (2, 8):     0.07142857142857142      0.00325732899023
   (9, 8):     0.07142857142857142      0.0260586319218
   (9, 3):     0.07142857142857142      0.208469055375
   (9, 9):     0.07142857142857142      0.208469055375
   (1, 4):     0.07142857142857142      0.0260586319218
   (8, 9):     0.07142857142857142      0.0260586319218
   (1, 8):     0.07142857142857142      0.0260586319218
   (8, 8):     0.07142857142857142      0.00325732899023
   (2, 3):     0.07142857142857142      0.0260586319218
   (1, 9):     0.07142857142857142      0.208469055375
   (1, 1):     0.0                      0.0
   (2, 4):     0.07142857142857142      0.00325732899023
*** q6) Particle filter observe test: 84 inference errors.
*** FAIL: test_cases/q6/2-ParticleUpdate.test
dict_items([((8, 9), 0.045454545454545456), ((5, 5), 0.045454545454545456), ((6, 6), 0.045454545454545456), ((5, 6), 0.045454545454545456), ((6, 7), 0.045454545454545456), ((4, 6), 0.045454545454545456), ((9, 8), 0.045454545454545456), ((8, 8), 0.045454545454545456), ((4, 7), 0.045454545454545456), ((5, 7), 0.045454545454545456), ((2, 4), 0.045454545454545456), ((1, 3), 0.045454545454545456), ((1, 4), 0.045454545454545456), ((1, 8), 0.045454545454545456), ((1, 9), 0.045454545454545456), ((2, 3), 0.045454545454545456), ((2, 8), 0.045454545454545456), ((2, 9), 0.045454545454545456), ((4, 5), 0.045454545454545456), ((6, 5), 0.045454545454545456), ((9, 3), 0.045454545454545456), ((9, 9), 0.045454545454545456), ((1, 1), 0.0)])
dict_items([((4, 7), 0.000152973182823), ((1, 3), 0.0783222696052), ((6, 6), 0.0), ((5, 6), 0.0), ((2, 8), 0.0195805674013), ((9, 8), 0.0391611348026), ((8, 9), 0.0391611348026), ((6, 7), 3.5853089724e-06), ((5, 5), 0.0), ((2, 9), 0.00979028370064), ((1, 1), 0.0), ((4, 5), 1.15775602234e-06), ((9, 3), 0.313289078421), ((1, 4), 0.15664453921), ((2, 3), 0.15664453921), ((1, 9), 0.00122378546258), ((6, 5), 0.0), ((4, 6), 4.78041196321e-06), ((5, 7), 4.78041196321e-06), ((9, 9), 0.0783222696052), ((1, 8), 0.00979028370064), ((8, 8), 0.0195805674013), ((2, 4), 0.0783222696052)])
*** q6) Distribution deviated at move 7 by 0.1242 (squared norm) from the correct answer.
      key:     student                  reference
   (4, 7):     0.045454545454545456     0.000152973182823
   (1, 3):     0.045454545454545456     0.0783222696052
   (6, 6):     0.045454545454545456     0.0
   (5, 6):     0.045454545454545456     0.0
   (2, 8):     0.045454545454545456     0.0195805674013
   (9, 8):     0.045454545454545456     0.0391611348026
   (8, 9):     0.045454545454545456     0.0391611348026
   (6, 7):     0.045454545454545456     3.5853089724e-06
   (5, 5):     0.045454545454545456     0.0
   (2, 9):     0.045454545454545456     0.00979028370064
   (1, 1):     0.0                      0.0
   (4, 5):     0.045454545454545456     1.15775602234e-06
   (9, 3):     0.045454545454545456     0.313289078421
   (1, 4):     0.045454545454545456     0.15664453921
   (2, 3):     0.045454545454545456     0.15664453921
   (1, 9):     0.045454545454545456     0.00122378546258
   (6, 5):     0.045454545454545456     0.0
   (4, 6):     0.045454545454545456     4.78041196321e-06
   (5, 7):     0.045454545454545456     4.78041196321e-06
   (9, 9):     0.045454545454545456     0.0783222696052
   (1, 8):     0.045454545454545456     0.00979028370064
   (8, 8):     0.045454545454545456     0.0195805674013
   (2, 4):     0.045454545454545456     0.0783222696052
*** q6) Particle filter observe test: 95 inference errors.
*** FAIL: test_cases/q6/3-ParticleUpdate.test
dict_items([((4, 3), 0.25), ((2, 3), 0.25), ((1, 3), 0.25), ((3, 3), 0.25), ((1, 1), 0.0)])
dict_items([((1, 3), 0.0), ((2, 3), 0.244094488189), ((3, 3), 0.251968503937), ((4, 3), 0.503937007874), ((1, 1), 0.0)])
*** q6) Distribution deviated at move 1 by 0.1270 (squared norm) from the correct answer.
      key:     student                  reference
   (1, 3):     0.25                     0.0
   (3, 3):     0.25                     0.251968503937
   (2, 3):     0.25                     0.244094488189
   (4, 3):     0.25                     0.503937007874
   (1, 1):     0.0                      0.0
*** q6) Particle filter observe test: 5 inference errors.
*** FAIL: test_cases/q6/4-ParticleUpdate.test
*** q6) error handling all weights = 0
*** FAIL: test_cases/q6/5-ParticleUpdate.test
ParticleFilter
[Distancer]: Switching to maze distances
Average Score: 182.5
Scores:        195, 174, 195, 144, 196, 193, 195, 157, 198, 178
Win Rate:      10/10 (1.00)
Record:        Win, Win, Win, Win, Win, Win, Win, Win, Win, Win
*** Won 10 out of 10 games. Average score: 182.500000 ***
*** oneHunt) Games won on q6 with score above 100: 10/10
*** PASS: test_cases/q6/6-ParticleUpdate.test
*** Tests failed.

### Question q6: 0/3 ###


Finished at 8:58:04

Provisional grades
==================
Question q6: 0/3
------------------
Total: 0/3 """
