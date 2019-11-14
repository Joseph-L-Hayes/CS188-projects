# analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.9
    answerNoise = 0 #why was this a question? Is this optimal answer?
    return answerDiscount, answerNoise

def question3a():
    answerDiscount = 0.31
    answerNoise = 0
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # Prefer the close exit (+1), risking the cliff (-10)
    #LOGIC: lower noise to 0 so it won't be so fearful of the pit and a low enough
    #discount that exit 1 gives enough points.

def question3b():
    answerDiscount = 0.3
    answerNoise = 0.2
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # Prefer the close exit (+1), but avoiding the cliff (-10)
    #LOGIC: give enough discount so that it can't wait for exit 10 and enough noise
    #so it avoids the pits when it's close but not when it is far away.

def question3c():
    answerDiscount = 0.9
    answerNoise = .05
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # Prefer the distant exit (+10), risking the cliff (-10)
    #LOGIC: discount high enough that reward from exit 1 isn't low and no noise so
    #agent won't avoid the pits.

def question3d():
    answerDiscount = 0.9
    answerNoise = 0.1
    answerLivingReward = .1
    return answerDiscount, answerNoise, answerLivingReward
    #LOGIC: Low discount
    # Prefer the distant exit (+10), avoiding the cliff (-10)

def question3e():
    answerDiscount = 1
    answerNoise = .9
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # Avoid both exits and the cliff (so an episode should never terminate)
    #LOGIC: high noise so won't risk the pits even with no discount

def question8():
    answerEpsilon = .5
    answerLearningRate = .5
    # return answerEpsilon, answerLearningRate
    return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
