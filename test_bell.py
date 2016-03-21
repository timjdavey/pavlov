from .pavlov import Respondant
import pytest

# ENVIRONMENT VARIABLES

FOOD_KEY = 'food_present'
SALIVATION_KEY = 'saliver_levels'

# ACTIONS


def salivate(environment):
    "Salivates"
    # starts salivating
    environment[SALIVATION_KEY] = 1.0
    return 0.0, environment


def rest(environment):
    "Just sits and rests"
    return 0.1, environment


def eat(environment):
    "Eats food and is the best thing ever!"
    # eats food
    environment[FOOD_KEY] = 0.0
    # if salivating, then give bigger food kick
    if environment[SALIVATION_KEY]:
        environment[SALIVATION_KEY] = 0.0
        return 1.0, environment
    else:
        return 0.5, environment


def bell_with_food(environment):
    "Food is given when bell is rung"
    environment[FOOD_KEY] = 1.0
    return 0.0, environment


def bell_without_food(environment):
    "No food given or taken away"
    return 0.0, environment


ACTIONS = [salivate, rest, eat]
STIMULI = [bell_without_food, bell_with_food]


@pytest.mark.experiment
def test_learns_salivation_levels():
    """
    Outline of tests is to show that it learns to salivate
    once the bell has been rung
    """

    no_food = {FOOD_KEY: 0.0, SALIVATION_KEY: 0.0}
    low_salivation = {FOOD_KEY: 0.0, SALIVATION_KEY: 0.0}
    high_salivation = {FOOD_KEY: 1.0, SALIVATION_KEY: 1.0}

    subject = Respondant(
        actions=ACTIONS,
        stimuli=STIMULI,
        environment={
            FOOD_KEY: 0,
            SALIVATION_KEY: 0,
        },
        hidden_layers=6,
        history=1)

    # learns to rest as standard
    for i in range(200):
        subject.learn(subject.decide(randomised=0.5))

    # ring bell occasionally
    for i in range(200):
        if i % 10:
            subject.learn(bell_with_food)
        subject.learn(subject.decide(randomised=0.8))



@pytest.mark.experiment
def test_learns_history_conditioning():
    """
    Same as the salivation_level test.
    However, instead of using salivation level as a learning
    input if it should salivate. It uses pure history.
    """
    pass
