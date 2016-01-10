from .pavlov import Respondant
import pytest

FOOD_KEY = 'food_present'
SALIVATION_KEY = 'saliver_levels'


def salivate(environment):
    "Salivates"
    return 0.0, environment


def rest(environment):
    "Just sits and rests"
    return


def eat(environment):
    "Eats food and is the best thing ever!"
    environment[FOOD_KEY] = 0.0
    return 1.0, environment


def bell_with_food(environment):
    "Food is given when bell is rung"
    environment[FOOD_KEY] = 1.0
    return 0.0, environment


def bell_without_food(environment):
    "No food given or taken away"
    return 0.0, environment


@pytest.mark.experiment
def test_learns_eat():
    subject = Respondant(actions=[eat])


@pytest.mark.experiment
def test_learns_salivate():
    pass


@pytest.mark.experiment
def test_learns_bell():
    pass
