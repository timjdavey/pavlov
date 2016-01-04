from .pavlov import Respondant
import pytest


def salivate(environment):
	pass

def eat(environment):
	pass

def bell_with_food(environment):
	environment['food_present'] = 1.0
	return 0.0, environment

def bell_without_food(environment):
	return 

@pytest.mark.experiment
def test_learns_salivate():
    pass

@pytest.mark.experiment
def test_learns_bell():
    pass
