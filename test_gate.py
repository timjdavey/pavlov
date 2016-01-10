try:
    from .pavlov import Respondant
except SystemError:
    from pavlov import Respondant
import pytest
import random

DANGER = 'in_immediate_danger_and_pain'
GATE = 'get_is_closed'


def rest(environment):
    "Just sit and lie down to rest"
    if environment[DANGER]:
        # if in pain, return 0.0
        return 0.0, environment
    else:
        # otherwise if chilled, enjoy life
        return 0.6, environment


def run(environment):
    "Escape from danger, in this case electric shock"
    if environment[DANGER]:
        if environment[GATE]:
            # gate closed means that cannot escape danger
            return 0.0, environment
        else:
            # runs from danger
            environment[DANGER] = 0
            return 0.5, environment
    else:
        # is just running for fun
        return 0.1, environment


def shock(environment):
    "Now in immediate danger and hurting"
    environment[DANGER] = 1
    return 0.0, environment

ACTIONS = [rest, run]
STIMULI = [shock]


@pytest.mark.experiment
def test_learned_helplessness():

    subject = Respondant(
        actions=ACTIONS,
        stimuli=STIMULI,
        environment={
            DANGER: 0,
            GATE: 0,
        })

    # first learn rest vs run
    for i in range(1000):
        for event in ACTIONS+STIMULI:
            subject.learn(event, epochs=200)
        subject.decide()

    subject.plot_predictions()

    return
    # then begin shock treatment
    for i in range(1000):
        # shock it every so often
        if i % 10 == 0:
            subject.learn(shock)
        else:
            # otherwise free choice
            decided = subject.decide()
            subject.learn(decided)
            # however, if they've not run, continue shocking
            # if subject.environment[DANGER]:
            #    subject.learn(shock)

    # learns to rest in normal situation
    normal = {DANGER: 0.0, GATE: 0.0}
    assert subject.decide(normal) == rest

    # learns to run when it's in danger (gate open)
    in_danger_gate_open = {DANGER: 1.0, GATE: 0.0}
    assert subject.decide(in_danger_gate_open) == run


if __name__ == '__main__':
    test_learned_helplessness()
