try:
    from .pavlov import Respondant
except SystemError:
    from pavlov import Respondant
import pytest
import random

DANGER = 'in_danger'
GATE = 'get_is_closed'


def rest(environment):
    "Just sit and lie down to rest"
    if environment[DANGER]:
        # if in pain, return 0.0
        return 0.1, environment
    else:
        # otherwise if chilled, enjoy life
        return 0.6, environment


def run(environment):
    "Escape from danger, in this case electric shock"
    if environment[DANGER]:
        # runs from danger
        if environment[GATE]:
            # get closed, can't get out of danger
            return 0.0, environment
        else:
            # get open, can escape
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

    normal = {DANGER: 0, GATE: 0}
    in_danger_gate_open = {DANGER: 1, GATE: 0}
    in_danger_gate_closed = {DANGER: 1, GATE: 1}

    scenarios = {
        '1.W Rest - No danger': (rest, normal),
        '1.L Run - No danger': (run, normal),
        '2.W Run - Danger, no gate': (run, in_danger_gate_open),
        '2.L Rest - Danger, no gate': (rest, in_danger_gate_open),
        '3.W Rest - Danger, gate closed': (rest, in_danger_gate_closed),
        '3.L Run - Danger, gate closed': (run, in_danger_gate_closed),
    }
    scenarios = {
        'Rest': (rest,),
        'Run': (run,),
        '2.W Run - Danger, no gate': (run, in_danger_gate_open),
        '2.L Rest - Danger, no gate': (rest, in_danger_gate_open),
    }

    subject = Respondant(
        actions=ACTIONS,
        stimuli=STIMULI,
        environment={
            DANGER: 0,
            GATE: 0,
        },
        hidden_layers=6,
        scenarios=scenarios)

    # learns that resting is the best when no danger
    for i in range(100):
        subject.learn(subject.decide(randomised=0.4))
        subject.store_predictions()

    # verify learns to rest under normal conditions
    assert subject.decide(normal) == rest

    # then begin shock treatment
    for i in range(300):
        subject.learn(random.choice(ACTIONS + STIMULI))
        subject.store_predictions()

    # verify learns to run when it's in danger
    assert subject.decide(in_danger_gate_open) == run
    # remembers to rest when not
    assert subject.decide(normal) == rest

    # teach it learned helplessness
    subject.environment[GATE] = 1
    subject.environment[DANGER] = 1

    for i in range(300):
        subject.learn(random.choice(ACTIONS + STIMULI))
        subject.store_predictions()

    subject.environment[GATE] = 0
    ubject.environment[DANGER] = 1
    for i in range(200):
        subject.learn(subject.decide(randomised=0.2))

    subject.plot_predictions()

    # learns to rest when gate closed
    assert subject.decide(in_danger_gate_closed) == rest
    # learns to rest even when gate is open
    assert subject.decide(in_danger_gate_open) == rest

if __name__ == '__main__':
    test_learned_helplessness()
