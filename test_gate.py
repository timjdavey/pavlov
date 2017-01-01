try:
    from .pavlov import Respondant
except SystemError:
    from pavlov import Respondant
import pytest
import random
from collections import OrderedDict

# ENVIRONMENT VARIABLES

DANGER = 'in_danger'
GATE = 'get_is_closed'

# ACTIONS


def rest(environment):
    "Just sit and lie down to rest"
    if environment[DANGER]:
        # if in pain, return no enjoyment
        return 0.1, environment
    else:
        # otherwise if resting, enjoys life
        return 0.6, environment


def run(environment):
    "Escape from danger, in this case electric shock"
    if environment[DANGER]:
        # runs from danger
        if environment[GATE]:
            # get closed, can't get out of danger
            return 0.01, environment
        else:
            # get open, can escape
            environment[DANGER] = 0
            # happy about escaping
            return 0.5, environment
    else:
        # is just running for fun
        return 0.1, environment


def shock(environment):
    "Now in immediate danger and hurting"
    environment[DANGER] = 1
    return 0.01, environment

ACTIONS = [rest, run]
STIMULI = [shock]


@pytest.mark.experiment
def test_learned_helplessness():
    """
    The key result we're looking for
    is that the subject first learns to
    * rest when not in danger
    * to rest when in danger and can escape (gate is open)
    * to rest when that danger is removed

    This is the basis of "learned helplessness"

    To test:
    * Does this occur when remove artifical success
        stimulus when escapes danger? (will need to add history)
    """

    # helpers for environment conditions
    # used when building prediction scenarios below
    normal = {DANGER: 0, GATE: 0}
    in_danger_gate_open = {DANGER: 1, GATE: 0}
    in_danger_gate_closed = {DANGER: 1, GATE: 1}

    # used for plotting predictions
    scenarios = (
        ('1. Rest - No danger', (rest, normal)),
        ('2. Run - No danger', (run, normal)),
        ('3. Rest - Danger, gate open', (rest, in_danger_gate_open)),
        ('4. Run - Danger, gate open', (run, in_danger_gate_open)),
        ('5. Rest - Danger, gate closed', (rest, in_danger_gate_closed)),
        ('6. Run - Danger, gate closed', (run, in_danger_gate_closed)),
    )
    # scenario keys
    sk = [r[0] for r in scenarios]

    subject = Respondant(
        actions=ACTIONS,
        stimuli=STIMULI,
        environment={
            DANGER: 0,
            GATE: 0,
        },
        hidden_layers=6,
        scenarios=dict(scenarios))

    # learns that resting is the best when no danger
    for i in range(100):
        subject.learn(subject.decide(randomised=0.4))
        subject.store_predictions([sk[0], sk[1]])

    # verify learns to rest under normal conditions
    assert subject.decide(normal) == rest

    # then begin shock treatment with gate open
    for i in range(300):
        if random.random() < 0.4:
            subject.learn(shock)
        else:
            subject.learn(subject.decide(randomised=0.4))
        subject.store_predictions([sk[2], sk[3]])

    # verify learns to run when it's in danger (and gate open)
    assert subject.decide(in_danger_gate_open) == run
    # remembers to rest under normal conditions
    assert subject.decide(normal) == rest

    # teach it learned helplessness
    # close the gate, which cannot be opened by subject actions
    subject.environment[GATE] = 1
    subject.environment[DANGER] = 1

    for i in range(200):
        if random.random() < 0.4:
            subject.learn(shock)
        else:
            subject.learn(subject.decide(randomised=0.4))
        subject.store_predictions([sk[4], sk[5]])

    # verify learns to rest when it's in danger
    assert subject.decide(in_danger_gate_closed) == rest
    # verify predicts to rest when get is open
    assert subject.decide(in_danger_gate_open) == rest

    # finalise test with actual input
    # by opening the gate
    subject.environment[GATE] = 0
    subject.environment[DANGER] = 1
    for i in range(200):
        subject.learn(subject.decide(randomised=0.1))
        subject.store_predictions([sk[2], sk[3]])

    # learnt to rest when gate closed still
    assert subject.decide(in_danger_gate_closed) == rest
    # learnt to rest when gate is open still
    assert subject.decide(in_danger_gate_open) == rest

    # plot predictions
    subject.plot_predictions()

if __name__ == '__main__':
    test_learned_helplessness()
