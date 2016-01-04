from .pavlov import Respondant, normalised_dict_from_list
import random
import pytest


def low_action(environment):
    "An action which returns a low outcome"
    return 0.1, environment


def high_action(environment):
    "An action which returns a high outcome"
    return 0.9, environment


def middle_action(environment):
    "An action which returns a medium outcome"
    return 0.5, environment


def environment_stimulus(environment):
    "An event that changes the environment by +0.1 for all variables"
    for key, value in environment.items():
        environment[key] = value + 0.1
    return 0.1, environment


@pytest.mark.utility
def test_normalised_dict():
    nd = normalised_dict_from_list
    assert nd([]) == {}
    assert nd(['a']) == {'a': 1.0}
    assert nd(['a', 'b']) == {'a': 0.0, 'b': 1.0}
    assert nd(['a', 'b', 'c']) == {'a': 0.0, 'b': 0.5, 'c': 1.0}
    assert nd(['a', 'b', 'c', 'd']) == {
        'a': 0.0, 'b': 1 / 3, 'c': 2 / 3, 'd': 1.0}

    with pytest.raises(TypeError):
        nd({"a": 1})


@pytest.mark.core
def test_input_data_history():
    # First test with no sequence memory
    fish = Respondant(
        actions=[low_action, high_action],
        sequence_memory=0)

    # simply check that input is purely low_action normalised input
    assert fish.input_data(low_action) == [[0.0]]
    assert fish.history == []
    # train and check has stored history but doesn't input
    fish.learn(low_action)
    assert fish.input_data(low_action) == [[0.0]]
    assert fish.history == [low_action]

    # Then check a good memory
    elephant = Respondant(
        actions=[low_action, high_action],
        sequence_memory=2)
    # for first few supply arbitaray data (i.e. the supplied action)
    assert elephant.input_data(low_action) == [[0.0, 0.0, 0.0]]
    assert elephant.history == []
    elephant.learn(low_action)

    # livable quirk, good to introduce randomness
    assert elephant.input_data(high_action) == [[1.0, 0.0, 1.0]]
    elephant.learn(high_action)
    assert elephant.history == [low_action, high_action]

    assert elephant.input_data(low_action) == [[0.0, 1.0, 0.0]]
    elephant.learn(low_action)

    assert elephant.input_data(low_action) == [[0.0, 0.0, 1.0]]
    elephant.learn(low_action)

    assert elephant.history == [low_action,
                                high_action, low_action, low_action]


@pytest.mark.core
def test_input_data_environment():
    TEST_ACTIONS = [low_action, high_action]
    ENVIRON = {'a': 0.1, 'z': 0.9}

    subject = Respondant(actions=TEST_ACTIONS,
                         environment=ENVIRON, sequence_memory=1)

    # action, memory(1), self.environment
    assert subject.input_data(low_action) == [[0.0, 0.0, 0.1, 0.9]]
    assert subject.input_data(high_action) == [[1.0, 1.0, 0.1, 0.9]]
    # action, memory(1), new environment
    assert subject.input_data(low_action, {'a': 0.2, 'z': 0.8}) == \
        [[0.0, 0.0, 0.2, 0.8]]


@pytest.mark.failures
def test_input_data_failures():
    TEST_ACTIONS = [low_action, high_action]
    ENVIRON = {'a': 0.1, 'z': 0.9}

    subject = Respondant(actions=TEST_ACTIONS,
                         environment=ENVIRON, sequence_memory=1)

    # action not in list
    with pytest.raises(KeyError):
        subject.input_data(middle_action)

    # squence not valid (negative)
    with pytest.raises(ValueError):
        Respondant(sequence_memory=-1)

    # environment doesn't match initial
    with pytest.raises(KeyError):
        subject.input_data(low_action, {'c': 1.0})

    # input value too high
    with pytest.raises(ValueError):
        subject.input_data(low_action, {'a': 0.1, 'z': 1.2})


@pytest.mark.core
def test_environment_updates():
    TEST_ACTIONS = [low_action, middle_action]
    TEST_STIMULI = [environment_stimulus]
    VAR_KEY = 'test_stimulus'
    ENVIRON = {VAR_KEY: 0.1}

    subject = Respondant(actions=TEST_ACTIONS,
                         stimuli=TEST_STIMULI, environment=ENVIRON)

    # events combined properly
    assert subject.events == {low_action: 0.0, middle_action: 0.5,
                              environment_stimulus: 1.0}

    # test environment variables get updated
    assert subject.environment[VAR_KEY] == 0.1
    subject.learn(environment_stimulus)
    assert subject.environment[VAR_KEY] == 0.2

    # test environment doesn't update on predict
    subject.predict(environment_stimulus)
    assert subject.environment[VAR_KEY] == 0.2


@pytest.mark.core
def test_decide():
    EPOCHS = 200
    REPS = 5
    TEST_ACTIONS = [low_action, high_action]

    subject = Respondant(actions=TEST_ACTIONS)

    for i in range(REPS):
        for action in TEST_ACTIONS:
            subject.learn(action, epochs=EPOCHS)
            print(subject.decide().__name__)

    assert subject.decide() == high_action


def assert_prediction(subject, error):
    "Simple wrapper to assert level of predictions"
    for action in subject.actions:
        predicted_outcome = subject.predict(action)
        # asserting outcomes only (no environment changes, dummy input)
        assert abs(predicted_outcome - action({})[0]) < error


@pytest.mark.core
def test_learn_with_repetition():
    EPOCHS = 200
    ERROR = 0.2
    REPS = 10
    TEST_ACTIONS = [low_action, high_action]

    subject = Respondant(actions=TEST_ACTIONS)

    # varied learning is far more efficient than
    # concentrated learning
    # i.e. first, second, first, second
    # vs. first, first, second, second
    for i in range(REPS):
        for action in TEST_ACTIONS:
            subject.learn(action, EPOCHS)

    # standard assertions across all
    assert_prediction(subject, error=ERROR)


@pytest.mark.experiment
def test_random_history():
    EPOCHS = 2000
    ERROR = 0.1
    REPS = 1000
    TEST_ACTIONS = [low_action, high_action]
    MEMORY = 1
    LAYERS = 150

    subject = Respondant(actions=TEST_ACTIONS,
                         hidden_layers=LAYERS, sequence_memory=MEMORY)

    for i in range(REPS):
        random_choice = random.choice(TEST_ACTIONS)
        print(subject.input_data(random_choice),
              random_choice({}), subject.predict(random_choice))
        subject.learn(random_choice)

    # standard assertions across all
    assert_prediction(subject, ERROR)


def test_epochs_vs_reps():
    pass


def test_network_layers():
    pass
