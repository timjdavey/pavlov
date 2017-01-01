from collections import OrderedDict
from neupy import algorithms
import numpy as np
import random


np.random.seed(0)

EPOCHS = 200
LAYERS = 2
STEPS = 0.1


def normalised_dict_from_list(basic_list):
    """
    Takes a list of items and returns a dictionary of
    with the items from the list as the keys,
    the values as values
    """
    d = OrderedDict()
    if type(basic_list) not in (list, tuple):
        raise TypeError("Input must be a list or tuple")
    length = len(basic_list)
    # deal with possible ZeroDivision error
    # return max normalisation
    if length == 1:
        d[basic_list[0]] = 1.0
    # otherwise return the normalised values
    else:
        for i, item in enumerate(basic_list):
            d[item] = i / (length - 1)
    return d


class Respondant(object):
    """
    A generic class for running experiments.
    """
    def __init__(self,
                 actions=None, environment=None, stimuli=None,
                 sequence_memory=0, verbose_neurons=True,
                 hidden_layers=LAYERS, steps=STEPS,
                 scenarios=None):
        # turn the actions&stimuli into network friendly input [0:1]
        self.actions = tuple() if actions is None else tuple(actions)
        self.stimuli = tuple() if stimuli is None else tuple(stimuli)
        # combine into a single list, for normalisation into network
        self.events = normalised_dict_from_list(self.actions + self.stimuli)

        # Both traits & external variables
        if environment is None:
            self.environment = dict()  # blank and will remain so
        else:
            # Ordered to ensure is passed into network consistently
            self.environment = OrderedDict(sorted(environment.items(),
                                                  key=lambda k: k[0]))

        # store a sequence of events & actions
        if sequence_memory < 0 or type(sequence_memory) is not int:
            raise ValueError("Sequence_memory must be a postive real number")
        else:
            self.sequence_memory = sequence_memory
            self.history = list()

        # number of input neurons =
        self.verbose_neurons = verbose_neurons
        if verbose_neurons:
            # 1 input per event, per history, and environment inputs
            inputs = len(self.events) * (self.sequence_memory + 1)\
                + len(self.environment)
        else:
            # current event + historical events, and environment
            inputs = 1 + self.sequence_memory + len(self.environment)

        # generate network
        self.net = algorithms.Backpropagation(
            (inputs, hidden_layers, 1), step=steps,
        )

        # for storage and plotting
        self.scenarios = scenarios
        if scenarios:
            scenes = [(k, []) for k in scenarios.keys()]
            self.predictions = OrderedDict(sorted(scenes, key=lambda k: k[0]))

    def input_defaults(self, environment, history):
        "Returns self values if None passed"
        if history is None:
            history = self.history

        # then append environment inputs
        if environment is None:
            environment = self.environment
        else:
            # ensure is ordered in the same way as self.env
            environment = OrderedDict(sorted(environment.items(),
                                             key=lambda k: k[0]))
            # keys must match to ensure passed correctly in order
            if environment.keys() != self.environment.keys():
                raise KeyError(
                    "Passed environment must match staring env")

        return environment, history

    def input_data(self, event, environment=None, history=None):
        "Given an event, returns the data normalised ready for the network"

        environment, history = self.input_defaults(environment, history)

        if self.verbose_neurons:
            data = []
            # signal which event is "on" as input
            for e in self.events:
                data.append(1 if e == event else 0)
            # do the same for memory
            for i in range(self.sequence_memory):
                try:
                    historical_event = history[-(i + 1)]
                # for the first few iterations where there is no history
                # simply use the inputted event
                except IndexError:
                    historical_event = event
                except TypeError:
                    raise TypeError("history must be a list")
                # then loop through and add that historical
                for e in self.events:
                    data.append(1 if e == historical_event else 0)

        else:
            # takes the normalised action as minimum input
            try:
                data = [self.events[event]]
            except KeyError:
                raise KeyError("%s is not a valid event" % event)

            # then append historical events
            for i in range(self.sequence_memory):
                try:
                    item = self.events[history[-(i + 1)]]
                # if no historical events, use the provided event
                # not particularly accurate but washes it's face
                except IndexError:
                    item = data[0]
                data.append(item)

        # pass (ordered) into network
        for key, value in environment.items():
            if value <= 1.0:
                data.append(value)
            else:
                raise ValueError(
                    "%s env var is %s exceeding maximum 1.0" % (key, value))

        # return as a single row event
        return [data]

    def learn(self, event, epochs=EPOCHS):
        "The act of learning from an event, including storing history."
        input_environment = self.environment
        outcome, output_environment = event(input_environment.copy())

        self.net.train(
            np.array(self.input_data(event, input_environment)),
            np.array([outcome]),
            epochs=epochs)

        # update the environment based on event
        self.environment = output_environment
        # store the event it's sequence memory
        self.history.append(event)

    def predict(self, event, environment=None, history=None):
        "Prediction of an outcome based on an event and environment"
        # predict based on given
        raw_inputs = self.input_data(event, environment, history)
        predicted = self.net.predict(raw_inputs)
        # [0][0] to return just the predicted outcome, rather than the array
        return predicted[0][0]

    def decide(self, environment=None, randomised=0):
        """Work out which action is best to take,
        based on the situation and events"""
        best_action = None
        best_outcome = 0
        for action in self.actions:
            # predict the outcome
            outcome = self.predict(action, environment)
            # if randomised decision making
            outcome += randomised * random.random()
            # take note of the best
            if best_outcome < outcome or best_action is None:
                best_action = action
                best_outcome = outcome

        # return the best action
        return best_action

    def store_predictions(self, keys=None):
        """
        Stores predictions given a set of scenarios.
        Can pass optional list of keys to only store for those values.
        """
        if self.scenarios:
            for key, value in self.scenarios.items():
                if keys is None or key in keys:
                    self.predictions[key].append(self.predict(*value))
                else:
                    self.predictions[key].append(0.0)
        else:
            raise ValueError("scenarios must be set at __init__")

    def plot_predictions(self):
        "Plots any predictions generated from calling decide"
        import matplotlib.pyplot as plt
        if self.scenarios:
            for key, data in self.predictions.items():
                plt.plot(data, label=key)
            plt.axis([0, len(data), 0, 1])
            plt.legend(loc=1)
            plt.grid(True)
            plt.show()
