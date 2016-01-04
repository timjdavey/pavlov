from collections import OrderedDict
from neupy import algorithms
import numpy as np
import logging


np.random.seed(0)

EPOCHS = 2000
LAYERS = 150
STEPS = 0.1


def normalised_dict_from_list(basic_list):
    """
    Takes a list of items and returns a dictionary of
    with the items from the list as the keys,
    the values as values
    """
    d = {}
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

    def __init__(self,
                 actions=None, environment=None, stimuli=None,
                 sequence_memory=0,
                 hidden_layers=LAYERS, steps=STEPS):
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
        # current action, historical actions
        inputs = 1 + self.sequence_memory + len(self.environment)

        # generate network
        self.net = algorithms.Backpropagation(
            (inputs, hidden_layers, 1),
            step=steps,
            # show_epoch=1000,
        )

    def input_data(self, event, environment=None):
        "Given an event, returns the data normalised ready for the network"
        # takes the normalised action as minimum input
        try:
            data = [self.events[event]]
        except KeyError:
            raise KeyError("%s is not a valid event" % event)

        # then append historical events
        for i in range(self.sequence_memory):
            try:
                item = self.events[self.history[-(i + 1)]]
            # if no historical events, use the provided event
            # not particularly accurate but washes it's face
            except IndexError:
                item = data[0]
            data.append(item)

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

    def predict(self, event, environment=None):
        "Prediction of an outcome based on an event and environment"
        # predict based on given
        predicted = self.net.predict(self.input_data(event, environment))
        # [0][0] to return just the predicted outcome, rather than the array
        return predicted[0][0]

    def decide(self):
        """Work out which action is best to take,
        based on the situation and events"""
        best_action = None
        best_outcome = 0
        for event in self.events.keys():
            # can only decide to do actions
            # stimuli act on it
            if event in self.actions:
                outcome = self.predict(event)
                print(event.__name__, outcome)
                if best_outcome < outcome or best_action is None:
                    best_action = event
                    best_outcome = outcome
        return best_action
