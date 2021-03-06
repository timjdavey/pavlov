# pavlov

This is a story about a dog. A dog which learned.

## Description

Goal of project is to recreate [Pavlov's famous conditioning experiments](https://en.wikipedia.org/wiki/Classical_conditioning) using artifical neural networks rather than living dogs. Also, it's a great excuse to have a play with the basics of neural nets.
 
## Terminology

#### Respondant
The test subject (i.e. dog). It's also the name of the primary experiment class in `pavlov.py`

#### Actions
During an experiment, the `Respondant` is asked to decide what it'd like to do for any given situation. Actions are the list of things which the `Respondant` is able to do e.g. run away from a shock, sit down and rest, or eat.

#### Stimuli
Similar to `Action` but these are things which act upon the `Respondant` e.g. is shocked, hears bell ring, sees food appear.

#### Environment
These are the environmental conditions around the `Respondant` e.g. Gate is closed, food is present.

#### Sequence Memory
Also known as episodic memory. All animals have a limited ability to remember events in sequence. It's crucial in this case, as remembering that food comes after a bell requires then to remember that a bell just rang. In this case it's modelled as a variable, so it can be adjusted how many steps back the `Respondant` can remember.

## Network

Is a basic `Backpropagation` network using the [Neupy framework](http://neupy.com/pages/home.html).

#### Inputs 
These are the previous actions and stimuli for each step in it's memory. Along with the current environmental conditions. Does not remember the environmental conditions for each step of memory, just the present.

#### Output
For a given action, environment and memory of sequence of events - it outputs a number of how much it wants to do that action. This culminates in a function `decide` which for a given environment, will test all it's available actions and return which it most wants to do (with a small bit of introduced randomness).


## Learned Helplessness (gate.py)

Successfully recreated the [experiment](https://en.wikipedia.org/wiki/Learned_helplessness)!.

The basic outline of learned helplessness is as follows

1. Dog is placed on a floor
   * Nothing happens, standard & control
   * Learns to just chill out and rest
2. Dog is placed on electrified floor
   * Dog is randomly shocked
   * Learns to run away when shocked
3. Dog is placed on floor but within a cage
   * Dog is shocked but is unable to run away from floor / pain
   * Dog is continually shocked until it no longer tries to escape
4. Cage is removed from the floor
   * Dog is shocked, but no longer bothers to try and escape despite being able to

### Clean Example

There are four stages to the experiment (as outlined above). The following represents the dogs learning.

The x-axis shows steps in time e.g. every step in time, the Respondant decides what it wants to do at that given step, it does that action and learns from the output. Occassionally a step is overridden by a stimuli and the dog is shocked, learns from that output and the output of the subsequent steps.

Each of the lines represents how desirable the action is (Run or Rest) in a given environmental condition
* No danger - not being shocked
* Danger, no gate - being shocked, can escape from danger if Run
* Danger, gate closed - being shocked, but cannot escape from danger

![pavlov_clean](https://cloud.githubusercontent.com/assets/13322/21580604/3132913e-cffb-11e6-8db1-153116d9c5d1.png)

#### 1. Control
* Steps 0-100 in no danger.
* Blue line shows it learns to rest and chill out.

#### 2. Being shocked, gate open
* Steps 100-400
* Turquiose line shows it learns to run if in danger (i.e. being shocked)
* Takes longer and more noise as shocks are random

#### 3. Being shocked, gate closed
* Steps 400-600 
* Learns to rest as cannot escape shock
* Although neither outcome is particularly desirable as still in pain

#### 4. Being shocked, gate open
* Steps 600-800 
* Finally, we open the gate again but continue to shock
* Can now see by red line that the outcomes have flipped and it would rather rest

### Raw Example

Below shows what it would do if the situation were changed as it learns about it's current situation.
* Initially it learns Rest is favourable in any situation.
* Then in phase two it learns Rest is still favourable when not in danger (1. dark blue), but Running is better when in Danger (4. Turq & 6. Mustard).
* Then in the third phase (gate closed), it still remembers Rest is best when no danger (dark blue), but now also learns Rest is best when in danger (Red & Pink).
* Finally, because that's so strongly engrained, it keeps that memory that Rest is best even when the gate is open.

![pavlov_raw](https://cloud.githubusercontent.com/assets/13322/21580605/3137fe80-cffb-11e6-8f46-244e2abd4c60.png)
