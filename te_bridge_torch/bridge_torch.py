"""
Author:      www.tropofy.com

Copyright 2015 Tropofy Pty Ltd, all rights reserved.

This source file is part of Tropofy and governed by the Tropofy terms of service
available at: http://www.tropofy.com/terms_of_service.html

This source file is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE. See the license files for details.
"""

"""

    Created: 20/11/2016
    @author: veng.thai@gmail.com
"""

from sqlalchemy import Column, Text, Float, Integer, UniqueConstraint
from tropofy.database.tropofy_orm import DataSetMixin
from tropofy.app import AppWithDataSets, Step, StepGroup, Parameter
from tropofy.widgets import ExecuteFunction, SimpleGrid, Chart, ParameterForm

import numpy as np
import itertools
import random


class Person(DataSetMixin):
    name = Column(Text)
    speed = Column(Integer)


class LearningRecord(DataSetMixin):
    episode = Column(Integer)
    time_cross_bridge = Column(Integer)


class FinalSolution(DataSetMixin):
    departure = Column(Text)
    who = Column(Text)
    direction = Column(Text)
    destination = Column(Text)


class LearningChart(Chart):
    def get_chart_type(self, app_session):
        return Chart.LINECHART

    def get_table_schema(self, app_session):
        return {
            "episode": ("number", "episode"),
            "time_cross_bridge": ("number", "time to cross bridge")
        }

    def get_table_data(self, app_session):
        data = []
        for row in app_session.data_set.query(LearningRecord).all():
            data.append({'time_cross_bridge': row.episode, 'episode': row.time_cross_bridge})
        return data

    def get_column_ordering(self, app_session):
        return ["time_cross_bridge", "episode"]

    def get_order_by_column(self, app_session):
        return "time_cross_bridge"

    def get_chart_options(self, app_session):
        return {
            'title': 'Learning progress',
            'vAxis': {
                'title': 'time to cross bridge',
                'titleTextStyle': {'color': 'red'}
            },
            'hAxis': {
                'title': 'episode',
                'titleTextStyle': {'color': 'blue'}
            },
            'orientation': 'horizontal',
            'legend' : {
                'position': 'none'
            }
        }

class ExecuteLocalSolver(ExecuteFunction):
    def get_button_text(self, app_session):
        return "Solve Bridge Torch Problem"

    def execute_function(self, app_session):
        mtcp = MonteCarloEstimation(app_session)
        mtcp.solve(app_session)

class BridgeTorchProblemApp(AppWithDataSets):
    def get_name(self):
        return "Bridge and Torch Problem"

    def get_gui(self):
        step_group1 = StepGroup(name='Enter your Data')
        step_group1.add_step(Step(name='Person speed', widgets=[SimpleGrid(Person)]))
        step_group1.add_step(Step(
            name='Training parameters',
            widgets=[{"widget": ParameterForm(), "cols": 6}],
        ))

        step_group2 = StepGroup(name='Solve')
        step_group2.add_step(Step(name='Learning using Monte Carlo estimation', widgets=[ExecuteLocalSolver()]))

        step_group3 = StepGroup(name='View the Solution')
        step_group3.add_step(Step(name='Result', widgets=[LearningChart(), SimpleGrid(FinalSolution)]))

        return [step_group1, step_group2, step_group3]

    def get_examples(self):
        return {
            "Classic example data set 1": load_classic_data_set,
            "Classic example data set 2 ": load_classic_data_set2,
            "6 person data set": load_data_set_6_person,
            "12 person data set": load_data_set_12_person
        }

    def get_parameters(self):
        return [
            Parameter(name='max_people_crossing', label='Max number of person crossing bridge', default=2, allowed_type=int, validator=validate_value_g_zero),
            Parameter(name='learning_iterations', label='Learning iterations', default=5000, allowed_type=int, validator=validate_value_g_zero),
            Parameter(name='min_epsilon_threshold', label='Epsilon threshold', default=5000, allowed_type=int, validator=validate_value_g_zero)
        ]

def validate_value_g_zero(value):
    return True if value > 0 else "Value must be > 0."

def load_classic_data_set(app_session):
    app_session.data_set.add_all([
        Person(name="Alice", speed=1),
        Person(name="Bob", speed=2),
        Person(name="Charles", speed=5),
        Person(name="David", speed=8),
    ])

    app_session.data_set.set_param('max_people_crossing', 2, app_session.app)
    app_session.data_set.set_param('learning_iterations', 1000, app_session.app)
    app_session.data_set.set_param('min_epsilon_threshold', 100, app_session.app)

def load_classic_data_set2(app_session):
    app_session.data_set.add_all([
        Person(name="Alice", speed=1),
        Person(name="Bob", speed=2),
        Person(name="Charles", speed=5),
        Person(name="David", speed=10),
    ])

    app_session.data_set.set_param('max_people_crossing', 2, app_session.app)
    app_session.data_set.set_param('learning_iterations', 1000, app_session.app)
    app_session.data_set.set_param('min_epsilon_threshold', 100, app_session.app)

def load_data_set_6_person(app_session):
    app_session.data_set.add_all([
        Person(name="Alice", speed=1),
        Person(name="Bob", speed=2),
        Person(name="Charles", speed=3),
        Person(name="David", speed=7),
        Person(name="Eric", speed=10),
        Person(name="Federic", speed=12),
    ])

    app_session.data_set.set_param('max_people_crossing', 3, app_session.app)
    app_session.data_set.set_param('learning_iterations', 5000, app_session.app)
    app_session.data_set.set_param('min_epsilon_threshold', 500, app_session.app)

def load_data_set_12_person(app_session):
    app_session.data_set.add_all([
        Person(name="Alice", speed=1),
        Person(name="Bob", speed=2),
        Person(name="Charles", speed=3),
        Person(name="David", speed=7),
        Person(name="Eric", speed=10),
        Person(name="Federic", speed=12),
        Person(name="Gary", speed=5),
        Person(name="Henry", speed=4),
        Person(name="Ivan", speed=11),
        Person(name="Jake", speed=18),
        Person(name="Kevin", speed=16),
        Person(name="Larry", speed=9),
    ])

    app_session.data_set.set_param('max_people_crossing', 4, app_session.app)
    app_session.data_set.set_param('learning_iterations', 10000, app_session.app)
    app_session.data_set.set_param('min_epsilon_threshold', 1000, app_session.app)

class MonteCarloEstimation(object):

    def __init__(self, app_session, gamma=0.9):
        self.gamma = gamma
        self.learning_iterations = app_session.data_set.get_param('learning_iterations')
        self.max_people_crossing = app_session.data_set.get_param('max_people_crossing')
        self.n0 = app_session.data_set.get_param('min_epsilon_threshold')

        # initial people set contains the speed for each individual

        self.initial_people_set = [p.speed for p in app_session.data_set.query(Person).all()]
        self.size = len(self.initial_people_set)
        
        print '\nThese are the speeds for each person:', self.initial_people_set, '\n'
        
        power_set_counts = np.power(2, len(self.initial_people_set)) - 1

        # the power set records the speed combination --> state encoding
        self.power_set = {}

        # the reverse power set records the state --> speed combination encoding
        self.reverse_power_set = {}

        # state is used to encodes the speed combination(which also indicate what state we are currently in)
        state = 0

        # generate all the possible combination people's speed, this is to reflect all the possible scenario
        # on either side of the bridge (there might be some invalid scenario, but it would be ignored)

        for r in range(1, self.size + 1):

            # a state is one of the possible combination of the people's speed

            for speed_combination in itertools.combinations(self.initial_people_set, r):
                # we need to sort the speed combination so that it is consistence
                sorted_speed_combination = tuple(sorted(speed_combination))
                self.power_set[sorted_speed_combination] = state
                self.reverse_power_set[state] = sorted_speed_combination
                state += 1

        print 'Power set:', self.power_set, '\n\nReverse Power set:', self.reverse_power_set, '\n'

        self.N = np.zeros((2, power_set_counts, power_set_counts))
        self.Q = np.zeros((2, power_set_counts, power_set_counts))

        print 'N Shape:', self.N.shape, 'Q Shape', self.Q.shape, '\n'


    """
        This method is used to generate all the possible cross bridge combination from the current people left on
        the left hand side of the bridge (departure point).
        For example, if the max people crossing the bridge is 2, and currently we have 3 people left(3,5,4), this
        method will generate a C(3,2) combination which are (3,5), (3,4), (4,5)

        Paramaters:
            state: the current people left (encoded as a state)
            max_people: this is used to indicate the direction of crossing, if coming back, it will be 1
    """
    def _all_possible_actions(self, state, max_people):
        speed_combination = self.reverse_power_set[state]

        possible_states = np.empty((0,), dtype=int)

        for sub_combintaion in itertools.combinations(speed_combination, max_people):
            sorted_sub_combination = tuple(sorted(sub_combintaion))
            possible_states = np.append(possible_states, self.power_set[sorted_sub_combination])

        #print 'From', state, 'we got', speed_combination, '->', possible_states
        return possible_states

    """
        This is the epsilon greedy method used in reinforcement learning, which tries to balance between exploration
        and exploitation. The threshold to do exploration or exploitation is also affected by the number of tries on
        a certain state
    """
    def _eps_greedy_choice(self):
        # encodes the current people left as a state
        state = self.power_set[tuple(np.sort(self.people_left))]
        # counts the number of visit to this state
        visits_to_state = np.sum(self.N[0, state])

        # compute epsilon
        current_epsilon = self.n0 / (self.n0 + visits_to_state)

        # epsilon greedy policy
        if random.random() < current_epsilon:
            # exploration
            threshold = self.max_people_crossing if len(self.people_left) >= self.max_people_crossing else len(self.people_left)
            return tuple(np.sort(np.random.choice(self.people_left, size=threshold, replace=False)))
        else:
            # exploitation
            possible_actions = self._all_possible_actions(state, self.max_people_crossing)
            power_set_index = np.argmax(self.Q[0, state, possible_actions])
            power_set_index = possible_actions[power_set_index]

            return self.reverse_power_set[power_set_index]

    """
        This method works the same way as _eps_greedy_choice, but with a little variation which is it operates on the 
        number of people already crossed the bridge, this is used when a person holds a torch coming back
    """
    def _eps_greedy_choice_return_torch(self):
        # encodes the current people left as a state
        state = self.power_set[tuple(np.sort(self.people_crossed))]
        # counts the number of visit to this state
        visits_to_state = np.sum(self.N[1, state])

        # compute epsilon
        current_epsilon = self.n0 / (self.n0 + visits_to_state)

        # epsilon greedy policy
        if random.random() < current_epsilon:
            # exploration
            return tuple(np.sort(np.random.choice(self.people_crossed, size=1, replace=False)))
        else:
            # exploitation
            possible_actions = self._all_possible_actions(state, 1)
            power_set_index = np.argmax(self.Q[1, state, possible_actions])
            power_set_index = possible_actions[power_set_index]

            return self.reverse_power_set[power_set_index]


    """
        This method computes the naive approach of always using the fasted person to bring the torch back, this serves as
        a reference from the agent to learn (check if their new solution is better or worse, hence get a reward or not)
    """
    def _reference_elapsed_time(self):
        quickest_person = np.min(self.initial_people_set)
        return (quickest_person * (self.size - 2)) + np.sum(self.initial_people_set)


    """
        This method updates the people of the two sides of the bridge when a number of person is chosen to cross the bridge

        Parameters:
            people_chosen: the array of people (speed) chosen to cross the bridge (1 person when coming back)
            back: direction indicating whether we are crossing or going back with the torch

            return: returns the speed of the slowest person in the chosen group as we always move with the 
            slowest person's speed
    """
    def _cross_the_bridge(self, people_chosen, back):
        if not back:
            self.people_left = np.setdiff1d(self.people_left, people_chosen)
            self.people_crossed = np.append(self.people_crossed, people_chosen)

            slowest_person = np.amax(people_chosen)
            
            return slowest_person
        else:
            self.people_left = np.append(self.people_left, people_chosen)
            self.people_crossed = np.setdiff1d(self.people_crossed, people_chosen)
            
            return people_chosen[0]

    """
        Main method used to solve the puzzle, the method employs Reinforcement learning technique to learn from history
        and improve by iterations

        Parameters:
            app_session: The tropofy app_session
    """
    def solve(self, app_session):
        learning_outcomes = []

        history = []
        direction_debug = {True:'<--', False:'-->'}
        min_total_time = 1000000
        min_solution = []

        ref_elapsed_time = self._reference_elapsed_time()

        print 'Reference time is:', ref_elapsed_time

        for iteration in range(self.learning_iterations):
            back = False
            total_time = 0

            solution = []
            self.people_left = np.array(self.initial_people_set, dtype=int)
            self.people_crossed = np.empty((0,), dtype=int)

            people_chosen = self._eps_greedy_choice()

            while len(self.people_left) > 0:
                direction = 0 if back else 1

                #print 'pleft', self.people_left, '-', self.people_crossed, ' choosen ', people_chosen, direction_debug[back]

                state = self.power_set[tuple(sorted(self.people_left))]
                action = self.power_set[tuple(people_chosen)]
                history.append( (state, action, back) )

                self.N[direction, state, action] += 1

                prev_people_left = self.people_left
                elapsed_time = self._cross_the_bridge(people_chosen, back)
                solution.append((prev_people_left, people_chosen, direction_debug[back], self.people_crossed))

                #print '\t', self.people_left, people_chosen, self.people_crossed

                total_time += elapsed_time

                lr = 1.0 / self.N[direction, state, action]

                back = not back
                if len(self.people_left) > 0:
                    if not back:
                        people_chosen = self._eps_greedy_choice()
                    else:
                        people_chosen = self._eps_greedy_choice_return_torch()
                    next_action = self.power_set[tuple(sorted(self.people_left))]
                    delta = self.gamma * self.Q[direction, state, next_action] - self.Q[direction, state, action]
                else:
                    reward = total_time - ref_elapsed_time
                    reward *= -1

                    if total_time < min_total_time:
                        min_solution = solution
                        min_total_time = total_time

                    #print 'time=', total_time, 'rewarad=', reward
                    app_session.task_manager.send_progress_message("iteration: %d total time: %d" % (iteration, total_time))
                    learning_outcomes.append(LearningRecord(episode=iteration+1, time_cross_bridge=total_time))
                    delta = reward - self.Q[direction, state, action]

                self.Q[direction, state, action] += lr * delta

        print 'Last solution', min_solution

        app_session.data_set.add_all(learning_outcomes)

        for record in min_solution:
            #print type(record[0]), record[0], type(record[1]), record[1]
            app_session.data_set.add(FinalSolution(departure=','.join(map(str, record[0])), who=','.join(map(str, record[1])), direction=record[2], destination=','.join(map(str, record[3]))))

        app_session.task_manager.send_progress_message("<br> Reference crossing bridge time = %s" % ref_elapsed_time)
        app_session.task_manager.send_progress_message("End of iteration solution cross bridge time = %s" % total_time)
        app_session.task_manager.send_progress_message("Minimum cross bridge time = %s" % min_total_time)


