#!/usr/bin/env python3

import json
import sys

from docplex.cp.expression import INTERVAL_MAX
from docplex.cp.function import CpoSegmentedFunction, CpoStepFunction
from docplex.cp.model import CpoModel
from docplex.cp.parameters import CpoParameters
import docplex.cp.utils_visu as visu

if len(sys.argv) != 2:
    print('Filename should be given as first parameter!')
    exit(-1)

filename = sys.argv[1]
with open(filename, 'r') as f:
    data = json.load(f)

TIMESLOTS = int((24*60)/data['time_resolution'])
NUM_TASKS = len(data['tasks'])
NUM_RESOURCES = data['resources']
NUM_MACHINES = len(data['machines'])
energy_prices = data['energy_prices']

# Create energy function
# It can be evaluated at start and end of intervals
# so the accumulated value is added to the function
energy_sum_array = []
energy_sum_all = 0
assert len(energy_prices) == TIMESLOTS

for i in range(TIMESLOTS):
    energy_sum_all += energy_prices[i]
    energy_sum_array.append(energy_sum_all)
# Ensure the energy sum is defined to the end
energy_sum_array.append(energy_sum_all)

# This array contains duration i+1 energy sums at the
# ith position and is used for fixed durations
energy_intervals_array = []
for i in range(1, TIMESLOTS):
    energy_intervals = CpoSegmentedFunction()
    energy_sum = 0
    for j in range(i):
        energy_sum += energy_prices[j]
    energy_intervals.set_value(0, 1, energy_sum)
    energy_sum_array.append(energy_sum)
    for j in range(i, TIMESLOTS):
        energy_sum -= energy_prices[j-i]
        energy_sum += energy_prices[j]
        energy_intervals.set_value(j-i+1, j-i+2, energy_sum)
    energy_intervals.set_value(TIMESLOTS-i+1, INTERVAL_MAX, 10000)
    energy_intervals_array.append(energy_intervals)

model = CpoModel()
# Function is positive when a machine is on
# This should be also a state function but it cannot be well defined
# with optional intervals as machine intervals
machine_on_off = {m['id']: model.step_at(0, 0) for m in data['machines']}

# State is 1 when a task is running on a machine
tasks_running_on_machines = {
    m['id']: model.state_function(name='tasks_on_machines_{}'.format(m['id']))
    for m in data['machines']
}

# Denotes the intervals when a machine is ON
on_intervals = {
    m['id']: model.interval_var_list(
        min(TIMESLOTS, NUM_TASKS),
        start=(0, TIMESLOTS),
        end=(0, TIMESLOTS),
        size=(1, TIMESLOTS),
        name='machine_{}'.format(m['id']),
        optional=True)
    for m in data['machines']
}
# Store the current used capacity of every machine resource
machine_resources = {
    m['id']: [model.step_at(0, 0) for i in range(NUM_RESOURCES)]
    for m in data['machines']
}

# Phase 1
#
# Cost of tasks is calculated here without concerning ourselves
# about the machine costs.

task_intervals_on_machines = {m['id']: [] for m in data['machines']}
task_intervals = []

for machine in data['machines']:
    m_id = machine['id']

    # Sequencing of intervals
    num_intervals = len(on_intervals[m_id])
    for i in range(1, num_intervals):
        model.add(model.if_then(
            model.presence_of(on_intervals[m_id][i]),
            model.presence_of(on_intervals[m_id][i-1])))
        model.add(model.end_before_start(
            on_intervals[m_id][i-1],
            on_intervals[m_id][i],
            1))

    # Bind with task intervals
    for on_i in on_intervals[m_id]:
        machine_on_off[m_id] += model.pulse(on_i, 1)
        model.add(model.always_constant(
            tasks_running_on_machines[m_id],
            on_i, True, True))

for task in data['tasks']:
    # Master task interval
    task_interval = model.interval_var(
        size=task['duration'],
        start=(task['earliest_start_time'],
               task['latest_end_time']-task['duration']),
        end=(task['earliest_start_time']+task['duration'],
             task['latest_end_time']),
        name='task_{}'.format(task['id']))
    # Optional task intervals that belong to each machine
    task_machine_intervals = model.interval_var_list(
        NUM_MACHINES,
        name='task_{}_on_'.format(task['id']),
        optional=True)

    for i in range(NUM_MACHINES):
        m_id = data['machines'][i]['id']

        # Bind with machine switched on intervals
        model.add(model.always_equal(
            tasks_running_on_machines[m_id],
            task_machine_intervals[i],
            1))
        model.add(model.always_in(
            machine_on_off[m_id],
            task_machine_intervals[i],
            1, 1))

        # Add resource usage by task
        for j in range(NUM_RESOURCES):
            machine_resources[m_id][j] += model.pulse(
                task_machine_intervals[i],
                task['resource_usage'][j])

        # For visualization
        task_intervals_on_machines[m_id].append(
            (task, task_machine_intervals[i]))

    # Only one interval will be effective
    model.add(model.alternative(task_interval, task_machine_intervals))

    task_intervals.append((task, task_interval))

# Add power consumption by tasks
cost_tasks = model.sum([
    model.start_eval(task_int, energy_intervals_array[task['duration']-1]) *
    task['power_consumption'] for task, task_int in task_intervals
])

# Add resource capacity constraints
for machine in data['machines']:
    m_id = machine['id']
    for j in range(NUM_RESOURCES):
        model.add(machine_resources[m_id][j] <=
                  machine['resource_capacities'][j])

# Add minimize constraint
model.add(model.minimize(cost_tasks))

msol = model.solve(
    params=CpoParameters(
        TimeLimit=300,
        SearchType='IterativeDiving',
        LogVerbosity='Terse'
    ),
    trace_log=True)
msol.print_solution()

# Phase 2
#
# This phase starts from a minimal (or relatively minimal,
# depending on the timeout of the first phase) solution
# that will be further minimized with the additional machine
# costs.

# This solution is consistent with the second phase,
# set it as starting point
model.set_starting_point(msol.get_solution())
bounds = msol.get_objective_bounds()
model.remove(cost_tasks)

cost_overall = cost_tasks

for machine in data['machines']:
    m_id = machine['id']
    # Add machine idle consumption
    cost_overall += model.sum([
        (model.element(energy_sum_array, model.end_of(on_i)) -
         model.element(energy_sum_array, model.start_of(on_i))) *
        machine['idle_consumption'] for on_i in on_intervals[m_id]
    ])

    # Add power up/down cost
    on_off_cost = machine['power_up_cost'] + machine['power_down_cost']
    cost_overall += model.sum([
        on_off_cost * model.presence_of(on_interval)
        for on_interval in on_intervals[m_id]
    ])

# We can be sure that the lower bound of
# first phase is a lower bound of this phase
model.add(cost_overall >= bounds[0])

model.add(model.minimize(cost_overall))
msol = model.solve(
    params=CpoParameters(
        TimeLimit=300,
        SearchType='IterativeDiving',
        LogVerbosity='Terse'
    ),
    trace_log=True)
msol.print_solution()

# Draw solution
if msol and visu.is_visu_enabled():
    for m in data['machines']:
        m_id = m['id']
        ons = []
        cost_sum = 0
        energy_costs = CpoStepFunction()
        j = 0
        # Add machine intervals
        for interval in on_intervals[m_id]:
            val = msol.get_value(interval)
            if val != ():
                ons.append((msol.get_var_solution(interval),
                            j, interval.get_name()))
                j += 1
                start_value = energy_sum_array[val[0]]
                end_value = energy_sum_array[val[1]]
                cost_sum += (end_value - start_value) * m['idle_consumption']
                cost_sum += m['power_up_cost'] + m['power_down_cost']
                # Add segments to cost function
                for i in range(val[0], val[1]):
                    cost_i = energy_prices[i] * m['idle_consumption']
                    energy_costs.add_value(i, i+1, cost_i)
        # Add task intervals
        tasks = []
        for task, interval in task_intervals_on_machines[m_id]:
            val = msol.get_value(interval)
            if val != ():
                tasks.append((msol.get_var_solution(interval),
                              1, interval.get_name()))
                cost_sum += energy_intervals_array[val[2]-1].get_value(val[0]) * task['power_consumption']
                # Add segments to cost function
                for i in range(val[0], val[1]):
                    cost_i = energy_prices[i] * task['power_consumption']
                    energy_costs.add_value(i, i+1, cost_i)
        # Do not show this machine if no task if assigned to it
        if len(tasks) > 0 or len(ons) > 0:
            visu.timeline("Machine " + str(m_id), 0, int(TIMESLOTS))
            visu.panel("Tasks")
            visu.sequence(name='Machine', intervals=ons)
            visu.sequence(name='Tasks', intervals=tasks)
            visu.function(name='Cost={}'.format(cost_sum),
                          segments=energy_costs)

            for j in range(NUM_RESOURCES):
                visu.panel('resources_{}'.format(j))
                res = CpoStepFunction()
                for task, interval in task_intervals_on_machines[m_id]:
                    val = msol.get_value(interval)
                    if val != ():
                        res.add_value(val[0], val[1],
                                      task['resource_usage'][j])
                visu.function(segments=res, color=j)

    visu.show()
