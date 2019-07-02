#!/usr/bin/env python3

import json

filename = 'sample_instances/sample01.json'
with open(filename, 'r') as f:
    data = json.load(f)

from docplex.cp.model import CpoModel
from docplex.cp.parameters import CpoParameters
from docplex.cp.function import CpoSegmentedFunction
from docplex.cp.function import CpoStepFunction
import docplex.cp.utils_visu as visu

model = CpoModel()
timeslots = int((24*60)/data['time_resolution'])

# Create energy function
# It can be evaluated at start and end of intervals
# so the accumulated value is added to the function
energy = CpoSegmentedFunction(name='energy')
energy_sum = 0
assert(len(data['energy_prices']) == timeslots)
for i in range(timeslots):
    energy_sum += data['energy_prices'][i]
    energy.add_value(i, i+1, energy_sum)
# Ensure the energy sum is defined to the end
energy.add_value(timeslots, timeslots+1, energy_sum)

num_tasks = len(data['tasks'])
num_resources = data['resources']
num_machines = len(data['machines'])

machine_resources = {m['id']: [model.step_at(0,0) for i in range(num_resources)]
                               for m in data['machines']}
machine_on_off = {m['id']: model.step_at(0, 0) for m in data['machines']}
# machine_on_off = {m['id']: model.state_function(
    # trmtx=model.transition_matrix([[0]]),
    # name='m_o_{}'.format(m['id'])) for m in data['machines']}
tasks_running_on_machines = {m['id']: model.state_function(name='tasks_running_on_machines_{}'.format(m['id'])) for m in data['machines']}
on_intervals = {m['id']: model.interval_var_list(min(timeslots, num_tasks),
                name='machine_{}'.format(m['id']),
                optional=True) for m in data['machines']}
cost = 0

for machine in data['machines']:
    id = machine['id']

    # Sequencing of intervals
    num_intervals = len(on_intervals[id])
    # seq = model.sequence_var(on_intervals[id])
    # model.add(model.no_overlap(seq, model.transition_matrix(
        # [[(1) for j in range(num_intervals)] for i in range(num_intervals)])))
    for i in range(1, num_intervals):
        model.add(model.if_then(model.presence_of(on_intervals[id][i]), model.presence_of(on_intervals[id][i-1])))
        # model.add(model.before(seq, on_intervals[id][i-1], on_intervals[id][i]))
        model.add(model.end_before_start(on_intervals[id][i-1], on_intervals[id][i], 1))

    for on_i in on_intervals[id]:
        model.add(model.start_of(on_i) >= 0)
        model.add(model.end_of(on_i) <= timeslots)
        on_i.set_size_min(1)

        # Bind with task intervals
        machine_on_off[id] += model.pulse(on_i, 1)
        # model.add(model.always_equal(machine_on_off[id], on_i, 1, True, True))
        model.add(model.always_constant(tasks_running_on_machines[id], on_i, True, True))

    # Add machine idle consumption
    cost += model.sum([(model.end_eval(on_i, energy) - model.start_eval(on_i, energy))
        * machine['idle_consumption'] for on_i in on_intervals[id]])

    # Add power up/down cost
    on_off_cost = machine['power_up_cost'] + machine['power_down_cost']
    cost += model.sum([on_off_cost*model.presence_of(on_interval) for on_interval in on_intervals[id]])

task_intervals_on_machines = {m['id']: [] for m in data['machines']}
task_intervals = []

for task in data['tasks']:
    task_interval = model.interval_var(size=task['duration'], name='task_{}'.format(task['id']))
    task_interval.set_start_min(task['earliest_start_time'])
    task_interval.set_end_max(task['latest_end_time'])
    task_machine_intervals = model.interval_var_list(num_machines,
                name='task_{}_on_'.format(task['id']),
                optional=True)
    
    for i in range(num_machines):
        id = data['machines'][i]['id']
        # Bind with machine switched on intervals
        model.add(model.always_equal(tasks_running_on_machines[id], task_machine_intervals[i], 1))
        model.add(model.always_in(machine_on_off[id], task_machine_intervals[i], 1, 1))
        # model.add(model.always_constant(machine_on_off[id], task_machine_intervals[i]))

        # Add resource usage by task
        for j in range(num_resources):
            machine_resources[id][j] += model.pulse(
                task_machine_intervals[i], task['resource_usage'][j])

        # For visualization
        task_intervals_on_machines[id].append((task, task_machine_intervals[i]))

    # Only one interval will be effective
    model.add(model.alternative(task_interval, task_machine_intervals))
    task_intervals.append((task, task_interval))

# Add power consumption by tasks
cost += model.sum([(model.end_eval(task_i, energy) - model.start_eval(task_i, energy))
                  * task['power_consumption'] for task, task_i in task_intervals])

for machine in data['machines']:
    id = machine['id']
    for i in range(num_resources):
        model.add(machine_resources[id][i] <= machine['resource_capacities'][i])

model.add(model.minimize(cost))
msol = model.solve(
    params=CpoParameters(TimeLimit=300),
    trace_log=True)

msol.print_solution()

# Draw solution
if msol and visu.is_visu_enabled():
    for m in data['machines']:
        id = m['id']
        ons = []
        cost_sum_a = 0
        cost_sum_b = 0
        energy_costs=CpoStepFunction()
        for j in range(len(on_intervals[id])):
            val = msol.get_value(on_intervals[id][j])
            if val != ():
                var = msol.get_var_solution(on_intervals[id][j])
                ons.append((var, j, 'M_{}_{}'.format(id, str(j))))
                start = var.get_start()
                end = var.get_end()
                start_value = energy.get_value(start)
                end_value = energy.get_value(end)
                cost_a = (end_value - start_value) * m['idle_consumption']
                cost_b = 0
                for i in range(start, end-1):
                    cost_i = data['energy_prices'][i] * m['idle_consumption']
                    cost_b += cost_i
                    energy_costs.add_value(i, i+1, cost_i)
                cost_sum_a += cost_a
                cost_sum_b += cost_b
                print('{} {}'.format(cost_a, cost_b))
                cost_sum_a += m['power_up_cost'] + m['power_down_cost']
                cost_sum_b += m['power_up_cost'] + m['power_down_cost']

        tasks = []
        for task, interval in task_intervals_on_machines[id]:
            val = msol.get_value(interval)
            if val != ():
                tasks.append((msol.get_var_solution(interval), 1, interval.get_name()))
        if len(tasks) > 0 or len(ons) > 0:
            visu.timeline("Machine " + str(id), 0, int(timeslots))
            visu.panel("Tasks")
            visu.sequence(name='Machine', intervals=ons)
            visu.sequence(name='Tasks', intervals=tasks)

            for task, interval in task_intervals_on_machines[id]:
                if msol.get_value(interval) != ():
                    var = msol.get_var_solution(interval)
                    start = var.get_start()
                    end = var.get_end()
                    start_value = energy.get_value(start)
                    end_value = energy.get_value(end)
                    cost_a = (end_value - start_value) * task['power_consumption']
                    cost_b = 0
                    for i in range(start, end-1):
                        cost_i = data['energy_prices'][i] * task['power_consumption']
                        cost_b += cost_i
                        energy_costs.add_value(i, i+1, cost_i)
                    cost_sum_a += cost_a
                    cost_sum_b += cost_b
                    print('{} {}'.format(cost_a, cost_b))
            visu.function(name='Cost={}, {}'.format(cost_sum_a, cost_sum_b), segments=energy_costs)
            # visu.function(name='Energy', segments=energy)

            for j in range(num_resources):
                visu.panel('resources_{}'.format(j))
                res = CpoStepFunction()
                for task, interval in task_intervals_on_machines[id]:
                    if msol.get_value(interval) != ():
                        var = msol.get_var_solution(interval)
                        res.add_value(var.get_start(), var.get_end(), task['resource_usage'][j])
                visu.function(segments=res, color=j)

    visu.show()
