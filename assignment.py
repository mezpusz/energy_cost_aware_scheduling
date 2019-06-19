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
energy = CpoSegmentedFunction(name='energy')
for i in range(0, len(data['energy_prices'])):
    energy.add_value(i, i+1, data['energy_prices'][i])
timeslots = (24*60)/data['time_resolution']
num_tasks = len(data['tasks'])
num_resources = data['resources']
machine_resources = {m['id']: [model.step_at(0,0),
                                model.step_at(0,0),
                                model.step_at(0,0)] for m in data['machines']}
machine_power = {m['id']: model.step_at(0, 0) for m in data['machines']}
machine_on_off = {m['id']: model.step_at(0, 0) for m in data['machines']}
tasks_running_on_machines = {m['id']: model.state_function(name='tasks_running_on_machines_{}'.format(m['id'])) for m in data['machines']}
on_intervals = {m['id']: model.interval_var_list(min(timeslots, num_tasks),
                name='machine_{}'.format(m['id']),
                optional=True) for m in data['machines']}
cost = 0

for machine in data['machines']:
    id = machine['id']

    # Sequencing of intervals
    num_intervals = len(on_intervals[id])
    seq = model.sequence_var(on_intervals[id])
    model.add(model.no_overlap(seq,model.transition_matrix(
        [[(1) for j in range(num_intervals)] for i in range(num_intervals)])))
    for i in range(1, num_intervals):
        model.add(model.if_then(model.presence_of(on_intervals[id][i]), model.presence_of(on_intervals[id][i-1])))
        model.add(model.before(seq, on_intervals[id][i], on_intervals[id][i-1]))
    # 
    for on_i in on_intervals[id]:
        model.add(model.start_of(on_i) >= 0)
        model.add(model.end_of(on_i) <= timeslots)
        machine_on_off[id] += model.pulse(on_i, 1)
        model.add(model.always_in(tasks_running_on_machines[id], on_i, 1, 1))

        # Add machine idle consumption
        cost += model.size_eval(on_i, energy, 0)*machine['idle_consumption']

    # Add power up/down cost
    on_off_cost = machine['power_up_cost'] + machine['power_down_cost']
    cost += model.sum([on_off_cost*model.presence_of(on_interval) for on_interval in on_intervals[id]])

task_intervals = []

for task in data['tasks']:
    task_interval = model.interval_var(size=task['duration'], name='task_{}'.format(task['id']))
    task_interval.set_start_min(task['earliest_start_time'])
    task_interval.set_end_max(task['latest_end_time'])
    task_machines_intervals = []
    for machine in data['machines']:
        m_id = machine['id']
        task_machine_interval = model.interval_var(size=task['duration'],
                                                   name='task_{}_on_{}'.format(task['id'], m_id),
                                                   optional=True)
        model.add(model.always_equal(tasks_running_on_machines[m_id], task_machine_interval, 1))
        model.add(model.always_in(machine_on_off[m_id], task_machine_interval, 1, 1))

        cost += model.size_eval(task_machine_interval, energy, 0)*task['power_consumption']

        task_machines_intervals.append(task_machine_interval)
        for i in range(num_resources):
            machine_resources[m_id][i] += model.pulse(
                task_machine_interval, task['resource_usage'][i])

    model.add(model.alternative(task_interval, task_machines_intervals))
    task_intervals.append((task, task_interval))

for machine in data['machines']:
    id = machine['id']
    for i in range(num_resources):
        model.add(model.less_or_equal(
            machine_resources[id][i],
            machine['resource_capacities'][i]))
        # model.add(model.cumul_range(machine_resources[id][i], 0, machine['resource_capacities'][i]))

model.add(model.minimize(cost))
msol = model.solve(
    params=CpoParameters(TimeLimit=300),
    trace_log=False)

msol.print_solution()

# Draw solution
if msol and visu.is_visu_enabled():
    visu.timeline("Solution for " + filename)
    visu.panel("Energy")
    visu.pause(energy)
    for id, intervals in on_intervals.items():
        vars = []
        for j in range(len(intervals)):
            val = msol.get_value(intervals[j])
            if val != ():
                vars.append((msol.get_var_solution(intervals[j]), j, 'M_{}_{}'.format(id, str(j))))
        visu.sequence(name='Machines_{}'.format(id), intervals=vars)

    tasks = []
    for task, interval in task_intervals:
        tasks.append((msol.get_var_solution(interval), 1, interval.get_name()))
    visu.sequence(name='Tasks', intervals=tasks)
    visu.function(name='energy', segments=energy)

    for i in data['machines']:
        for j in range(num_resources):
            visu.panel('resources_{}_{}'.format(i, j))
            res = CpoStepFunction()
            for task, interval in task_intervals:
                if msol.get_value(interval) != ():
                    var = msol.get_var_solution(interval)
                    res.add_value(var.get_start(), var.get_end(), task['resource_usage'][j])
            visu.function(segments=res, color=j)

    # for id, resources in machine_resources.items():
    #     for i in range(num_resources):
    #         print(msol.get_value(resources[i]))
    #         print(resources[i])
            # visu.function(name='machine_{}_resource_{}'.format(id, i), segments=resources[i])
    visu.show()
