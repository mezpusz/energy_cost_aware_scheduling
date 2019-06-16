#!/usr/bin/env python2.7

import json

with open('sample_instances/sample01.json', 'r') as f:
    data = json.load(f)

from docplex.cp.model import CpoModel

model = CpoModel()
timeslots = (24*60)/data['time_resolution']
num_resources = data['resources']
machine_resources = {}
machine_power = {}
#TODO: remove these initial pulses
cost = 0

for machine in data['machines']:
    on_intervals = model.interval_var_list(timeslots/2+1, 0, timeslots-1,
                                           name='machine_{}'.format(machine['id']),
                                           optional=True)
    machine_power[machine['id']] = model.pulse(model.interval_var(0, timeslots-1), 0)
    for i in xrange(0, len(on_intervals)-1):
        model.add(model.if_then(model.presence_of(on_intervals[i]), model.presence_of(on_intervals[i+1])))
        model.add(model.end_before_start(on_intervals[i], on_intervals[i+1]))
        machine_power[machine['id']] += model.pulse(on_intervals[i], machine['idle_consumption'])
        on_off_cost = machine['power_up_cost'] + machine['power_down_cost']
        cost += model.sum([on_off_cost*model.presence_of(on_interval) for on_interval in on_intervals])

    machine_resources[machine['id']] = [model.pulse(model.interval_var(0, timeslots-1), 0),
                                        model.pulse(model.interval_var(0, timeslots-1), 0),
                                        model.pulse(model.interval_var(0, timeslots-1), 0)]
    for i in xrange(0, num_resources):
        model.add(model.less_or_equal(
            machine_resources[machine['id']][i],
            machine['resource_capacities'][i]))

for task in data['tasks']:
    task_interval = model.interval_var(size=task['duration'], name='task_{}'.format(task['id']))
    task_interval.set_start_min(task['earliest_start_time'])
    task_interval.set_end_max(task['latest_end_time'])
    task_machines_intervals = []
    for machine in data['machines']:
        task_machine_interval = model.interval_var(size=task['duration'],
                                                   name='task_{}_on_{}'.format(task['id'], machine['id']),
                                                   optional=True)
        task_machines_intervals.append(task_machine_interval)
        for i in xrange(0, num_resources):
            machine_resources[machine['id']][i] += model.pulse(
                task_machine_interval, task['resource_usage'][i])
    model.add(model.alternative(task_interval, task_machines_intervals))

model.add(model.minimize(cost))
print(model.solve())