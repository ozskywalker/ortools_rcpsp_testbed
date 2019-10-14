from __future__ import print_function
import json
import numpy
import collections
from ortools.sat.python import visualization
from ortools.sat.python import cp_model

concurrencyLimit = 6

#
# Results:
# 15 concurrency limit
# >>> (3230120/1000)/60.0
# 53.83533333333333 min
# Horizon = 47421416
# Optimal = 3230120
#
# 30 concurrenct limit
# >>> (1643713/1000)/60.0
# 27.395216666666666 min
# Horizon = 47421416
# Optimal = 1643713
#
# 38 concurrent limit
# >>> (1306271/1000)/60.0
# 21.771183333333333 min
# Horizon = 47421416
# Optimal = 1306271
#
# We /WANT/ to report model (OPTIMAL, etc.) into a metric & back to Slack
# Would be great to also report optimal schedule length & horizon, and also
# push into a CloudWatch metric
#

class Job():
    def __init__(self, job_id, 
                 processing_time, mach_id):
        self.job_id = int(job_id)
        self.processing_time = int(processing_time)
        self.mach_id = int(mach_id)
        self.dep_on = []
        self.start_date = 0

    def add_job_dep(self, temp_dep_on):
        self.dep_on.append(int(temp_dep_on))

    def set_start_date(self, temp_start_date):
        self.start_date = temp_start_date

#
# TEMP NOTES:
#  new = None
#  new = {}
#  new['jobs'] = []
#  temp_job = {"start": 1000, "lastDuration": 46000}
#  new['jobs'].append(temp_job)
#
def step1_loadFromJSON(filename, split_array_num=1):
    with open(filename, "r") as file:
        rawdata = json.load(file)
    
    count = len(rawdata['jobs'])
    print(f"Step 1 (load from JSON): loaded {count} records")
    return rawdata


def step2_buildJobList(data, machine_count=1):
    job_list = []
    machine_id = 0

    for machine in numpy.array_split(data['jobs'], machine_count):
        for singlejob in machine:
            job_list.append( Job(singlejob['start'], float(singlejob['lastDuration']), machine_id) )
        machine_id += 1
    
    print(f"Step 2 (build Job List): processed {machine_id} machines and {len(job_list)} jobs")
    return job_list


def step3_buildModel(job_list, machine_count=1):
    model = cp_model.CpModel()
    horizon = sum([job.processing_time for job in job_list])
    print(f"Step 3 (build Model): Horizon is {horizon}")

    task_type = collections.namedtuple('task_type', 'start end interval') # task type?
    all_tasks = populate_all_tasks(job_list, model, horizon, task_type)
    model = populate_intervals(job_list, all_tasks, machine_count, model)

    # Add precedence constraints
    for job in job_list:
        for dep in job.dep_on:
            model.Add(all_tasks[job.mach_id, job.job_id].start >=
                      all_tasks[job.mach_id, dep].end)

    # Makespan objective.
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(
        obj_var,
        [all_tasks[task].end for task in all_tasks])
        # The above is our dirty hack...
        #
        # [all_tasks[(machine_count,
        #             len(job_list) - 1)].end for job in job_list]) 
    model.Minimize(obj_var)

    # Solve model.
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    # Update Job objects with scheduled start times.
    for job in job_list:
        job.set_start_date(
            solver.Value(all_tasks[job.mach_id,
                                   job.job_id].start))
    
    # Sort the jobs in schedule order
    job_list.sort(key=lambda x: x.start_date, reverse=False)

    # Print schedule.
    print_schedule(job_list, cp_model, solver, status, machine_count)


def print_schedule(job_list, cp_model, solver, status, num_of_machines):
    for machine in range(num_of_machines):
        print(f"Jobs for Machine ID {machine}")
        print(f"=============================")
        for job in job_list:
            if (job.mach_id == machine + 1):
                temp = ''
                temp += "Job %2d | start %3dms" % (job.job_id, job.start_date)
                temp += " (duration %2dms)" % (job.processing_time)
                print(temp)
        print("")

    if status == cp_model.OPTIMAL:
         print('Model is OPTIMAL\nOptimal Schedule Length: %i' % solver.ObjectiveValue())
    if status == cp_model.FEASIBLE:
        print('Model is FEASIBLE\nFeasible Schedule Length: %i' % solver.ObjectiveValue())
    if status == cp_model.INFEASIBLE:
        print('The problem was proven infeasible')
    if status == cp_model.UNKNOWN:
        print('The status of the model is unknown '
              'because a search limit was reached.')


# Add interval blocks that prevent jobs in the same machine overlapping
def populate_intervals(job_list, all_tasks, num_of_machines, model):
    for machine in range(num_of_machines):
        intervals = []
        for job in job_list: 
            if (job.mach_id == machine + 1):
                intervals.append(all_tasks[job.mach_id, job.job_id].interval)
        # Seperate AddNoOverlap for each machine allows concurrent scheduling
        model.AddNoOverlap(intervals)

    return model


def populate_all_tasks(job_list, model, horizon, task_type):
    all_tasks = {}
    for job in job_list:
        start_var = model.NewIntVar(0, horizon, 'start_%i_%i'
                                    % (job.mach_id,
                                       job.job_id))
        duration_var = job.processing_time
        end_var = model.NewIntVar(0, horizon, 'end_%i_%i'
                                  % (job.mach_id,
                                     job.job_id))
        interval_var = model.NewIntervalVar(start_var, duration_var, end_var,
                                            'interval_%i_%i'
                                            % (job.mach_id, job.job_id))
        all_tasks[job.mach_id, job.job_id] = task_type(
                                                start=start_var,
                                                end=end_var,
                                                interval=interval_var)

    return all_tasks


if __name__ == "__main__":
    json = step1_loadFromJSON("./sampledata.json")
    job_list = step2_buildJobList(json, concurrencyLimit)
    step3_buildModel(job_list, concurrencyLimit)

