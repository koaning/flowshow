<img src="imgs/icon.png" alt="flowshow logo" width="125" align="right"/>

### flowshow

> Just a super thin wrapper for Python tasks that form a flow.

## Installation

```bash
uv pip install flowshow
```

## Usage

Flowshow provides a `@task` decorator that helps you track and visualize the execution of your Python functions. Here's how to use it:

```python
import time
import random

from flowshow import task

# Turns a function into a Task, which tracks a bunch of stuff
@task
def my_function(x):
    time.sleep(0.5)
    return x * 2

# Tasks can also be configured to handle retries
@task(retry_on=ValueError, retry_attempts=10)
def might_fail():
    time.sleep(0.5)
    if random.random() < 0.5:
        raise ValueError("oh no, error!")
    return "done"

@task
def main_job():
    print("This output will be captured by the task")
    for i in range(3):
        my_function(10)
        might_fail()
    return "done"

# Run like you might run a normal function
main_job()
```

Once you run your function you can expect some nice visuals, like this one:

```python
main_job.plot()
```

![](imgs/screenshot.png)

You can also inspect the raw data yourself by running:

```python
main_job.last_run.to_dict()
```

<details>
<summary>Show the full dictionary.</summary>

```
{
  "task_name": "main_job",
  "start_time": "2025-02-04T21:25:17.045576+00:00",
  "duration": 8.864794875029474,
  "inputs": {},
  "error": None,
  "retry_count": 0,
  "end_time": "2025-02-04T21:25:25.909997+00:00",
  "logs": "This output will be captured by the task\n",
  "output": "done",
  "subtasks": [
    {
      "task_name": "my_function",
      "start_time": "2025-02-04T21:25:17.045786+00:00",
      "duration": 0.5050525842234492,
      "inputs": {
        "arg0": 10
      },
      "error": None,
      "retry_count": 0,
      "end_time": "2025-02-04T21:25:17.550808+00:00",
      "logs": "",
      "output": 20
    },
    {
      "task_name": "might_fail",
      "start_time": "2025-02-04T21:25:17.550853+00:00",
      "duration": 0.5053939162753522,
      "inputs": {},
      "error": None,
      "retry_count": 0,
      "end_time": "2025-02-04T21:25:18.056233+00:00",
      "logs": "",
      "output": "done"
    },
    {
      "task_name": "my_function",
      "start_time": "2025-02-04T21:25:18.056244+00:00",
      "duration": 0.5052881669253111,
      "inputs": {
        "arg0": 10
      },
      "error": None,
      "retry_count": 0,
      "end_time": "2025-02-04T21:25:18.561502+00:00",
      "logs": "",
      "output": 20
    },
    {
      "task_name": "might_fail",
      "start_time": "2025-02-04T21:25:18.561516+00:00",
      "duration": 2.1351009169593453,
      "inputs": {},
      "error": None,
      "retry_count": 0,
      "end_time": "2025-02-04T21:25:20.696477+00:00",
      "logs": "",
      "output": "done"
    },
    {
      "task_name": "my_function",
      "start_time": "2025-02-04T21:25:20.696511+00:00",
      "duration": 0.5026454580947757,
      "inputs": {
        "arg0": 10
      },
      "error": None,
      "retry_count": 0,
      "end_time": "2025-02-04T21:25:21.199158+00:00",
      "logs": "",
      "output": 20
    },
    {
      "task_name": "might_fail",
      "start_time": "2025-02-04T21:25:21.199213+00:00",
      "duration": 4.711003000382334,
      "inputs": {},
      "error": None,
      "retry_count": 0,
      "end_time": "2025-02-04T21:25:25.909979+00:00",
      "logs": "",
      "output": "done"
    }
  ]
}
```

</details>

You can also get a flat representation of the same data in a dataframe via:

```python
main_job.to_dataframe()
```

This is what it looks like in Marimo when you evaluate this. Note that we also track the logs of the print statements for later inspection. 

![](imgs/dataframe.png)

### Multiple runs

If you run the function multiple times you can also inspect multiple runs:

```python
main_job.runs
```

This can be useful, but most of the times you're probably interested in the last run. 

