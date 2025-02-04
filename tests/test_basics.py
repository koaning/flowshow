import time
from datetime import datetime, timezone

import pytest

from flowshow import task


def test_basic_task():
    @task
    def simple_task(x: int) -> int:
        return x * 2

    result = simple_task(5)
    assert result == 10
    
    # Check task run information
    last_run = simple_task.last_run
    assert last_run.task_name == "simple_task"
    assert isinstance(last_run.start_time, datetime)
    assert isinstance(last_run.end_time, datetime)
    assert last_run.duration > 0
    assert last_run.error is None
    assert last_run.inputs == {"arg0": 5}
    assert last_run.output == 10


def test_nested_tasks():
    @task
    def inner_task(x: int) -> int:
        return x + 1

    @task
    def outer_task(x: int) -> int:
        return inner_task(x) * 2

    result = outer_task(5)
    assert result == 12
    
    # Check task hierarchy
    outer_run = outer_task.last_run
    assert len(outer_run.subtasks) == 1
    inner_run = outer_run.subtasks[0]
    assert inner_run.task_name == "inner_task"
    assert inner_run.inputs == {"arg0": 5}
    assert inner_run.output == 6


def test_task_with_logs():
    @task(log=True)
    def logging_task():
        print("Starting task")
        print("Task complete")
        return 42

    result = logging_task()
    assert result == 42
    
    last_run = logging_task.last_run
    assert "Starting task" in last_run.logs
    assert "Task complete" in last_run.logs


def test_task_with_error():
    @task
    def failing_task():
        raise ValueError("Task failed")

    with pytest.raises(ValueError, match="Task failed"):
        failing_task()
    
    last_run = failing_task.last_run
    assert isinstance(last_run.error, ValueError)
    assert str(last_run.error) == "Task failed"
    assert last_run.end_time is not None
    assert last_run.duration is None


def test_task_timing():
    @task
    def slow_task():
        time.sleep(0.1)
        return True

    slow_task()
    last_run = slow_task.last_run
    assert last_run.duration >= 0.1
    assert last_run.start_time < last_run.end_time


def test_task_run_history():
    @task
    def counter_task(x: int) -> int:
        return x

    for i in range(3):
        counter_task(i)
    
    history = counter_task.get_all_runs_history()
    assert len(history) == 3
    assert all(isinstance(run, dict) for run in history)
    assert [run["inputs"]["arg0"] for run in history] == [0, 1, 2]


def test_task_visualization():
    @task
    def visualized_task():
        time.sleep(0.1)
        return True

    visualized_task()
    chart = visualized_task.plot
    
    # Check that the chart has the basic Altair structure
    assert chart.mark == "bar"
    assert "start_time" in str(chart.encoding.x)
    assert "end_time" in str(chart.encoding.x2)
    assert "task_name" in str(chart.encoding.y)