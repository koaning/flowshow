import marimo

__generated_with = "0.9.20"
app = marimo.App(width="medium")


@app.cell
def __():
    from functools import wraps
    import time
    from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple
    from dataclasses import dataclass, field, asdict
    from datetime import datetime, timezone
    import inspect
    from contextlib import contextmanager, redirect_stdout
    import threading
    import io
    import sys
    import stamina

    # Thread-local storage for tracking the current task
    _task_context = threading.local()

    @dataclass
    class TaskRun:
        task_name: str
        start_time: datetime
        end_time: Optional[datetime] = None
        duration: Optional[float] = None
        inputs: Dict[str, Any] = field(default_factory=dict)
        output: Any = None
        error: Optional[Exception] = None
        subtasks: List['TaskRun'] = field(default_factory=list)
        logs: Optional[str] = None
        retry_count: int = 0

        def add_subtask(self, subtask: 'TaskRun'):
            self.subtasks.append(subtask)
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert the task run and all its subtasks to a nested dictionary."""
            result = {
                "task_name": self.task_name,
                "start_time": self.start_time.isoformat(),
                "duration": self.duration,
                "inputs": self.inputs,
                "error": str(self.error) if self.error else None,
                "retry_count": self.retry_count
            }
            
            if self.end_time:
                result["end_time"] = self.end_time.isoformat()
                
            if self.logs is not None:
                result["logs"] = self.logs
                
            # Only include output if it's a simple type that can be serialized
            if isinstance(self.output, (str, int, float, bool, type(None))):
                result["output"] = self.output
                
            if self.subtasks:
                result["subtasks"] = [task.to_dict() for task in self.subtasks]
                
            return result

    @contextmanager
    def _task_run_context(run: TaskRun):
        parent = getattr(_task_context, 'current_run', None)
        _task_context.current_run = run
        try:
            yield
        finally:
            if parent is not None:
                parent.add_subtask(run)
            _task_context.current_run = parent

    @dataclass
    class TaskDefinition:
        func: Callable
        name: str
        capture_logs: bool = False
        runs: List[TaskRun] = field(default_factory=list)

        def __call__(self, *args, **kwargs):
            # Create a new run
            run = TaskRun(
                task_name=self.name,
                start_time=datetime.now(timezone.utc),
                inputs={
                    **{f"arg{i}": arg for i, arg in enumerate(args)},
                    **kwargs
                }
            )
            
            with _task_run_context(run):
                try:
                    # Execute the task
                    start = time.perf_counter()
                    
                    if self.capture_logs:
                        stdout_capture = io.StringIO()
                        with redirect_stdout(stdout_capture):
                            result = self.func(*args, **kwargs)
                        run.logs = stdout_capture.getvalue()
                    else:
                        result = self.func(*args, **kwargs)
                    
                    end = time.perf_counter()
                    
                    # Record successful completion
                    run.end_time = datetime.now(timezone.utc)
                    run.duration = end - start
                    run.output = result
                    
                except stamina.StaminaError as e:
                    # Record retry information if it was a retried task
                    run.retry_count = e.attempts
                    run.error = e.last_exception
                    raise e.last_exception
                except Exception as e:
                    # Record error if task fails
                    run.end_time = datetime.now(timezone.utc)
                    run.error = e
                    raise
                
                finally:
                    # Always add the run to history if this is a top-level task
                    if getattr(_task_context, 'current_run', None) is run:
                        self.runs.append(run)
                
                return result

        @property
        def last_run(self) -> Optional[TaskRun]:
            """Returns the most recent run of this task"""
            return self.runs[-1] if self.runs else None
        
        def get_run_history(self) -> List[Dict[str, Any]]:
            """Returns the complete history of all runs with their nested subtasks."""
            return [run.to_dict() for run in self.runs]

    def task(
        func: Optional[Callable] = None, 
        *, 
        log: bool = False,
        retry_on: Optional[Union[Type[Exception], Tuple[Type[Exception], ...]]] = None,
        retry_attempts: Optional[int] = None
    ) -> Callable:
        """Decorator to mark a function as a trackable task.
        
        Args:
            func: The function to decorate
            log: If True, capture stdout during task execution
            retry_on: Exception or tuple of exceptions to retry on
            retry_attempts: Number of retry attempts
        """
        def decorator(f: Callable) -> TaskDefinition:
            # Apply stamina retry if retry parameters are provided
            if retry_on is not None and retry_attempts is not None:
                f = stamina.retry(on=retry_on, attempts=retry_attempts)(f)
            return TaskDefinition(func=f, name=f.__name__, capture_logs=log)
        
        if func is None:
            return decorator
        return decorator(func)

    return (
        Any,
        Callable,
        Dict,
        List,
        Optional,
        TaskDefinition,
        TaskRun,
        Tuple,
        Type,
        Union,
        asdict,
        contextmanager,
        dataclass,
        datetime,
        field,
        inspect,
        io,
        redirect_stdout,
        stamina,
        sys,
        task,
        threading,
        time,
        timezone,
        wraps,
    )


@app.cell
def __(task):
    @task
    def load_data(a):
        return [a for i in range(10)]

    @task(log=True)
    def merge(a, b):
        print("merging")
        return zip(a, b)

    @task
    def doit():
        a = load_data(1)
        b = load_data(2)
        print(len(list(merge(a, b))))
        
    @task(log=True)
    def main():
        print("yo")
        doit()
        print("yeeheaw")
        doit()


    result = main()
    return doit, load_data, main, merge, result


@app.cell
def __(main):
    main.last_run.to_dict()
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
