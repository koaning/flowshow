import uuid
import inspect
import io
import time
import contextvars
from contextlib import contextmanager, redirect_stdout, asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, Literal
import traceback

import altair as alt
import pandas as pd
import stamina

from .visualize import flatten_tasks

# Replace threading.local() with contextvars.ContextVar
_task_context = contextvars.ContextVar('task_context', default=None)

LogLevel = Literal["INFO", "WARNING", "ERROR", "DEBUG"]

@dataclass
class TaskRun:
    task_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    output: Any = None
    error: Optional[Exception] = None
    error_traceback: Optional[str] = None
    subtasks: List["TaskRun"] = field(default_factory=list)
    logs: List[List[str]] = field(default_factory=list)
    retry_count: int = 0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    artifacts: Dict[str, Any] = field(default_factory=dict)
    table: Dict[str, Any] = field(default_factory=dict)

    def add_subtask(self, subtask: "TaskRun"):
        self.subtasks.append(subtask)

    def _log(self, level: LogLevel, message: str) -> None:
        """Add a log entry with the specified level."""
        self.logs.append([level, message, datetime.now(timezone.utc).isoformat()])

    def to_dict(self) -> Dict[str, Any]:
        """Convert the task run and all its subtasks to a nested dictionary."""
        result = {
            "id": self.id,
            "task_name": self.task_name,
            "start_time": self.start_time.isoformat(),
            "duration": self.duration,
            "inputs": self.inputs,
            "error": str(self.error) if self.error else None,
            "error_traceback": self.error_traceback,
            "retry_count": self.retry_count,
            "artifacts": self.artifacts,
            "table": self.table,
        }

        if self.end_time:
            result["end_time"] = self.end_time.isoformat()

        if self.logs:
            result["logs"] = self.logs

        if self.artifacts:
            result["artifacts"] = self.artifacts

        # Only include output if it's a simple type that can be serialized
        if isinstance(self.output, (str, int, float, bool, type(None))):
            result["output"] = self.output

        if self.subtasks:
            result["subtasks"] = [task.to_dict() for task in self.subtasks]

        return result

    def to_dataframe(self):
        return pd.DataFrame(flatten_tasks(self.to_dict()))

    def plot(self):
        dataf = self.to_dataframe()
        return (
            alt.Chart(dataf)
            .mark_bar()
            .encode(
                x=alt.X("start_time:T", title="Time"),
                x2="end_time:T",
                y=alt.Y("task_name:N", title="Task", sort=alt.EncodingSortField(field="start_time", order="ascending")),
                tooltip=["task_name", "duration"],
            )
            .properties(width=800, height=400, title="Task Timeline")
        )


@contextmanager
def _task_run_context(run: TaskRun):
    # Get the parent task run (if any)
    parent = _task_context.get()
    # Save the previous context and set the new one
    token = _task_context.set(run)
    try:
        yield
    finally:
        if parent is not None:
            parent.add_subtask(run)
        # Restore the previous context
        _task_context.reset(token)


@asynccontextmanager
async def _async_task_run_context(run: TaskRun):
    parent = _task_context.get()
    token = _task_context.set(run)
    try:
        yield
    finally:
        if parent is not None:
            parent.add_subtask(run)
        _task_context.reset(token)


@dataclass
class TaskDefinition:
    func: Callable
    name: str
    capture_logs: bool = False
    callback: Optional[Callable[[Dict[str, Any]], None]] = None
    runs: List[TaskRun] = field(default_factory=list)
    is_async: bool = field(default=False)

    def __post_init__(self):
        # Detect if the wrapped function is async
        self.is_async = inspect.iscoroutinefunction(self.func)

    def __call__(self, *args, **kwargs):
        if self.is_async:
            # Return awaitable for async functions
            return self._async_call(*args, **kwargs)
        else:
            # Execute synchronously for regular functions
            return self._sync_call(*args, **kwargs)
    
    def _sync_call(self, *args, **kwargs):
        # Create a new run
        run = TaskRun(
            task_name=self.name,
            start_time=datetime.now(timezone.utc),
            inputs={**{f"arg{i}": arg for i, arg in enumerate(args)}, **kwargs},
        )
        
        with _task_run_context(run):
            try:
                # Execute the task
                start = time.perf_counter()

                if self.capture_logs:
                    # We still capture stdout but now we'll add it as an INFO log
                    stdout_capture = io.StringIO()
                    with redirect_stdout(stdout_capture):
                        result = self.func(*args, **kwargs)
                    
                    # Add captured stdout as INFO logs, one per line
                    captured_output = stdout_capture.getvalue()
                    if captured_output:
                        for line in captured_output.splitlines():
                            if line.strip():  # Skip empty lines
                                info(line)
                else:
                    result = self.func(*args, **kwargs)

                end = time.perf_counter()

                # Record successful completion
                run.end_time = datetime.now(timezone.utc)
                run.duration = end - start
                run.output = result

            except Exception as e:
                # Record error if task fails
                end = time.perf_counter()  # Calculate end time when error occurs
                run.end_time = datetime.now(timezone.utc)
                run.duration = end - start  # Set duration even when there's an error
                run.error = e
                # Capture the full traceback as a formatted string with linebreaks
                run.error_traceback = traceback.format_exc()
                # Add the error to logs as well
                error(str(e))
                raise

            finally:
                # Always add the run to history if this is a top-level task
                if _task_context.get() is run:
                    self.runs.append(run)
                    
                # Execute the callback if provided
                if self.callback is not None:
                    try:
                        # Convert TaskRun to dictionary before passing to callback
                        self.callback(run.to_dict())
                    except Exception as callback_error:
                        # Log but don't propagate callback errors
                        error_msg = f"Task callback error: {str(callback_error)}"
                        run._log("ERROR", error_msg)
                        
            return result

    @property
    def last_run(self) -> Optional[TaskRun]:
        """Returns the most recent run of this task"""
        return self.runs[-1] if self.runs else None

    def plot(self):
        return self.last_run.plot()
    
    def to_dataframe(self):
        return self.last_run.to_dataframe()

    def get_all_runs_history(self) -> List[Dict[str, Any]]:
        """Returns the complete history of all runs with their nested subtasks."""
        return [run.to_dict() for run in self.runs]

    async def _async_call(self, *args, **kwargs):
        # Create a new run
        run = TaskRun(
            task_name="CALLING: " + self.name,
            start_time=datetime.now(timezone.utc),
            inputs={**{f"arg{i}": arg for i, arg in enumerate(args)}, **kwargs},
        )
        
        # Need async context manager
        async with _async_task_run_context(run):
            try:
                # Execute the task
                start = time.perf_counter()
                
                if self.capture_logs:
                    # Handling logs in async is more complex
                    # Simplified version for now
                    result = await self.func(*args, **kwargs)
                else:
                    result = await self.func(*args, **kwargs)
                
                end = time.perf_counter()
                
                # Record successful completion
                run.end_time = datetime.now(timezone.utc)
                run.duration = end - start
                run.output = result
                
            except Exception as e:
                # Calculate duration even for errors
                end = time.perf_counter()
                run.end_time = datetime.now(timezone.utc)
                run.duration = end - start  # Set duration for failed tasks
                run.error = e
                run.error_traceback = traceback.format_exc()
                error(str(e))
                raise
                
            finally:
                # Same callback and bookkeeping logic
                if _task_context.get() is run:
                    self.runs.append(run)
                    
                if self.callback is not None:
                    try:
                        self.callback(run.to_dict())
                    except Exception as callback_error:
                        error_msg = f"Task callback error: {str(callback_error)}"
                        run._log("ERROR", error_msg)
                        
            return result


def task(
    func: Optional[Callable] = None,
    *,
    log: bool = True,
    retry_on: Optional[Union[Type[Exception], Tuple[Type[Exception], ...]]] = None,
    retry_attempts: Optional[int] = None,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Callable:
    """Decorator to mark a function as a trackable task.

    Args:
        func: The function to decorate
        log: If True, capture stdout during task execution
        retry_on: Exception or tuple of exceptions to retry on
        retry_attempts: Number of retry attempts
        callback: Function to call after task completion (success or failure)
                 The callback receives the task run data as a dictionary
    """

    def decorator(f: Callable) -> TaskDefinition:
        # Apply stamina retry if retry parameters are provided
        if retry_on is not None and retry_attempts is not None:
            # Create a wrapper that logs retries
            original_func = f
            
            # This will be called by stamina on each retry
            @wraps(original_func)
            def retry_wrapper(*args, **kwargs):
                current_run = _task_context.get()
                if current_run is not None:
                    current_run.retry_count += 1
                    warning(f"Retrying task (attempt {current_run.retry_count}/{retry_attempts}) after error")
                return original_func(*args, **kwargs)
            
            # Apply stamina retry to our wrapper
            f = stamina.retry(on=retry_on, attempts=retry_attempts)(retry_wrapper)
            
        return TaskDefinition(func=f, name=f.__name__, capture_logs=log, callback=callback)

    if func is None:
        return decorator
    return decorator(func)

def add_artifacts(**artifacts: Dict[str, Any]) -> bool:
    """Add artifacts to the currently running task.
    
    Args:
        artifacts: Dictionary of artifact name to artifact value
        
    Returns:
        True if artifacts were added successfully, False if no task is running
    """
    current_run = _task_context.get()
    if current_run is None:
        return False
    
    # Update the artifacts dictionary with the new artifacts
    current_run.artifacts.update(**artifacts)
    return True

def add_table(**table_items: Dict[str, Any]) -> bool:
    """Add artifacts to the currently running task.
    
    Args:
        artifacts: Dictionary of artifact name to artifact value
        
    Returns:
        True if artifacts were added successfully, False if no task is running
    """
    current_run = _task_context.get()
    if current_run is None:
        return False
    
    # Update the artifacts dictionary with the new artifacts
    current_run.table.update(**table_items)
    return True

def log(level: LogLevel, message: str) -> bool:
    """Add a log message to the currently running task.
    
    Args:
        level: Log level ("INFO", "WARNING", "ERROR", "DEBUG")
        message: The log message
        
    Returns:
        True if log was added successfully, False if no task is running
    """
    current_run = _task_context.get()
    if current_run is None:
        return False
    
    current_run._log(level, message)
    return True

# Convenience methods for different log levels
def info(message: str) -> bool:
    """Add an INFO level log message to the currently running task."""
    return log("INFO", message)

def warning(message: str) -> bool:
    """Add a WARNING level log message to the currently running task."""
    return log("WARNING", message)

def error(message: str) -> bool:
    """Add an ERROR level log message to the currently running task."""
    return log("ERROR", message)

def debug(message: str) -> bool:
    """Add a DEBUG level log message to the currently running task."""
    return log("DEBUG", message)
