from datetime import datetime
from typing import Dict, List, Optional


def flatten_tasks(task: Dict, parent_name: Optional[str] = None, counters: Optional[Dict[str, int]] = None) -> List[Dict]:
    """
    Recursively flatten the nested task structure into a list of dictionaries.
    
    Args:
        task: Dictionary containing task information
        parent_name: Name of the parent task if any
        counters: Dictionary to keep track of task name counts
        
    Returns:
        List of flattened task dictionaries
    """
    if counters is None:
        counters = {}
    
    tasks = []
    
    # Update counter for this task name
    task_key = f"{parent_name}_{task['task_name']}" if parent_name else task['task_name']
    if task_key not in counters:
        counters[task_key] = 0
    else:
        counters[task_key] += 1
    
    # Create a task entry with counter
    task_dict = {
        'task_name': f"{task_key}_{counters[task_key]}",
        'start_time': datetime.fromisoformat(task['start_time']),
        'end_time': datetime.fromisoformat(task['end_time']),
        'duration': task['duration'],
        'logs': task.get('logs', '')
    }
    tasks.append(task_dict)
    
    # Process subtasks if they exist
    if 'subtasks' in task:
        for subtask in task['subtasks']:
            tasks.extend(flatten_tasks(subtask, task['task_name'], counters))
    
    return tasks


