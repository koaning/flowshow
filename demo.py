import marimo

__generated_with = "0.12.8"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md("""Flowshow provides a `@task` decorator that helps you track and visualize the execution of your Python functions. Here's how to use it:""")
    return


@app.cell
def _():
    import time
    import random

    from flowshow import task, add_artifacts, info, debug, warning, error

    d = {}

    def store(b):
        global d
        d = b 

    # Turns a function into a Task, which tracks a bunch of stuff
    @task
    def my_function(x):
        info("This function should always run")
        time.sleep(0.2)
        add_artifacts(foo=1, bar=2)
        return x * 2

    # Tasks can also be configured to handle retries
    @task(retry_on=ValueError, retry_attempts=5)
    def might_fail():
        info("This function call might fail")
        time.sleep(0.2)
        my_function(2)
        raise ValueError("oh noes")
        debug("The function has passed! Yay!")
        return "done"

    @task(callback=store)
    def main_job():
        info("This output will be captured by the task")
        for i in range(3):
            my_function(10)
            might_fail()
        return "done"

    # Run like you might run a normal function
    _ = main_job()
    return (
        add_artifacts,
        d,
        debug,
        error,
        info,
        main_job,
        might_fail,
        my_function,
        random,
        store,
        task,
        time,
        warning,
    )


@app.cell
def _(main_job):
    main_job.last_run.to_dict()
    return


@app.cell
def _(main_job, mo, template):
    mo.iframe(template.render(data=main_job.last_run.to_dict()))
    return


@app.cell
def _():
    from pathlib import Path 
    from jinja2 import Template 

    template = Template(Path("index.jinja2").read_text())

    # mo.iframe(template.render(data=run_many_nested.last_run.to_dict()))
    return Path, Template, template


@app.cell
def _(d):
    d
    return


@app.cell
def _(main_job):
    main_job.last_run.to_dict()['error_traceback']
    return


@app.cell
async def _(error, info, task, time, warning):
    import asyncio

    @task
    async def async_sleep(seconds: float, name: str) -> str:
        """Asynchronous sleep function that returns a message after completion"""
        info("it works, right?")
        await asyncio.sleep(seconds)
        info("it did!")
        return f"{name} finished sleeping for {seconds} seconds"

    @task
    async def run_concurrent_tasks():
        """Run multiple sleep tasks concurrently"""
        start_time = time.time()

        # Create multiple sleep tasks
        tasks = [
            async_sleep(2, "Task 1"),
            async_sleep(1, "Task 2"),
            async_sleep(3, "Task 3")
        ]

        # Run tasks concurrently and gather results
        results = await asyncio.gather(*tasks)

        end_time = time.time()
        total_time = end_time - start_time

        # Return results and timing information
        return {
            "results": results,
            "total_time": f"Total execution time: {total_time:.2f} seconds"
        }

    @task 
    async def run_many_nested():
        info("About to start task 1")
        await run_concurrent_tasks()
        info("About to start task 2")
        await run_concurrent_tasks()
        warning("They both ran!")
        error("They both ran!")

    await run_many_nested()
    return async_sleep, asyncio, run_concurrent_tasks, run_many_nested


@app.cell
def _(main_job):
    out = main_job.to_dataframe()
    return (out,)


@app.cell
def _(out):
    out
    return


@app.cell(hide_code=True)
def _(main_job, mo):
    chart = mo.ui.altair_chart(main_job.plot())
    chart
    return (chart,)


@app.cell(hide_code=True)
def _(chart):
    if chart.value["logs"].shape[0] > 0:
        print(list(chart.value["logs"])[0])
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
