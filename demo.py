import marimo

__generated_with = "0.12.9"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md("""Flowshow provides a `@task` decorator that helps you track and visualize the execution of your Python functions. Here's how to use it:""")
    return


@app.cell
def _():
    import time
    import random
    from pydantic import BaseModel
    from typing import List
    from flowshow import task, add_artifacts, info, debug, warning, error, span
    return (
        BaseModel,
        List,
        add_artifacts,
        debug,
        error,
        info,
        random,
        span,
        task,
        time,
        warning,
    )


@app.cell
def _(BaseModel, List, add_artifacts, debug, info, span, task, time):
    class Foobar(BaseModel):
        x: int
        y: int
        saying: str

    class ManyBar(BaseModel):
        desc: str
        stuff: List[Foobar]

    @task
    def many_things(many: ManyBar):
        info("This runs for demo purposes")

    @task
    def my_function(x):
        info("This function should always run")
        time.sleep(0.2)
        add_artifacts(foo=1, bar=2, buz={"hello": "there"})
        return x * 2

    @task(retry_on=ValueError, retry_attempts=5)
    def might_fail():
        info("This function call might fail")
        time.sleep(0.2)
        my_function(2)
        # raise ValueError("oh noes")
        debug("The function has passed! Yay!")
        return "done"

    @task()
    def main_job():
        info("This output will be captured by the task")
        add_artifacts(manybar=ManyBar(desc="hello", stuff=[Foobar(x=1, y=2, saying="ohyes")]))
        with span("hello") as s:
            info("test test")
            with span("foobar") as f:
                info("whoa whoa")

        for i in range(3):
            my_function(10)
            might_fail()
        return "done"

    # Run like you might run a normal function
    _ = main_job()
    return Foobar, ManyBar, main_job, many_things, might_fail, my_function


@app.cell
def _(main_job):
    main_job.last_run.to_dict()['artifacts']
    return


@app.cell
def _(main_job, mo):
    mo.iframe(main_job.last_run.render())
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
def _(mo, run_many_nested):
    mo.iframe(run_many_nested.last_run.render())
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
