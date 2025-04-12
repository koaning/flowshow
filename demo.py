import marimo

__generated_with = "0.12.8"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""Flowshow provides a `@task` decorator that helps you track and visualize the execution of your Python functions. Here's how to use it:""")
    return


@app.cell
def _():
    import time
    import random

    from flowshow import task, add_artifacts, info, debug


    # Turns a function into a Task, which tracks a bunch of stuff
    @task
    def my_function(x):
        info("This function should always run")
        time.sleep(0.5)
        add_artifacts({
            "foo": "Bar"
        })
        return x * 2

    # Tasks can also be configured to handle retries
    @task(retry_on=ValueError, retry_attempts=10)
    def might_fail():
        info("This function call might fail")
        time.sleep(1.0)
        if random.random() < 0.2:
            raise ValueError("oh no, error!")
        debug("The function has passed! Yay!")
        return "done"

    @task
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
        debug,
        info,
        main_job,
        might_fail,
        my_function,
        random,
        task,
        time,
    )


@app.cell
def _(main_job):
    main_job.last_run.to_dict()
    return


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
