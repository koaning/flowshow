import marimo

__generated_with = "0.10.19"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from flowshow import task

    return mo, task


@app.cell
def _(task):
    import time
    import datetime as dt

    def loggg(msg):
        print(f"{dt.datetime.now()} - {msg}")

    @task
    def load_data(a):
        loggg("loading the data")
        time.sleep(0.6)
        loggg("totally loaded, aye!")
        return [a for i in range(10)]

    @task
    def merge(a, b):
        loggg("about to merge")
        time.sleep(0.4)
        loggg("merging")
        loggg("done")
        return zip(a, b)

    @task
    def doit():
        loggg("starting doit")
        time.sleep(0.5)
        a = load_data(1)
        b = load_data(2)
        loggg("it done!")

    @task
    def main():
        for i in range(2):
            time.sleep(0.7)
            loggg("starting job 1")
            doit()
            loggg("starting job 2")
            doit()
            loggg("done")
            return "done !"

    result = main()
    return doit, dt, load_data, loggg, main, merge, result, time


@app.cell
def _(main, mo):
    chart = mo.ui.altair_chart(main.last_run.plot())
    chart
    return (chart,)


@app.cell
def _(chart):
    if chart.value["logs"].shape[0] > 0:
        print(list(chart.value["logs"])[0])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
