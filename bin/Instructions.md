# Instructions for the user
---
# Requirements
Before starting to explain how to execute the algorithm for finding frequent topics in tweets, it is necessary to explain which are the prerequisites for the execution.

First of all, the latest version of Python should be installed in the user device. Secondly, some packages are explicitly fundamental for the script execution, which are contained inside the file `requirements.txt`. It is possible to automatically install them all with a single command to execute inside the terminal, placed inside the `bin` folder:

```{python}
pip install -r requirements.txt
```

Notice that, if the user desires to execute also other py scripts contained inside the `src` folder, additional packages are necessary, which are specified in `src/requirements.txt` and can be installed in the same way by executing the command in the different directory.

## Execution
After all requirements are installed, the user should open a terminal inside the directory `bin` and execute `find_topics.py` by writing:

```{python}
python find_topics.py
```

However, the execution described above uses default parameters, set as:

- **support = 0.02** (proportion of tweets in which the topic can be found);
- **lines = 0** (number of lines that will be visualized in the terminal at the end of the execution, as preview);
- **days = 1** (minimum number of days in which we can find the topic);
- **min_items = 1** (minimum number of words inside a topic);
- **max_items = 4** (maximum number of words inside a topic).

If the user wants to specify customized parameters, it is possible by writing:

```{python}
python find_topics.py support=0.05 lines=10 days=2 min_items=1 max_items=3
```
No variables except for those illustrated above should be entered and the one specified must have a numerical value (i.e. only digits). In particular, the support should assume a value between 0 and 1. However, it should be a number between 0.05 and 0.01:

- if higher than 0.05, we would obtain few or no results;
- if lower than 0.01, we would encounter a high computational cost, therefore the script execution would require some minutes to end.

Notwithstanding the possibility to customize parameters to filter results, it is also possible to omit one or more of them or insert them in a different order compared to the one specified.

## Results

In case the parameter **lines** is omitted (=0), no preview will be provided; if it has a value higher than the total number of lines in the dataset returned, this will be showed entirely.

The output will be saved inside the path `data/output_files/`, with names `output.csv`, which is the readable format. The other format can be used for additional analysis and imported through `pickle` package, since it maintains data types of columns without the necessity of parsing them.
