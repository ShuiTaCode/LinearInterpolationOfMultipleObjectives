# Linear Interpolation of Multiple Objectives (LIMO)

LIMO is a multiple-policy approach for Multi-Objective Reinforcement Learning (MORL)
problems with conflicting objectives. This application is used to interactively demonstrate the procedure of LIMO in two well known MORL domains. It is also able to generate plots from data and even LATEX-Code.

## Installation

The application requires (was tested on) [Python 3.6](https://www.python.org/downloads/release/python-360/) as well as the libraries [numpy](https://numpy.org/install/) and [matplotlib](https://matplotlib.org/3.5.1/).You can simply install them in the directory of this README.md file using [pip](https://pypi.org/project/pip/):
```
pip install numpy
```
```
pip install matplotlib
```
After this you can start the main.py file the directory of this README.md file using:

```
python main.py
```

## Getting started

To get started, simply click the "run (single preference)"-Button and wait a few seconds. What you now see:
1. Left canvas: This canvas visualizes the first step of the LIMO approach, which is to find optimal value-functions for each objective. Each of these two Gridworlds represents an objective according to the current environment. Per default, this is the Cliffowrld-Environment The upper grid shows the optimal-value function for the "avoid the cliff"-objective. The one below for the "shortest goal-path"-objective
2. Middle canvas: The middle canvas shows LIMO (top) and the "usual" MOMDP result (below) for a specific preference. In this case (alpha = 0) the complete priority is on the "shortest goal-path"-objective.  
3. Right canvas: Per default, clicking the "run (single preference)"-Button causes a Monte Carlo Simulation where the agent completes 1000 episodes in the Cliffworld-Environment. With every transitions the agent increases the "counter" of the state it is leaving. This right canvas shows a heatmap, visualizing the "counter" along those 1000 episodes.

## Usage
With this application, you can test LIMO in two different domains and compare it to the usual approach. Step by step, we will now demonstrate LIMO:
1. **Find optimal value-functions for each objective**: By pressing the "Increment Iteration" or "Solve MDPs"-Button, you produce an optimal value-functions for each of the two MDPs, visualized in the left canvas
2. **Create new value-functions using a linear combination**: To perform a linear combination, simply **change the "alpha"-value**. This will automatically trigger a linear combination for the current preference defined through alpha. The resulting value-function (and policy) is visualized in the middle canvas (upper region). The action will also trigger the scalarization of the Cliffworld-MOMDP defined through alpha and solve the resulting MDP using Value Iteration. The resulting value-function is shown below the previous.
3. **Compare LIMO to the usual approach**: First select your desired approach from the "Approach"-Dropdown. Then click the "run (multiple preferences)"-Button. Depending on the number of episodes you should see a plot within a few minutes. It shows the averaged reward obtained from the episodes for different preferences (default 20). It is compared to the initial-state value of the corresponding MDP (LIMO or MOMDP). You can do this for both approaches and compare them to each other.

You can also change further variables, such as the noise-factor, in the "LIMOconfig.py", which is placed in the current directory.

## Documentation

This little documentation explains usage and meaning of every UI component.
1. Select Environment: The "Select Environment"-Dropdown points at the currently selected environment. The two available options are the "Cliffworld" and the "Deep Sea Treasure"-Environment. Switching environment will clear all heatmap data and calculations.
2. Alpha: The "alpha"-Inputfield is used to define the prioritization of the first objective. In the Cliffworld, this would be the "avoid the cliff"-objective. In the Deep Sea Treasure, it is the "time minimization"-objective. Note that e.g. alpha = 80% also implicates that the other objective gets a prioritization of (100-80)% = 20%. Changing the "alpha" value will instantly **calculate the both approaches for the new preference**
3. Gamma: The "gamma"-Inputfield is used to define the discount-rate of the agent during the simulation, as well as for the "Value Iteration"-Algorithm. Changing the "gamma" value will also **calculate the both approaches for the new value**
4. Size: The "size"-Inputfield measures the side-length of our grid. After changing this value, you need to press the "resize MDPs"-Button for your changes to take effect.
5. Increment Iteration: The "Increment Iteration"-Button is used to visualize the only training aspect of the LIMO approach: Estimating the optimal value-function for each objective. With every click you can run through the "Value Iteration"-Algorithm and see how the optimal value-function is being calculated. Note that this feature is not repeatable as long as you don't switch environment or restart the application.
6. Solve Mdps: The "Solve Mdps"-Button is repeating the previous explained action until convergence. 
7. Number of Episodes: The "Number of Episodes"-Inputfield declares how many episodes the simulation contains.
8. Approach: The "Approach"-Dropdown lets you select the approach for the Experiment.
9. Run (single preference): The "Run (single preference)"-Button lets the agent completes multiple episodes in a Monte Carlo Simulation. The approach as well as the number of episodes can be defined through the UI as explained so far. This action will **produce a heatmap** after the simulation finished.
10. Run (multiple preferences): The "Run (multiple preferences):"-Button does the same as the previous button iteratively for multiple preferences. Per default, those preferences (alpha) follow 0% 5% 10% ... 100%, but they can be changed in the configuration. This action will **produce a plot**, where the average return of the agent during the experiment is compared to the value of the initial-state (expected return).
11. Print Latex Comparison: This action produces LATEX sourcecode to visualise the value-function, policy as well as the heatmap for both approaches. To display the figure, it requires the LATEX package [TikZ](https://de.overleaf.com/learn/latex/TikZ_package). Change the LATEX_PRINT_MODE in the configuration to 'multiple', this action will print the comparison for every preference.


