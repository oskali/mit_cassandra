# covid19_team2 : 


# Markov Decision Process Model Description:

The `MDP Model` Markov Decision Process Model for Covid Prediction applies a Markov Decision Process inference model (confer. ) in order to capture the dynamic of time series of the growth rate across regions (states `states`, county `fips` as well as zip code `zipcode`.

## Model assumptions :

### Learning a Markov Chain
The model relies on the assumption that the growth rates of the number of cases / deaths follows a Descrete Markov Decision Process that is stationary across regions. 
- The `STATES` (also called `clusters`) of the Markov Decision Process represent a approximation of the minimum representation of the discrete graph. The `STATES` are then distributed across regions (ex : `Massachusetts`) and time (ex `2019-05-05`).
The models uses some features in order to facilitate / improve the population of the states of the markovian process.

### Learning from actions (policy changing): causation and counterfactual modelling
We incorporated an additional feature that is thought to affect the distribution of the growth rates. Looking at discretized changes on this variable, we defined a set of actions (`actions`), that allowed to improve the complexity of the modelling, from a Markov Chain to a deterministic Markov Decision Process.

### Completeness of the MDP
One additional extension of the model consisted assuming that the Discrete Markov Process is complete, i.e. for any given `state` and `action`, we can make a prediction.
This evolves either:
- providing an algorithm to complete the transition matrix
or :
- providing an algorithm to locally estimates a next cluster and a adjusted growth rate

## Application : Case Predictions using the MDP, model specifications
In order to learn the growth rates through the MDP, we preprocessed the data and specified the following : 
- `target` : compute an empirical estimate of the expected growth rate over 3 days and we hold it fixed every 3 days
ex : `[1.1, 1.2, 1.3, 1.4, 1.5, 1.6]` becomes `[1.2, 1.5]`
- `ACTION` : actions are defined by the weekly changes in a standardized mobility data (ex : `workplace`). We used quantile thresholds to discretize the actions
ex : `[q(workplace_chg_7days, 5%) q(workplace_chg_7days, 95%)]` defines actions `[-1, 0, 1]`
