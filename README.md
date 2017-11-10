# evolve-dfs
A genetic algorithm to optimize your baseball and football daily fantasy sports lineups. Currently supports lineups optimized for draft kings.  Optimization is limited by your projections... be warned. The actual GA is modified from https://github.com/remiomosowon/pyeasyga

## Getting Started

```
git clone https://github.com/conorkcorbin/evolve-dfs.git
```

## Python Environment

Python 3.6: requires numpy and pandas

## How to Use

Player projection data can be found in the NFL and MLB folders in this repo.  The projections are taken from rotogrinders: https://rotogrinders.com/projected-stats/nfl-qb?site=draftkings

To create new lineups simply download projections and format them into columns like shown in the example .csv files I have uploaded. This tool works independently of the projections that are used. 

generate_lineups.py needs to be editted to load whatever projections sheet you create. It supports football and baseball. Simply uncomment the code corresponding to whatever sport you want to run. Default - runs football. 

You can play with parameters like population size, number of generations, and mutation rate.

Use: 

```
python generate_lineups.py
```

## Authors

* **Conor K Corbin** 


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details





