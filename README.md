             _____ _____ _____ _____ 
            |   __|  _  |   __|  _  |
            |  |  |     |  |  |     |
            |_____|__|__|_____|__|__|

Header-only parallel multi-objective genetic algorithm library written in C++14.

## Installation
Simply clone this repository and include gaga.hpp in your project. Don't forget to set the -std=c++14 flag when compiling.

## Usage
### DNA & Individual
The main thing you need to provide is a valid DNA class, whose minimal requirements are:
 - a `std::string serialize()` method
 - a constructor taking the output of `serialize()`
 - a `void mutate()` method that mutates your dna
 - a `DNA crossover(const DNA &other)` method that returns an offspring
 - a `void reset()` method that resets your DNA before it can be used in a new evaluation

Internally, GAGA manipulates `Individuals<DNA>` struct instances, whose raw dna member is accessible through `individual.dna`.

You can then initialize your GAGA instance using your custom DNA (command lines argument are required):
```c++
GAGA::GA<DNA> ga(argc, argv);
```

### Evaluator
An evaluator is a lambda function that takes an individual and sets its map (and footprints when novelty is enabled). It has to be passed to the GAGA instance through the `setEvaluator` method.
```c++
ga.setEvaluator([](auto &i) { 
	i.fitnesses["obj0"] = i.dna.doYourThing(); 
	i.fitnesses["obj1"] = i.dna.doYourOtherThing(); 
});
```
### Population initialization
Once you have set all of the desired options (see below) for your evolutionary run, just initialize your first population using the `initPopulation` method, which takes a lambda returning a DNA.

```c++
	ga.initPopulation([]() { 
		DNA dna;
		dna.randomInit();
		return dna; 
	});
```

You can now run gaga
```c++
	const int nbGenerations = 200;
	ga.step(nbGenerations);
```
See tests for more examples.

## Parallelism
GAGA supports both MPI and OpenMP based parallelism. For OpenMP parallelisation (recommended on shared memory architectures), you need to `#define OMP` before including gaga's header (don't forget to compile with the -fopenmp flag).
If you need to use MPI parralelism (when running on a cluster for example), `#define CLUSTER` before including gaga. You then need to link the MPI library of your choice (OpenMPI or IntelMPI for example) when compiling.

## Options
### General
 - `setMutationProba(double)`: sets the probability for an individual to be mutated.
 - `setCrossoverProba(double)`: sets the probability that a crossover will be happening.
 - `setSelectionMethod(const SelectionMethod&)`: specifies the selection method to use. (Available: paretoTournament, randomObjTournament)
 - `setTournamentSize(unsigned int)`: when a tournament based selection is used, changes the tournament size.
 - `setNbElites(unsigned int n)`: for each new generation, the n bests individuals will be preserved. (with multiple objectives, "best" can have different meanings depending on the current selection method) 
 - `setVerbosity(unsigned int)`: sets the verbosity level. 0: silent. 1: generation recaps. 2: 1 + individuals recaps. 3: 2 + various debug infos.
 - `setPopulation(const vector<Individual<DNA>>&)`: manually sets the population.

### Saving individuals
 - `setSaveFolder(std::string)`: where to save the results (populations & stats). Default: "../evos".
 - `enablePopulationSave()` & `disablePopulationSave()`: enables/disables saving of the population in saveFolder. Default: enabled.
 - `setPopSaveInterval(unsigned int)`: interval at which the whole population should be saved (in nb of generation). Default: 1.
 - `enableAchiveSave()` & `disableArchiveSave()`: enables/disables saving of the novelty archive. (No effect when novelty is disabled). Default: false.
 - `setNbSavedElites(unsigned int)`: sets how many of the best individual gaga must save after each generation.

### Novelty
In order for novelty to be used, you need to provide a footprint (vector of vector of doubles) for each individuals (through the evaluator.
 - `enableNovelty()` & `disableNovelty()`: enables/disables novelty
 - `setKNN(unsigned int)`: number of neighbors to consider when computing the novelty of an individual. Default: 15.
 - `setMinNoveltyForArchive(double)`: novelty (average distance to the KNN) above which an individual is saved in the archive.
 - `enableArchiveSave()` & `disableArchiveSave()`: enables/disables saving of the whole archive after each generation.
