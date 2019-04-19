             _____ _____ _____ _____ 
            |   __|  _  |   __|  _  |
            |  |  |     |  |  |     |
            |_____|__|__|_____|__|__|

Header-only parallel multi-objective genetic algorithm library written in C++14.

## Features
 - Extensible & highly customizable
 - Multi-objective 
 - Speciation 
 - Novelty search
 - Native C++ parralelism
 - No external dependencies
 - Meant to be easy to use and easy to install (header-only)
 - Advanced console logging and stats generations
 - Network distributed evaluations (cf. gagatools repository)

## Installation
Simply clone this repository, copy the directory in your project (or just the gaga.hpp file and the include folder) and include gaga.hpp in your project. Don't forget to enable c++14 when compiling.

## Basic Usage
### DNA & Individual
The main thing you need to provide is a valid DNA class, whose minimal requirements are:
 - a `std::string serialize()` method
 - a constructor taking the output of `serialize()`
 - a `void mutate()` method that mutates your dna
 - a `DNA crossover(const DNA &other)` method that returns an offspring
 - a `void reset()` method that resets your DNA before it can be used in a new evaluation

Internally, GAGA manipulates `Individuals` struct instances, whose raw dna member is accessible through `individual.dna`.

You can initialize your GAGA instance using your custom DNA:
```c++
GAGA::GA<DNA> ga;
```
Or if you plan on using novelty search with a custom footprint type `footprint_t`:
```c++
GAGA::GA<DNA, footprint_t> ga;
```

### Evaluator
An evaluator is a lambda function that takes an individual and sets its fitnesses (and footprints when novelty is enabled). It has to be passed to the GAGA instance through the `setEvaluator` method, which takes a reference to an individual and the id of the thread that is being used.
```c++
ga.setEvaluator([](auto &i, int procId) { 
	i.fitnesses["obj0"] = i.dna.doYourThing(); 
	i.fitnesses["obj1"] = i.dna.doYourOtherThing(); 
	// if novelty search is enabled:
	i.footprint = i.dna.computeFootprint();;
	// here we ignore procId but it could be useful for debuging, for example.
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
GAGA supports both MPI and native C++ thread based parallelism. For the natice c++ parallelisation (recommended on shared memory architectures), you need to set the number of threads using GAGA::setNbThreads(). Default is 1 thread.
If you need to use MPI parralelism (when running on a cluster for example), `#define GAGA_MPI_CLUSTER` before including gaga. You then need to link the MPI library of your choice (OpenMPI or IntelMPI for example) when compiling.

## Advanced customization
Although GAGA provides sane "basic" defaults, most key algorithmic steps of the Genetic Algorithm can be swapped and augmented. Here is a brief overview of the main blocks:
 - `newGenerationFunction()` is a lambda called at the begining of every `step()`. Default doesn't do anything, so feel free to use it as a hook.
 - `nextGenerationFunction()` is a lambda that acts as a wrapper for the main method, currently either `classic()` or `speciation()`. You can write your own main method.
 - `evaluate()` is a lambda used both by the classic and the speciation main methods. It calls the evaluator on every individuals, evaluating them in parallel using a threadpool. You could replace this if you know what you are doing. That's, for example, what the zero-mq distributed evaluation does. (Cf. gagatools)

## Main options and configuration methods
### General
 - `setPopSize(unsigned int)`: sets the number of individual in the population
 - `setMutationRate(double)`: sets the proportion of mutated individuals
 - `setCrossoverRate(double)`: sets the proportion of crossed individuals
 - `setSelectionMethod(const SelectionMethod&)`: specifies the selection method to use. (Available: paretoTournament, randomObjTournament)
 - `setTournamentSize(unsigned int)`: when a tournament based selection is used, changes the tournament size.
 - `setNbElites(unsigned int n)`: for each new generation, the n bests individuals will be preserved. (with multiple objectives, "best" can have different meanings depending on the current selection method) 
 - `setVerbosity(unsigned int)`: sets the verbosity level. 0: silent. 1: generation recaps. 2: 1 + individuals recaps. 3: 2 + various debug infos.
 - `setPopulation(const vector<Individual<DNA>>&)`: manually sets the population. Can be used to initialise the population at the begining of the evolutionnary run (although initPopulation is recommended), recovering from a save or just switching to a new population in the middle of a run.
 - `setNewGenerationFunction(std::function<void(void)>)`: sets a function that will be called before the current population is evaluated. The current population can be accessed through the public attribute GAGA::population, which is just a vector of individuals.

### Saving individuals and stats
 - `setSaveFolder(std::string)`: where to save the results (populations & stats). Default: "../evos".
 - `enablePopulationSave()` & `disablePopulationSave()`: enables/disables saving of the population in saveFolder. Default: enabled.
 - `setPopSaveInterval(unsigned int)`: interval at which the whole population should be saved (in nb of generation). Default: 1.
 - `enableAchiveSave()` & `disableArchiveSave()`: enables/disables saving of the novelty archive. (No effect when novelty is disabled). Default: false.
 - `setNbSavedElites(unsigned int)`: sets how many of the best individual gaga must save after each generation.
 - `setSaveParetoFront(bool)`: when true, saves the individuals on the pareto front to a file.
 - `setSaveGenStats(bool)`: when true, appends the general stats (best/worst/avg fitnesses, duration, ...) of each generation to a csv file
 - `setSaveIndStats(bool)`: when true, appends, after each generation, one line per individual to a csv file (generation number, individual ID, fitness values, evaluation duration)


### Novelty search
In order for novelty to be used, you need to provide a footprint (vector of vector of doubles) for each individuals (through the evaluator.
 - `enableNovelty()` & `disableNovelty()`: enables/disables novelty
 - `setKNN(unsigned int)`: number of neighbors to consider when computing the novelty of an individual. Default: 15.
 - `enableArchiveSave()` & `disableArchiveSave()`: enables/disables saving of the whole archive after each generation. The archive is filled at each generation with a certain number (nbOfArchiveAdditionsPerGeneration) of new individuals. These individuals are randomly selected (cf. "Devising Effective Novelty Search Algorithms: A Comprehensive Empirical Study")


### Speciation
When speciation is enabled, you have to provide a distance function that will returns the distance between two individuals. It is this distance that will be used to determine if two individuals belong to the same specie. Between each generation, if need be, you can access the list of species and the individuals in them through the public attribute GAGA::specie, which is a two dimensional vector of pointers to individuals. 
 - `setIndDistanceFunction(std::function<double(const Individual &, const Individual &)>)`: sets the distance function.
 - `enableSpeciation()/disableSpeciation()`: enable or disable speciation.
 - `setMinSpeciationThreshold(double)/setMaxSpeciationThreshold(double)`: sets the min and max genotypic distances for two individuals to be considered of the same species. The speciation threshold is indeed dynamic and will fulctuate between these 2 values.
 - `setSpeciationThreshold(double)`: sets the initial speciation threshold value.
 - `setSpeciationThresholdIncrement(double)`: the speed at which the speciation threshold will fluctuate.
 - `setMinSpecieSize(double)`: the minimum specie size.
 
## Logging
In addition to the verbosity options and the Generation + Individual stats (see above), some precious lineage information is stored in each individual. You can access every individual in the current generation through the `GAGA::population` container.
It is thus relatively trivial to create, for example, an ancestry tree for your evolutionary runs. 
In the gagatools repository, a SQLite wrapper will conveniently store all of this information in a neat sql satabase.

