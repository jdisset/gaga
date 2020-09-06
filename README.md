             _____ _____ _____ _____ 
            |   __|  _  |   __|  _  |
            |  |  |     |  |  |     |
            |_____|__|__|_____|__|__|

Header-only parallel multi-objective genetic algorithm library written in C++17.

## Features
 - Multi-objective 
 - No external dependencies, header only
 - Modular thanks to hooks defined at every important steps. Major features (speciation, novelty search, ...) are added through official extensions, and you can easily create your own.
 - Sane "invisible" defaults but highly customizable.
 - Native local C++ thread-based parralelism + Network distributed evaluations with ZeroMQ (clients in other languages are possible!)
 - Advanced logging and advanced stats (with SQLite support)

## Installation
Simply clone this repository in your project and include gaga.hpp. Enable c++17 when compiling. The examples directory contains some simple CMakeLists.txt.

## Basic Usage
### DNA & Individual
The main thing you need to provide is a valid DNA class.
A valid DNA class *MUST* have:
 - a `std::string serialize()` method
 - a constructor taking the output of `serialize()`

GAGA deals with `Individuals` instances; the population of the current generation is for example accessible through `GAGA::population`, and is a vector of Individuals (of type `Ind_t`).
The default Individual type used by GAGA encapsulates your raw DNA type, which you can easily access at anytime through `Ind_t::dna`.
A custom Individual type can be specified, and some extensions provide their individual type, as they can require different informations to be attached to an individual. For example, the novelty search module requires that each individual also have a signature, and therefore provides the `GAGA::NoveltyIndividual<DNA_t, signature_t>` type.

To initialize a simple GAGA instance using your custom DNA and the default Individual type:
```c++
GAGA::GA<DNA> ga;
```

### Mutation & Crossover
You need a Crossover and/or a Mutation operator.
By default, GAGA will try to call 
`DNA::mutate() // mutates the current DNA` and `DNA::crossover(const DNA& other) // returns an offspring DNA`
it these methods are defined in your DNA type. 
You can however manually define the mutation and crossover methods:
```c++
	GAGA::setMutateMethod([](DNA& dna){myMutation(dna)});
	GAGA::setCrossoverMethod([](const DNA& dna1, const DNA& dna2){return myCrossover(dna1, dna2);});
```

### Evaluator
An evaluator is a function that takes an individual and sets its fitnesses (and other necessary members such as its signature when novelty search is enabled). 
It's probably the most important function you have to provide. It has to be passed to the GAGA instance through the `setEvaluator` method. 
An evaluator takes a reference to an individual and the id of the thread that is being used (which you probably don't care about most of the time, but it is sometimes useful for some thread_safe constructs).

```c++
ga.setEvaluator([](auto &i, int procId) { 
	i.fitnesses["obj0"] = i.dna.doYourThing(); // we set the fitness value for objective "obj0"
	i.fitnesses["obj1"] = i.dna.doYourOtherThing(); // automatically becomes a multi-objective optimisation
});
```

### Population initialization
Once you have set all of the desired options (see below) for your evolutionary run, just initialize your first population using the `initPopulation` method, which takes a function returning a DNA.

```c++
	ga.initPopulation([]() { 
		return DNA::randomDNA();
	});
```

You can now run gaga
```c++
	const int nbGenerations = 200;
	ga.step(nbGenerations);
```

See the examples folder for usage of novely search and SQlite export.

## Parallelism
GAGA supports both native C++ thread based parallelism and ZeroMQ network distribution. For the native c++ parallelisation (recommended on shared memory architectures), you need to set the number of threads using `GAGA::setNbThreads()`.
With ZeroMQ based parallelisation, you can distribute the evaluation of the individuals on a local machine or on several networked machines. The GAGAZMQ extension provides C++ workers that you can run from the same program as the server, in different threads, or as their own independent program, potentially on different machine. A simple python worker is also provided, meaning you can evaluate individuals in python, as long as you are able to parse your own DNA.
More details and a working example are in tests/gagazmq_tests.hpp 



## Advanced customization
Although GAGA provides sane "basic" defaults, most key algorithmic steps of the Genetic Algorithm can be swapped and augmented. See examples folder.

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
c.f. examples/onemax/simple_novelty.cpp

### Speciation
With the speciation extension, you have to provide a distance function that will return the distance between two individuals. It is this distance that will be used to determine if two individuals belong to the same specie. Between each generation, if need be, you can access the list of species and the individuals. See speciation.hpp for more details.

## Logging
In addition to the verbosity options and the Generation + Individual stats, some precious lineage information is stored in each individual. You can access every individual in the current generation through the `GAGA::population` container.
c.f definition of the default Individual type in gaga.hpp:
```c++
template <typename DNA> struct Individual {
	DNA dna;
	std::map<std::string, double> fitnesses;  // std::map {"fitnessCriterName" -> "fitnessValue"}
	double evalTime = 0.0;
	std::pair<size_t, size_t> id{0u, 0u};  // gen id , ind id
	std::string infos;  // custom infos, description, whatever... filled by user
	std::map<string, double> stats;  // custom stats, filled by user
	std::vector<std::pair<size_t, size_t>> parents;  // vector of {{ generation id, individual id }}
	std::string inheritanceType = "exnihilo";  // inheritance type : mutation, crossover, copy, exnihilo
...
```

It is thus relatively trivial to create, for example, an ancestry tree for your evolutionary runs. 
The SQLite extension will conveniently store all of this information in a neat sql satabase. Check examples/onemax/simple_onemax.cpp and examples/onemax/simple_novelty.cpp for more details.

