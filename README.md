
# Adaptive Grouping-Based Genetic Algorithm (AG-GA)

This repository contains the implementation of the **Adaptive Grouping-Based Genetic Algorithm (AG-GA)** for hyperparameter optimization of neural networks. The AG-GA is compared with the traditional **Genetic Algorithm (GA)** and **Particle Swarm Optimization (PSO)** algorithms.

## Overview
- **AG-GA**: A novel genetic algorithm that uses adaptive grouping for better exploration and optimization.
- **GA**: Standard Genetic Algorithm for comparison.
- **PSO**: Basic Particle Swarm Optimization for comparison.

## Algorithms Compared
1. **AG-GA** (Adaptive Grouping Genetic Algorithm)
2. **GA** (Genetic Algorithm)
3. **PSO** (Particle Swarm Optimization)

## Results
The algorithms are compared using the **MNIST** dataset for neural network hyperparameter tuning. Results are stored in `results/` and include accuracy for each algorithm's performance.

## Purpose of the Project
The goal of this project is to introduce the **Adaptive Grouping-Based Genetic Algorithm (AG-GA)** as a new optimization technique for hyperparameter tuning in neural networks. The method aims to improve upon traditional optimization algorithms (like GA and PSO) by using adaptive grouping to enhance exploration and convergence.

This is an ongoing project that is a part of my research in the field of AI and optimization techniques. The project is being actively worked on and will be submitted to a **conference or journal** at a later stage.

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Install dependencies:

```bash
pip install -r requirements.txt
```

### Usage
Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/ag-ga.git
cd ag-ga
```

Run the experiment:

```bash
python ag_ga_experiment.py
```

The results will be saved in the `results/` directory.

## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Future Plans
This project is a work in progress. A full research paper comparing the performance of AG-GA, GA, and PSO on hyperparameter optimization will be submitted to a conference or journal in the future.

## Acknowledgements
- **MNIST Dataset**: The MNIST dataset is used for evaluating the performance of the algorithms. It is a well-known dataset in the machine learning community, originally created by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges. [More Info](http://yann.lecun.com/exdb/mnist/)
  
- **Genetic Algorithm (GA)**: The standard Genetic Algorithm was used as a benchmark in this project. Special thanks to the open-source contributors who have made GA implementations available, which served as the basis for comparison in this work.

- **Particle Swarm Optimization (PSO)**: PSO was included as a benchmark algorithm. It is a widely used optimization algorithm, originally introduced by James Kennedy and Russell C. Eberhart in 1995. [More Info on PSO](https://en.wikipedia.org/wiki/Particle_swarm_optimization)

- **Contributors**: Thanks to the contributors of the open-source repositories and frameworks for their support in implementing and providing algorithms for GA and PSO. These resources helped in creating fair comparisons for this research.
