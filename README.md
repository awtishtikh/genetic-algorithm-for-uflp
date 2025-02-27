﻿# Genetic Algorithm for Uncapacitated Facility Location Problem(UFLP).
This repository contains a Python implementation of a Genetic Algorithm (GA) to solve the Uncapacitated Facility Location Problem (UFLP). The UFLP is a classic optimization problem where we aim to determine which facilities to open and which facility should serve each customer to minimize the total cost.

Features

Random Data Generation: Generate facility costs and customer distances for testing.

Customizable Parameters: Configure the number of facilities, customers, population size, generations, and mutation rate.

Efficient Cost Calculation: Includes assignment costs and facility opening costs.

Genetic Algorithm Components:

Initialization: Randomly initializes the population of solutions.

Selection: Tournament-based parent selection.

Crossover: One-point crossover for creating new solutions.

Mutation: Bit-flipping mutation based on a given mutation rate.

Progress Visualization: Track progress across generations using the tqdm progress bar.

Best Solution Tracking: Records the best solution and its generation.
