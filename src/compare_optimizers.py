import time
import numpy as np
from random import choice, randint, random, shuffle
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Hyperparameter options
lr_choices = [0.001, 0.01]
unit_choices = [64, 128]
opt_choices = ['adam', 'rmsprop']
act_choices = ['relu', 'tanh']

# Generate a random individual
def random_individual():
    return [choice(unit_choices), choice(lr_choices), choice(opt_choices), choice(act_choices)]

# Mock model evaluator (simulates accuracy)
def mock_evaluate_model(params):

    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_data = np.concatenate([x_train, x_test]).astype('float32') / 255.0
    y_data = to_categorical(np.concatenate([y_train, y_test]), 10)

    x_data = x_data.reshape(-1, 28 * 28)  # Flatten images
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2)
    
    units, lr, optimizer, activation = params
    try:
        # Build a deeper neural network
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(int(units), activation=activation, input_shape=(784,)),
            tf.keras.layers.Dense(int(units / 2), activation=activation),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.get({
                'class_name': optimizer,
                'config': {'learning_rate': float(lr)}
            }),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train for more epochs to better differentiate configurations
        model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=0)

        # Evaluate on validation set
        _, accuracy = model.evaluate(x_val, y_val, verbose=0)
        return accuracy
    except:
        return 0.0  # Fallback in case of failure


# Adaptive Grouping-Based GA
def run_ag_ga():
    population_size = 10
    generations = 5
    initial_group_size = 2
    mutation_prob = 0.3
    population = [random_individual() for _ in range(population_size)]
    global_best_score = 0
    global_best_config = None
    group_size = initial_group_size
    scores_by_gen = []
    start_time = time.time()

    for _ in range(generations):
        shuffle(population)
        groups = [population[i:i + group_size] for i in range(0, len(population), group_size)]
        best_in_groups = []

        for group in groups:
            fitnesses = [mock_evaluate_model(ind) for ind in group]
            best_idx = int(np.argmax(fitnesses))
            best = group[best_idx]
            best_score = fitnesses[best_idx]
            best_in_groups.append(best)

            if best_score > global_best_score:
                global_best_score = best_score
                global_best_config = best

        scores_by_gen.append(global_best_score)

        # Crossover + Mutation among best
        new_population = []
        while len(new_population) < population_size:
            p1, p2 = choice(best_in_groups), choice(best_in_groups)
            child = p1[:2] + p2[2:]
            if random() < mutation_prob:
                idx = randint(0, 3)
                if idx == 0:
                    child[idx] = choice(unit_choices)
                elif idx == 1:
                    child[idx] = choice(lr_choices)
                elif idx == 2:
                    child[idx] = choice(opt_choices)
                else:
                    child[idx] = choice(act_choices)
            new_population.append(child)

        population = new_population
        group_size = min(len(population), group_size + 1)

    duration = time.time() - start_time
    return {
        'algorithm': 'AG-GA',
        'best_score': global_best_score,
        'best_config': global_best_config,
        'time': duration,
        'convergence': scores_by_gen
    }

# Standard Genetic Algorithm
def run_ga():
    population_size = 10
    generations = 5
    mutation_prob = 0.3
    population = [random_individual() for _ in range(population_size)]
    best_score = 0
    best_config = None
    convergence = []
    start_time = time.time()

    for _ in range(generations):
        fitnesses = [mock_evaluate_model(ind) for ind in population]
        paired = list(zip(fitnesses, population))
        paired.sort(key=lambda x: x[0], reverse=True)
        sorted_pop = [x[1] for x in paired]
        best_score = max(best_score, paired[0][0])
        best_config = paired[0][1]
        convergence.append(best_score)

        # Elitism + crossover
        new_population = sorted_pop[:2]
        while len(new_population) < population_size:
            p1, p2 = choice(sorted_pop[:5]), choice(sorted_pop[:5])
            child = p1[:2] + p2[2:]
            if random() < mutation_prob:
                idx = randint(0, 3)
                if idx == 0:
                    child[idx] = choice(unit_choices)
                elif idx == 1:
                    child[idx] = choice(lr_choices)
                elif idx == 2:
                    child[idx] = choice(opt_choices)
                else:
                    child[idx] = choice(act_choices)
            new_population.append(child)
        population = new_population

    duration = time.time() - start_time
    return {
        'algorithm': 'GA',
        'best_score': best_score,
        'best_config': best_config,
        'time': duration,
        'convergence': convergence
    }

# Basic Particle Swarm Optimization (PSO) Simulation
def run_pso():
    swarm_size = 10
    generations = 5
    velocity_prob = 0.2
    particles = [random_individual() for _ in range(swarm_size)]
    personal_best = particles[:]
    personal_best_scores = [mock_evaluate_model(p) for p in particles]
    global_best = personal_best[np.argmax(personal_best_scores)]
    global_best_score = max(personal_best_scores)
    convergence = [global_best_score]
    start_time = time.time()

    for _ in range(generations - 1):
        for i in range(swarm_size):
            if random() < velocity_prob:
                idx = randint(0, 3)
                if idx == 0:
                    particles[i][idx] = choice(unit_choices)
                elif idx == 1:
                    particles[i][idx] = choice(lr_choices)
                elif idx == 2:
                    particles[i][idx] = choice(opt_choices)
                else:
                    particles[i][idx] = choice(act_choices)

            score = mock_evaluate_model(particles[i])
            if score > personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best[i] = particles[i]
            if score > global_best_score:
                global_best_score = score
                global_best = particles[i]

        convergence.append(global_best_score)

    duration = time.time() - start_time
    return {
        'algorithm': 'PSO',
        'best_score': global_best_score,
        'best_config': global_best,
        'time': duration,
        'convergence': convergence
    }

ag_ga_result = run_ag_ga()
ga_result = run_ga()
pso_result = run_pso()

from pprint import pprint
pprint(ag_ga_result)
pprint(ga_result)
pprint(pso_result)
