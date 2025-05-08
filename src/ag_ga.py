import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

# Load and preprocess MNIST dataset
(x_train, y_train), (x_val, y_val) = mnist.load_data()
x_train, x_val = x_train / 255.0, x_val / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_val = tf.keras.utils.to_categorical(y_val, 10)

# Genetic Algorithm Hyperparameters
POPULATION_SIZE = 20
INITIAL_GROUP_SIZE = 4
MAX_GENERATIONS = 30
FITNESS_THRESHOLD = 0.97  # Early stopping threshold

# Possible hyperparameter choices
activation_functions = ['relu', 'tanh']
optimizers = ['adam', 'sgd', 'rmsprop']
learning_rates = [0.01, 0.001, 0.0001]
hidden_units = [32, 64, 128]

# Chromosome encoding: [activation_idx, optimizer_idx, lr_idx, hidden_units_idx]
def generate_chromosome():
    return [
        random.randint(0, len(activation_functions) - 1),
        random.randint(0, len(optimizers) - 1),
        random.randint(0, len(learning_rates) - 1),
        random.randint(0, len(hidden_units) - 1)
    ]

def decode_chromosome(chrom):
    return {
        'activation': activation_functions[chrom[0]],
        'optimizer': optimizers[chrom[1]],
        'lr': learning_rates[chrom[2]],
        'units': hidden_units[chrom[3]]
    }

def build_model(config):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(config['units'], activation=config['activation']),
        Dense(10, activation='softmax')
    ])

    if config['optimizer'] == 'adam':
        optimizer = Adam(learning_rate=config['lr'])
    elif config['optimizer'] == 'sgd':
        optimizer = SGD(learning_rate=config['lr'])
    else:
        optimizer = RMSprop(learning_rate=config['lr'])

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def fitness_function(chrom):
    config = decode_chromosome(chrom)
    model = build_model(config)
    history = model.fit(x_train, y_train, epochs=2, batch_size=128, verbose=0,
                        validation_data=(x_val, y_val))
    val_acc = history.history['val_accuracy'][-1]
    return val_acc

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

def mutate(chrom):
    idx = random.randint(0, len(chrom) - 1)
    gene_space = [activation_functions, optimizers, learning_rates, hidden_units]
    gene_space_idx = [len(activation_functions), len(optimizers), len(learning_rates), len(hidden_units)]
    chrom[idx] = random.randint(0, gene_space_idx[idx] - 1)
    return chrom

def evolve_population(pop, group_size):
    random.shuffle(pop)
    groups = [pop[i:i + group_size] for i in range(0, len(pop), group_size)]

    next_gen = []
    global_best = None
    global_best_score = -1

    for group in groups:
        scored = [(chrom, fitness_function(chrom)) for chrom in group]
        scored.sort(key=lambda x: x[1], reverse=True)

        best_chrom, best_score = scored[0]
        if best_score > global_best_score:
            global_best = best_chrom
            global_best_score = best_score

        # Top 2 from group for crossover
        top_chroms = [chrom for chrom, _ in scored[:2]]
        new_group = []
        while len(new_group) < len(group):
            p1, p2 = random.choices(top_chroms, k=2)
            c1, c2 = crossover(p1, p2)
            new_group.extend([mutate(c1), mutate(c2)])
        next_gen.extend(new_group[:len(group)])

    return next_gen[:POPULATION_SIZE], global_best, global_best_score

def run_ga():
    population = [generate_chromosome() for _ in range(POPULATION_SIZE)]
    group_size = INITIAL_GROUP_SIZE
    best_overall = None
    best_score_overall = -1

    for generation in range(MAX_GENERATIONS):
        population, gen_best, gen_score = evolve_population(population, group_size)
        print(f"Gen {generation+1}: Best Score = {gen_score:.4f}, Config = {decode_chromosome(gen_best)}")

        if gen_score > best_score_overall:
            best_score_overall = gen_score
            best_overall = gen_best

        if best_score_overall >= FITNESS_THRESHOLD:
            print(f"\nEarly stopping at Gen {generation+1} with fitness {best_score_overall:.4f}")
            break

        group_size = min(group_size + 1, POPULATION_SIZE)

    print("\nBest Configuration Found:")
    print("Validation Accuracy:", best_score_overall)
    print("Chromosome:", decode_chromosome(best_overall))

run_ga()
