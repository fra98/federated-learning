from copy import deepcopy
import numpy as np
import sys
import random

from src.utils import load_cifar

NUM_CLASSES = 10
NUM_CLIENTS = 100
ALPHA = 1
CB = True

trainset = load_cifar(name=f'CIFAR{NUM_CLASSES}', train=True)
classes_IDs = [i for i in range(NUM_CLASSES)]
client_data_size = len(trainset) // NUM_CLIENTS
num_samples_per_class = len(trainset) // NUM_CLASSES

# Indexes sorted by class label 
sorted_indexes = np.argsort(trainset.targets)


def print_stats(map):
    np.set_printoptions(precision=2, suppress=True)
    print(f"ALPHA = {ALPHA}, NUM_CLIENTS = {NUM_CLIENTS}")
    print("Class counter per user -> Row: USERS, Col: CLASS")
    print(map)
    print("Size dataset per user")
    print(map.sum(axis=1))
    print("Size dataset per class")
    print(map.sum(axis=0))
    print("Total size")
    print(map.sum())
    print("Percentage of classes for each user")
    perc_classes_users = 100 * map / map.sum(axis=1).reshape(NUM_CLIENTS, 1)
    print(perc_classes_users)
    print("Class percentage std dev for each client")
    print(np.std(perc_classes_users, axis=1))
    print("Class percentage total std dev")
    print(round(np.std(np.std(perc_classes_users, axis=1) / np.sum(perc_classes_users, axis=1)), 3))


def dirichlet_probabilities(num_classes, num_clients, alpha):
    priors = (1/num_classes) * np.ones(num_classes)
    probabilities = np.random.dirichlet(alpha * priors, num_clients)
    return probabilities


def main_cifar_split():
    map = np.zeros((NUM_CLIENTS, NUM_CLASSES), dtype=np.int32)

    # Dirichlet probability distribution -> 2D matrix, shape: (NUM_CLIENTS, NUM_CLASSES)
    dirichlet_probs = dirichlet_probabilities(NUM_CLASSES, NUM_CLIENTS, ALPHA)
    
    offset_class = np.zeros(NUM_CLASSES, dtype=np.int32)
    indexes = []
    for user_id in range(NUM_CLIENTS):
        user_indexes = []
        for _ in range(client_data_size):
            class_label = np.random.choice(classes_IDs, p=dirichlet_probs[user_id])
            map[user_id][class_label] += 1

            if offset_class[class_label] >= num_samples_per_class:
                offset_class[class_label] = 0
            pos = num_samples_per_class * class_label + offset_class[class_label]  # base + offset

            image_id = sorted_indexes[pos]
            user_indexes.append(image_id)
            offset_class[class_label] += 1

        indexes.append(deepcopy(user_indexes))

    print_stats(map)
    

def main_cifar_split_class_balanced():
    map = np.zeros((NUM_CLIENTS, NUM_CLASSES), dtype=np.int32)

    # Dirichlet probability distribution -> 2D matrix, shape: (NUM_CLIENTS, NUM_CLASSES)
    dirichlet_probs = dirichlet_probabilities(NUM_CLASSES, NUM_CLIENTS, ALPHA)

    offset_class = np.zeros(NUM_CLASSES, dtype=np.int32)
    users_indexes = []
    for _ in range(NUM_CLIENTS):
        users_indexes.append([])

    shuffled_classes = deepcopy(classes_IDs)
    for user_id in range(NUM_CLIENTS):      
        for _ in range(client_data_size):     
            class_label = np.random.choice(classes_IDs, p=dirichlet_probs[user_id])

            if offset_class[class_label] >= num_samples_per_class:
                random.shuffle(shuffled_classes)
                for c in shuffled_classes:
                    if offset_class[c] < num_samples_per_class:
                        class_label = c

            pos = num_samples_per_class * class_label + offset_class[class_label]  # base + offset
            image_id = sorted_indexes[pos]
            users_indexes[user_id].append(image_id)

            offset_class[class_label] += 1
            map[user_id][class_label] += 1

    print_stats(map)


def main_cifar_split_class_unbalanced():
    map = np.zeros((NUM_CLIENTS, NUM_CLASSES), dtype=np.int32)

    std_client_samples = 0.2

    avg_train_size = len(trainset) / NUM_CLIENTS
    clients_sizes = np.random.normal(avg_train_size, avg_train_size * std_client_samples, NUM_CLIENTS)
    delta = len(trainset) - np.sum(clients_sizes) # distribute difference over all clients
    clients_sizes = (clients_sizes + delta/len(clients_sizes)).astype(int)     

    # Dirichlet probability distribution -> 2D matrix, shape: (NUM_CLIENTS, NUM_CLASSES)
    dirichlet_probs = dirichlet_probabilities(NUM_CLASSES, NUM_CLIENTS, ALPHA)

    offset_class = np.zeros(NUM_CLASSES, dtype=np.int32)
    users_indexes = []
    for _ in range(NUM_CLIENTS):
        users_indexes.append([])

    shuffled_classes = deepcopy(classes_IDs)
    for user_id in range(NUM_CLIENTS):
        client_data_size = clients_sizes[user_id]

        for _ in range(client_data_size):     
            class_label = np.random.choice(classes_IDs, p=dirichlet_probs[user_id])

            if offset_class[class_label] >= num_samples_per_class:
                random.shuffle(shuffled_classes)
                for c in shuffled_classes:
                    if offset_class[c] < num_samples_per_class:
                        class_label = c

            pos = num_samples_per_class * class_label + offset_class[class_label]  # base + offset
            image_id = sorted_indexes[pos]
            users_indexes[user_id].append(image_id)

            offset_class[class_label] += 1
            map[user_id][class_label] += 1

    print_stats(map)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        ALPHA = float(sys.argv[1])
        V = int((sys.argv[2]))

    if V == 1:
        main_cifar_split_class_balanced()
    elif V == 2:
        main_cifar_split_class_unbalanced()
    else :
        main_cifar_split()