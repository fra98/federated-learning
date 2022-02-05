from copy import deepcopy
import random
import numpy as np

def indexes_split_IID(num_clients, num_classes, dataset, clients_sizes):
    # NOTE: this split assumes equal class distribution -> true for cifar
    num_samples_per_class = len(dataset) // num_classes
    classes = [i for i in range(num_classes)]

    # Indexes sorted by class label 
    sorted_indexes = np.argsort(dataset.targets)

    indexes = []
    for _ in range(num_clients):
        indexes.append([])
    
    offsets_class = np.zeros(num_classes, dtype=np.int32)
    shuffled_classes = deepcopy(classes)
    for user_id in range(num_clients):
        client_data_size = clients_sizes[user_id]

        temp_class = 0
        for _ in range(client_data_size):
            class_label = temp_class % num_classes

            # check if selected class is available. If not randomly select another one  
            if offsets_class[class_label] >= num_samples_per_class:
                random.shuffle(shuffled_classes)
                class_label = None
                for c in shuffled_classes:
                    if offsets_class[c] < num_samples_per_class:
                        class_label = c

            if class_label is None:  # entire dataset already used --> pick random
                pos = random.randrange(len(dataset))
            else:
                pos = num_samples_per_class * class_label + offsets_class[class_label]  # base + offset

            image_id = sorted_indexes[pos]
            indexes[user_id].append(image_id)

            offsets_class[class_label] += 1
            temp_class += 1

    return indexes


def indexes_split_NON_IID(num_clients, num_classes, alpha, dataset, clients_sizes):  # class-balanced
    # NOTE: this split assumes equal class distribution -> true for cifar
    num_samples_per_class = len(dataset) // num_classes
    classes = [i for i in range(num_classes)]

    # Indexes sorted by class label 
    sorted_indexes = np.argsort(dataset.targets)

    # Priors probabilities of each class
    priors = (1/num_classes) * np.ones(num_classes)

    # Dirichlet's distribution probabilities -> 2D matrix, shape: (NUM_CLIENTS, NUM_CLASSES)
    dirichlet_probs = np.random.dirichlet(alpha * priors, num_clients)

    indexes = []
    for _ in range(num_clients):
        indexes.append([])

    offsets_class = np.zeros(num_classes, dtype=np.int32)
    shuffled_classes = deepcopy(classes)
    for user_id in range(num_clients):   
        client_data_size = clients_sizes[user_id]

        for _ in range(client_data_size):     
            class_label = np.random.choice(classes, p=dirichlet_probs[user_id])

            # check if selected class is available. If not randomly select another one  
            if offsets_class[class_label] >= num_samples_per_class:
                random.shuffle(shuffled_classes)
                class_label = None
                for c in shuffled_classes:
                    if offsets_class[c] < num_samples_per_class:
                        class_label = c

            if class_label is None:  # entire dataset already used --> pick random
                pos = random.randrange(len(dataset))
            else:
                pos = num_samples_per_class * class_label + offsets_class[class_label]  # base + offset
                
            image_id = sorted_indexes[pos]
            indexes[user_id].append(image_id)

            offsets_class[class_label] += 1

    return indexes


def print_stats(targets, indexes, num_clients, num_classes, alpha=-1):
    map = np.zeros((num_clients, num_classes), dtype=np.int32)
    
    for i in range(len(indexes)):
        for j in range(len(indexes[i])):
            class_label = targets[indexes[i][j]]
            map[i][class_label] += 1

    np.set_printoptions(precision=2, suppress=True)
    print(f"ALPHA = {alpha}, NUM_CLIENTS = {num_clients}")
    print("Class counter per user -> Row: USERS, Col: CLASS")
    print(map)
    print("Size dataset per user")
    print(map.sum(axis=1))
    print("Size dataset per class")
    print(map.sum(axis=0))
    print("Total size")
    print(map.sum())
    print("Percentage of classes for each user")
    perc_classes_users = 100 * map / map.sum(axis=1).reshape(num_clients, 1)
    print(perc_classes_users)
    print("Class percentage std dev for each client")
    print(np.std(perc_classes_users, axis=1))
    print("Class percentage total std dev")
    print(round(np.std(np.std(perc_classes_users, axis=1) / np.sum(perc_classes_users, axis=1)), 3))
