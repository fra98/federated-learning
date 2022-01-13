from copy import deepcopy
import numpy as np

from src.utils import load_cifar

NUM_CLASSES = 10
NUM_CLIENTS = 100
ALPHA = 1

trainset = load_cifar(name=f'CIFAR{NUM_CLASSES}', train=True)
classes_IDs = [i for i in range(NUM_CLASSES)]
client_data_size = len(trainset) // NUM_CLIENTS
num_samples_per_class = len(trainset) // NUM_CLASSES


'''
def main_dirichlet_WRONG():
    trainset = load_cifar(name='CIFAR10', train=True)
    clients = [i for i in range(NUM_CLIENTS)]
    map = np.zeros((NUM_CLIENTS, NUM_CLASSES), dtype=np.int32)

    priors = np.ones(NUM_CLIENTS) # * (1/NUM_CLIENTS) ?
    probabilities = np.random.dirichlet(ALPHA * priors, NUM_CLASSES)
    # print("Probabilities matrix -> Row: CLASS, Col: USERS")
    # print(probabilities)

    for idx in range(len(trainset)):
        class_label = trainset[idx][1]
        user_id = np.random.choice(clients, p=probabilities[class_label])
        map[user_id][class_label] += 1

    print("Class counter per user -> Row: USERS, Col: CLASS")
    print(map)

    print("Size dataset per user")
    print(map.sum(axis=1))
'''


def dirichlet_probabilities(num_classes, num_clients, alpha):
    priors = (1/num_classes) * np.ones(num_classes)
    probabilities = np.random.dirichlet(alpha * priors, num_clients)
    return probabilities


def print_stats(map):
    print("Class counter per user -> Row: USERS, Col: CLASS")
    print(map)
    print("Size dataset per user")
    print(map.sum(axis=1))
    print("Size dataset per class")
    print(map.sum(axis=0))


def main_dirichlet():
    map = np.zeros((NUM_CLIENTS, NUM_CLASSES), dtype=np.int32)

    probabilities = dirichlet_probabilities(NUM_CLASSES, NUM_CLIENTS, ALPHA)
    print("Probabilities matrix -> Row: USERS, Col: CLASS")
    np.set_printoptions(precision=2, suppress=True)
    print(probabilities)
    
    for user_id in range(NUM_CLIENTS):
        for _ in range(client_data_size):
            class_label = np.random.choice(classes_IDs, p=probabilities[user_id])
            map[user_id][class_label] += 1

    print_stats(map)


def main_cifar_split():
    map = np.zeros((NUM_CLIENTS, NUM_CLASSES), dtype=np.int32)
    
    # Create 2D array of [image_id, class] SORTED by classes 
    indexer = np.empty((len(trainset), 2), dtype=np.int32)  # image_id, class
    for i in range(len(trainset)):
        indexer[i][0] = i
        indexer[i][1] = trainset[i][1]
    indexer = indexer[np.argsort(indexer[:, 1])]

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

            image_id = indexer[pos][0]
            user_indexes.append(image_id)
            offset_class[class_label] += 1

        indexes.append(deepcopy(user_indexes))

    print_stats(map)
    


if __name__ == '__main__':
    # main_dirichlet()
    main_cifar_split()