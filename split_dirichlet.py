import numpy as np
import csv
from src.utils import load_cifar

NUM_CLASSES = 10
NUM_CLIENTS = 100
ALPHA = 10


def main_dirichlet():
    trainset = load_cifar(name='CIFAR10', train=True)
    clients = [i for i in range(NUM_CLIENTS)]
    map = np.zeros((NUM_CLIENTS, NUM_CLASSES), dtype=np.int64)

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


def main_google():
    map = np.zeros((100, NUM_CLASSES), dtype=np.int64)

    with open(f"./CIFAR/CIFAR{NUM_CLASSES}_splits/federated_train_alpha_{ALPHA:.2f}.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)    # header
        for line in reader:
            user_id, image_id, class_label = line
            map[int(user_id)][int(class_label)] += 1

    print("Class counter per user -> Row: USERS, Col: CLASS")
    print(map)

    print("Size dataset per user")
    print(map.sum(axis=1))


if __name__ == '__main__':
    main_dirichlet()
    # main_google()