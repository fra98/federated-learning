import sys
import csv
import numpy as np

NUM_CLASSES = 10
ALPHA = 1

def print_stats(map):
    np.set_printoptions(precision=2, suppress=True)
    print(f"ALPHA = {ALPHA}, NUM_CLIENTS = 100")
    print("Class counter per user -> Row: USERS, Col: CLASS")
    print(map)
    print("Size dataset per user")
    print(map.sum(axis=1))
    print("Size dataset per class")
    print(map.sum(axis=0))
    print("Total size")
    print(map.sum())
    print("Percentage of classes for each user")
    print(100 * map / map.sum(axis=1).reshape(100, 1))


def main():
    map = np.zeros((100, NUM_CLASSES), dtype=np.int64)

    with open(f"./CIFAR/CIFAR{NUM_CLASSES}_splits/federated_train_alpha_{ALPHA:.2f}.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)    # header
        for line in reader:
            user_id, image_id, class_label = line
            map[int(user_id)][int(class_label)] += 1

    print_stats(map)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        ALPHA = float(sys.argv[1])

    main()

