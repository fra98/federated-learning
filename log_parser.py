import numpy as np
import matplotlib.pyplot as plt

BASE_PATH = "logs/"

LOGS = [     
             1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24,
            30, 31, 32, 34,
            40, 41, 42,
            60, 61, 62, 64,
            70, 71, 72, 74,
            82
        ]

vect_train_rounds = []
vect_train_acc = []
vect_train_loss = []

vect_test_rounds = []
vect_test_acc = []
vect_test_loss = []


def get_results(path):
    NULL = -42
    vet_rounds = np.ones(1000) * NULL
    vet_acc = np.ones(1000) * NULL
    vet_loss = np.ones(1000) * NULL

    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            round = int(line.split(" ")[0])
            accuracy = float(line.split(" ")[1])
            loss = float(line.split(" ")[2])
            vet_rounds[round] = round
            vet_acc[round] = accuracy
            vet_loss[round] = loss

    vet_rounds = vet_rounds[vet_rounds != NULL]
    vet_acc = vet_acc[vet_acc != NULL]
    vet_loss = vet_loss[vet_loss != NULL]

    return vet_rounds, vet_acc, vet_loss


if __name__ == '__main__':
    for config in range(100):
        if config in LOGS:
            round, train_acc, train_loss = get_results(BASE_PATH + f"train_acc_{config:02}.txt")
            vect_train_rounds.append(round)
            vect_train_acc.append(train_acc)
            vect_train_loss.append(train_loss)
            
            round, test_acc, test_loss = get_results(BASE_PATH + f"test_acc_{config:02}.txt")    
            vect_test_rounds.append(round)
            vect_test_acc.append(test_acc)
            vect_test_loss.append(test_loss)
        else:
            vect_train_rounds.append([])
            vect_train_acc.append([])
            vect_train_loss.append([])
            vect_test_rounds.append([])
            vect_test_acc.append([])
            vect_test_loss.append([])


    for i in range(100):
        assert len(vect_train_rounds[i]) == len(vect_train_acc[i])
        assert len(vect_test_rounds[i]) == len(vect_test_acc[i])
        assert len(vect_test_acc[i]) == len(vect_train_acc[i])


    # PLOTTING
    
    plt.figure()
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    
    # SHOW = [24, 34, 74]
    # SHOW = [19, 31, 71]
    SHOW = [31, 41, 71]

    for i in SHOW:
        # plt.plot(vect_train_rounds[i], vect_train_acc[i], label=f'train {i}')
        plt.plot(vect_test_rounds[i], vect_test_acc[i], label=f'test {i}')
    
    plt.legend(loc='lower right')
    plt.show()



