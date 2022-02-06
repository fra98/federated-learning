import numpy as np
import matplotlib.pyplot as plt

BASE_PATH = "logs/"

LOGS = ["config_01.log", "config_02.log", "config_03.log",
        "config_04.log", "config_05.log", " config_06.log",
        "config_07.log", "config_08.log", " config_09.log"]

def get_results(path):
    train_acc = np.zeros(500)
    train_loss = np.zeros(500)
    test_acc = np.zeros(500)
    test_loss = np.zeros(500)
    curr_round = None

    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "ROUND" in line:
                curr_round = int(line.split(" ")[4])

            elif "Server -> Train" in line:
                strings = line.split(" ")
                loss = float(strings[7])
                accuracy = float(strings[11])
                train_loss[curr_round-1] = loss
                train_acc[curr_round-1] = accuracy

            elif "Server -> Test" in line:
                strings = line.split(" ")
                loss = float(strings[7])
                accuracy = float(strings[11])
                test_loss[curr_round-1] = loss
                test_acc[curr_round-1] = accuracy

    return train_acc, train_loss, test_acc, test_loss


def plot(arr, color='r'):
    plt.figure()
    plt.plot(arr, color)
    plt.show()


if __name__ == '__main__':
    vect_train_acc = []
    vect_train_loss = []
    vect_test_acc = []
    vect_test_loss = []

    for log in LOGS:
        train_acc, train_loss, test_acc, test_loss = get_results(BASE_PATH + log)
        vect_train_acc.append(train_acc)
        vect_train_loss.append(train_loss)
        vect_test_acc.append(test_acc)
        vect_test_loss.append(test_loss)

    for i in range(len(vect_train_acc)):
        v = vect_train_acc[i]
        v = v[v > 0]
        STEP = 15
        v = v[STEP-1::STEP]
        vect_train_acc[i] = v

    for i in range(len(vect_test_acc)):
        v = vect_test_acc[i]
        v = v[v > 0]
        vect_test_acc[i] = v

    assert len(vect_test_acc[0]) == len(vect_train_acc[0])
    rounds = np.arange(1, len(vect_train_acc[0])+1)
    rounds = rounds * STEP


    plt.figure()
    plt.legend()
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.plot(rounds, vect_train_acc[0], 'r', label="train")
    plt.plot(rounds, vect_test_acc[0], 'b', label='test')
    plt.legend(loc='lower right')
    plt.show()



