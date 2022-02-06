import numpy as np
import matplotlib.pyplot as plt

BASE_PATH = "logs/"

LOGS = ["config_01.log", "config_02.log", "config_03.log",
        "config_04.log", "config_05.log", "config_06.log",
        "config_07.log", "config_08.log", "config_09.log"]

STEP = 15

vect_train_acc = []
vect_train_loss = []
vect_test_acc = []
vect_test_loss = []

def get_results(path):
    train_acc = np.zeros(1000)
    train_loss = np.zeros(1000)
    test_acc = np.zeros(1000)
    test_loss = np.zeros(1000)
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


if __name__ == '__main__':
    for log in LOGS:
        train_acc, train_loss, test_acc, test_loss = get_results(BASE_PATH + log)
        vect_train_acc.append(train_acc)
        vect_train_loss.append(train_loss)
        vect_test_acc.append(test_acc)
        vect_test_loss.append(test_loss)

    for i in range(len(vect_train_acc)):
        v = vect_train_acc[i]
        v = v[v > 0]
        v = v[STEP-1::STEP]
        vect_train_acc[i] = v

    for i in range(len(vect_test_acc)):
        v = vect_test_acc[i]
        v = v[v > 0]
        vect_test_acc[i] = v

    assert len(vect_test_acc[0]) == len(vect_train_acc[0])
    assert len(vect_test_acc[1]) == len(vect_train_acc[1])
    assert len(vect_test_acc[2]) == len(vect_train_acc[2])
    assert len(vect_test_acc[3]) == len(vect_train_acc[3])
    assert len(vect_test_acc[4]) == len(vect_train_acc[4])
    assert len(vect_test_acc[5]) == len(vect_train_acc[5])
    assert len(vect_test_acc[6]) == len(vect_train_acc[6])
    assert len(vect_test_acc[7]) == len(vect_train_acc[7])
    assert len(vect_test_acc[8]) == len(vect_train_acc[8])

    vet_rounds = []
    for i in range(len(vect_train_acc)):
        rounds = np.arange(1, len(vect_train_acc[i])+1) * STEP
        vet_rounds.append(rounds)

    plt.figure()
    plt.legend()
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    
    # Train
    plt.plot(vet_rounds[0], vect_train_acc[0], label='train conf 1')
    plt.plot(vet_rounds[1], vect_train_acc[1], label='train conf 2')
    plt.plot(vet_rounds[2], vect_train_acc[2], label='train conf 3')
    plt.plot(vet_rounds[3], vect_train_acc[3], label='train conf 4')
    plt.plot(vet_rounds[4], vect_train_acc[4], label='train conf 5')
    plt.plot(vet_rounds[5], vect_train_acc[5], label='train conf 6')
    plt.plot(vet_rounds[6], vect_train_acc[6], label='train conf 7')
    plt.plot(vet_rounds[7], vect_train_acc[7], label='train conf 8')
    plt.plot(vet_rounds[8], vect_train_acc[8], label='train conf 9')

    # Test
    plt.plot(vet_rounds[0], vect_test_acc[0], label='test conf 1')
    plt.plot(vet_rounds[1], vect_test_acc[1], label='test conf 2')
    plt.plot(vet_rounds[2], vect_test_acc[2], label='test conf 3')
    plt.plot(vet_rounds[3], vect_test_acc[3], label='test conf 4')
    plt.plot(vet_rounds[4], vect_test_acc[4], label='test conf 5')
    plt.plot(vet_rounds[5], vect_test_acc[5], label='test conf 6')
    plt.plot(vet_rounds[6], vect_test_acc[6], label='test conf 7')
    plt.plot(vet_rounds[7], vect_test_acc[7], label='test conf 8')
    plt.plot(vet_rounds[8], vect_test_acc[8], label='test conf 9')

    plt.legend(loc='lower right')
    plt.show()



