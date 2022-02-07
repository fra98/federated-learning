import numpy as np
import matplotlib.pyplot as plt

BASE_PATH = "logs/"

LOGS = ["config_01.log", "config_02.log", "config_03.log",
        "config_04.log", "config_05.log", "config_06.log",
        "config_07.log", "config_08.log", "config_09.log",
        "config_10.log", "config_11.log", "config_12.log",
        "config_13.log", "config_14.log", "config_15.log",
        "config_16.log", "config_17.log", "config_18.log"]

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

    vet_rounds = []
    
    STEP = 15
    for i in range(9):
        v = vect_train_acc[i]
        v = v[v > 0]
        v = v[STEP-1::STEP]
        vect_train_acc[i] = v

        s = vect_test_acc[i]
        s = s[s > 0]
        vect_test_acc[i] = s

        rounds = np.arange(1, len(vect_train_acc[i])+1) * STEP
        vet_rounds.append(rounds)

    
    STEP = 5
    for i in range(9, 19):
        v = vect_train_acc[i]
        v = v[v > 0]
        vect_train_acc[i] = v

        s = vect_test_acc[i]
        s = s[s > 0]
        vect_test_acc[i] = s

        rounds = np.arange(1, len(vect_train_acc[i])+1) * STEP
        vet_rounds.append(rounds)


    for i in range(len(vect_test_acc)):
        assert len(vect_test_acc[i]) == len(vect_train_acc[i])


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

    plt.plot(vet_rounds[9], vect_train_acc[9], label='train conf 10')
    plt.plot(vet_rounds[10], vect_train_acc[10], label='train conf 11')
    plt.plot(vet_rounds[11], vect_train_acc[11], label='train conf 12')
    plt.plot(vet_rounds[12], vect_train_acc[12], label='train conf 13')
    plt.plot(vet_rounds[13], vect_train_acc[13], label='train conf 14')
    plt.plot(vet_rounds[14], vect_train_acc[14], label='train conf 15')
    plt.plot(vet_rounds[15], vect_train_acc[15], label='train conf 16')
    plt.plot(vet_rounds[16], vect_train_acc[16], label='train conf 17')
    plt.plot(vet_rounds[17], vect_train_acc[17], label='train conf 18')

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

    plt.plot(vet_rounds[9], vect_test_acc[9], label='test conf 10')
    plt.plot(vet_rounds[10], vect_test_acc[10], label='test conf 11')
    plt.plot(vet_rounds[11], vect_test_acc[11], label='test conf 12')
    plt.plot(vet_rounds[12], vect_test_acc[12], label='test conf 13')
    plt.plot(vet_rounds[13], vect_test_acc[13], label='test conf 14')
    plt.plot(vet_rounds[14], vect_test_acc[14], label='test conf 15')
    plt.plot(vet_rounds[15], vect_test_acc[15], label='test conf 16')
    plt.plot(vet_rounds[16], vect_test_acc[16], label='test conf 17')
    plt.plot(vet_rounds[17], vect_test_acc[17], label='test conf 18')

    plt.legend(loc='lower right')
    plt.show()



