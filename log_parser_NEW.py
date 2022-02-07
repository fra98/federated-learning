import numpy as np
import matplotlib.pyplot as plt

BASE_PATH = "logs/"

LOGS_TRAIN = ["train_acc_01.txt", "train_acc_02.txt", "train_acc_03.txt",
              "train_acc_04.txt", "train_acc_05.txt", "train_acc_06.txt",
              "train_acc_07.txt", "train_acc_08.txt", "train_acc_09.txt",
              "train_acc_10.txt", "train_acc_11.txt", "train_acc_12.txt",
              "train_acc_13.txt", "train_acc_14.txt", "train_acc_15.txt",
              "train_acc_16.txt", "train_acc_17.txt", "train_acc_18.txt",
              "train_acc_19.txt"]

LOGS_TEST = ["test_acc_01.txt", "test_acc_02.txt", "test_acc_03.txt",
             "test_acc_04.txt", "test_acc_05.txt", "test_acc_06.txt",
             "test_acc_07.txt", "test_acc_08.txt", "test_acc_09.txt",
             "test_acc_10.txt", "test_acc_11.txt", "test_acc_12.txt",
             "test_acc_13.txt", "test_acc_14.txt", "test_acc_15.txt",
             "test_acc_16.txt", "test_acc_17.txt", "test_acc_18.txt",
             "test_acc_19.txt"]


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
    for log in LOGS_TRAIN:
        round, train_acc, train_loss = get_results(BASE_PATH + log)
        vect_train_rounds.append(round)
        vect_train_acc.append(train_acc)
        vect_train_loss.append(train_loss)
        
    for log in LOGS_TEST:
        round, test_acc, test_loss = get_results(BASE_PATH + log)    
        vect_test_rounds.append(round)
        vect_test_acc.append(test_acc)
        vect_test_loss.append(test_loss)


    for i in range(len(vect_test_acc)):
        assert len(vect_train_rounds[i]) == len(vect_train_acc[i])
        assert len(vect_test_rounds[i]) == len(vect_test_acc[i])
        assert len(vect_test_acc[i]) == len(vect_train_acc[i])

    plt.figure()
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    
    # Train
    # plt.plot(vect_train_rounds[0], vect_train_acc[0], label='train 1')
    # plt.plot(vect_train_rounds[1], vect_train_acc[1], label='train 2')
    # plt.plot(vect_train_rounds[2], vect_train_acc[2], label='train 3')
    # plt.plot(vect_train_rounds[3], vect_train_acc[3], label='train 4')
    # plt.plot(vect_train_rounds[4], vect_train_acc[4], label='train 5')
    # plt.plot(vect_train_rounds[5], vect_train_acc[5], label='train 6')
    # plt.plot(vect_train_rounds[6], vect_train_acc[6], label='train 7')
    # plt.plot(vect_train_rounds[7], vect_train_acc[7], label='train 8')
    # plt.plot(vect_train_rounds[8], vect_train_acc[8], label='train 9') 
    # plt.plot(vect_train_rounds[9], vect_train_acc[9], label='train 10') 
    # plt.plot(vect_train_rounds[10], vect_train_acc[10], label='train 11') 
    # plt.plot(vect_train_rounds[11], vect_train_acc[11], label='train 12') 
    # plt.plot(vect_train_rounds[12], vect_train_acc[12], label='train 13') 
    # plt.plot(vect_train_rounds[13], vect_train_acc[13], label='train 14') 
    # plt.plot(vect_train_rounds[14], vect_train_acc[14], label='train 15') 
    # plt.plot(vect_train_rounds[15], vect_train_acc[15], label='train 16') 
    # plt.plot(vect_train_rounds[16], vect_train_acc[16], label='train 17') 
    # plt.plot(vect_train_rounds[17], vect_train_acc[17], label='train 18') 
    # plt.plot(vect_train_rounds[18], vect_train_acc[18], label='train 19') 

    
    # Test
    plt.plot(vect_test_rounds[0], vect_test_acc[0], label='test 1')
    plt.plot(vect_test_rounds[1], vect_test_acc[1], label='test 2')
    plt.plot(vect_test_rounds[2], vect_test_acc[2], label='test 3')
    plt.plot(vect_test_rounds[3], vect_test_acc[3], label='test 4')
    plt.plot(vect_test_rounds[4], vect_test_acc[4], label='test 5')
    plt.plot(vect_test_rounds[5], vect_test_acc[5], label='test 6')
    plt.plot(vect_test_rounds[6], vect_test_acc[6], label='test 7')
    plt.plot(vect_test_rounds[7], vect_test_acc[7], label='test 8')
    plt.plot(vect_test_rounds[8], vect_test_acc[8], label='test 9')
    plt.plot(vect_test_rounds[9], vect_test_acc[9], label='test 10') 
    plt.plot(vect_test_rounds[10], vect_test_acc[10], label='test 11') 
    plt.plot(vect_test_rounds[11], vect_test_acc[11], label='test 12') 
    plt.plot(vect_test_rounds[12], vect_test_acc[12], label='test 13') 
    plt.plot(vect_test_rounds[13], vect_test_acc[13], label='test 14') 
    plt.plot(vect_test_rounds[14], vect_test_acc[14], label='test 15') 
    plt.plot(vect_test_rounds[15], vect_test_acc[15], label='test 16') 
    plt.plot(vect_test_rounds[16], vect_test_acc[16], label='test 17') 
    plt.plot(vect_test_rounds[17], vect_test_acc[17], label='test 18') 
    plt.plot(vect_test_rounds[18], vect_test_acc[18], label='test 19') 
    
    plt.legend(loc='lower right')
    plt.show()



