from DQN_HRL import DeepQNetwork
import tensorflow as tf
import numpy as np
import copy
import operator
import random
from client import Client
import  sys
from collections import defaultdict

MEMORY_CAPACITY = 1000
BATCH_SIZE = 32
s_dim = 7
Dsigpower = 0.2
Csigpower = 0.6

def make_layer(inputs, in_size, out_size, activate=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    basis = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    result = tf.matmul(inputs, weights) + basis
    if activate is None:
        return result
    else:
        return activate(result)


def estimate_value(service_demand, input_data, device_state, current_cost, observation):
    computing_delay = service_demand / device_state[0]
    trans = input_data / (device_state[1]*8)
    if device_state[2] == 0:
        # D2D trans
        power_com = trans * Dsigpower
    else:
        power_com = trans * Csigpower

    total = observation[4]*(computing_delay + trans) + observation[5]*power_com

    gain = min(0, total - current_cost) + device_state[3]
    return gain


def pick_device(service_demand, input_data, available_device, record_probed_action, device_state, current_cost, observation):
    count = {}
    ip = {}
    for i in range(len(available_device)):
        device = available_device[i][0]
        ip[device] = available_device[i][1]
        if record_probed_action[device] != 1:
            count[device] = estimate_value(service_demand, input_data, device_state[device], current_cost, observation)

    final = sorted(count.items(), key=operator.itemgetter(1))
    device = final[0][0]
    return device, ip[device], final[0][1]


def device_state_update(device, device_state, comp, comm, connec, probing_cost):
    if device in device_state.keys():
        ncomp = (device_state[device][0] * device_state[device][4] + comp)/(device_state[device][4] + 1)
        ncomm = (device_state[device][1] * device_state[device][4] + comm) / (device_state[device][4] + 1)
        nprobing_cost = (device_state[device][3] * device_state[device][4] + probing_cost) / (device_state[device][4] + 1)
        ntimes = device_state[device][4] + 1
        device_state[device] = [ncomp, ncomm, connec, nprobing_cost, ntimes]
    else:
        device_state[device] = [comp, comm, connec, probing_cost, 1]
    return device_state


def device_info():
    service_amount = 1
    service_size = 1
    capacity = 1
    power = 1
    lamda_1 = 1
    lamda_2 = 1
    return service_amount, service_size, capacity, power, lamda_1, lamda_2

def catch_list():
    # ip_list = dict({"192.168.1.128": "r1",
    #                 "192.168.26.66": "s1",
    #                 "192.168.1.101": "j1",
    #                 "192.168.1.199": "d1",
    #                 "127.0.0.1": "l1"})
    ip_list = dict({"192.168.1.101": "j1",
                    })
    # ip_list = [
    #     "192.168.1.128",  # Raspberry
    #     "192.168.26.66",  # Server
    #     "192.168.1.101",  # Jetson
    #     "192.168.1.199",  # Desktop
    #     "127.0.0.1",  # Local
    # ]
    return ip_list

def classification(available_edge_list, lamda_1, lamda_2):
    level_num = 2
    observation = np.zeros(level_num + 2)
    type_device = defaultdict(list)
    record_probed_action = defaultdict(list)
    # categorize the device
    for key in available_edge_list:
        type_device[available_edge_list[key]] = 0
        if available_edge_list[key] == 'r1':
            type_device[0].append([available_edge_list[key], key])
            observation[0] += 1
        else:
            type_device[1].append([available_edge_list[key], key])
            observation[1] += 1
    observation[level_num] = lamda_1
    observation[level_num + 1] = lamda_2

    return observation, type_device, record_probed_action



def receive_prob_info_change(observation, service_amount, service_size, comp, connec, comm, probing_cost, action, u, min_cost, best_device_ip, device_ip):
    computing_delay = service_amount / comp
    transmission_delay = service_size / comm  # + output_data / self.Dtrans_rate
    delay = computing_delay + transmission_delay
    power_com = transmission_delay * Dsigpower
    cost =  observation[2] * delay + observation[3] * power_com
    probe_cost = probing_cost

    observation[action] -= 1
    # self.features_space[action + app_k* connection_k] += 1
    # print("comp", computing_delay)
    # print('trans', transmission_delay)
    # print("time", delay)
    # print("power", power_com)
    # print("probe", probing_cost)

    current_cost = u + min(0, cost - min_cost) + probe_cost

    gain = min(0, cost - min_cost) + probe_cost

    min_cost = min(min_cost, cost)
    if min_cost == cost:
        best_device_ip = device_ip
    # print("best cost", current_cost)
    # print("mini", min_cost)
    # print("===================================================")
    return observation, -gain, best_device_ip, current_cost, min_cost, cost

class BPNeuralNetwork:
    def __init__(self, session):
        self.session = session
        self.loss = None
        self.optimizer = None
        self.input_n = 0
        self.hidden_n = 0
        self.hidden_size = []
        self.output_n = 0
        self.input_layer = None
        self.hidden_layers = []
        self.output_layer = None
        self.label_layer = None
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.s_dim = s_dim


    def setup(self, layers, learn_rate=0.02):
        # set size args
        if len(layers) < 3:
            return
        self.input_n = layers[0]
        self.hidden_n = len(layers) - 2  # count of hidden layers
        self.hidden_size = layers[1:-1]  # count of cells in each hidden layer
        self.output_n = layers[-1]

        # build network
        self.input_layer = tf.placeholder(tf.float32, [None, self.input_n])
        self.label_layer = tf.placeholder(tf.float32, [None, self.output_n])
        # build hidden layers
        in_size = self.input_n
        out_size = self.hidden_size[0]
        self.hidden_layers.append(make_layer(self.input_layer, in_size, out_size, activate=tf.nn.relu))
        for i in range(self.hidden_n-1):
            in_size = out_size
            out_size = self.hidden_size[i+1]
            inputs = self.hidden_layers[-1]
            self.hidden_layers.append(make_layer(inputs, in_size, out_size, activate=tf.nn.relu))
        # build output layer


        self.output_layer = make_layer(self.hidden_layers[-1], self.hidden_size[-1], self.output_n)
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square((self.label_layer - self.output_layer)), reduction_indices=[1]))

        self.optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(self.loss)
        self.session.run(tf.global_variables_initializer())

    def train(self, iter = 100):
        for _ in range (iter):
            indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
            bt = self.memory[indices, :]
            bs = bt[:, :self.s_dim]
            br = bt[:, self.s_dim: self.s_dim + 1]
            self.session.run(self.optimizer, feed_dict={self.input_layer: bs, self.label_layer: br})


    def predict(self, case):
        return self.session.run(self.output_layer, feed_dict={self.input_layer: case})

    def store_transition(self, s, r):
        transition = np.hstack((s, [r]))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1


    def action(self, state):

        return self.predict(np.array([state]))





def main():
    client = Client()
    # the table of device state, i.e., id, time-average computing capacity, transmission rate, communication type, probing cost, probed times, (recent probed time)
    device_state = {}
    s_dim = 5
    var = 1
    # threshold estimation
    with tf.Session() as session:
        bp = BPNeuralNetwork(session)
        bp.setup([s_dim, 4, 2, 1])
        # action sequencing
        RL = DeepQNetwork(2, 6,
                       learning_rate=0.01,
                       reward_decay=0.8,
                       e_greedy=0.95,
                       replace_target_iter=100,
                       memory_size=500,
                       batch_size=32
                       )

        step = 0
        step_ep = 0
        # f1 = open('resultData/HRL_result.txt', 'w')
        # f2 = open('resultData/HRL_stage.txt', 'w')


        for episode in range(10000):

            service_amount, service_size, capacity, power, lamda_1, lamda_2 = device_info()

            current_cost = lamda_1 * service_amount/capacity + lamda_2 * power * service_amount

            available_edge_list = catch_list()

            device_number = len(available_edge_list)
            min_cost = current_cost
            observation, type_device, record_probed_action = classification(available_edge_list, lamda_1, lamda_2)
            best_device_ip = "local"
            done = False
            reserve_threshold_estimator = {}
            reserve_optimal_cost = {}
            count = 0
            count_action = 0
            prob = np.zeros(device_number + 1)
            total_cost = np.ones(device_number + 1)
            total_cost = list(total_cost * float('inf'))
            total_cost[0] = current_cost
            print("====================================")
            print("episode:", episode)
            stage = 0
            while done is not True:
                # threshold estimator set the goal
                ob_thre = np.append(observation, stage)
                if stage == 0:
                    a = bp.action(ob_thre)
                    a = np.random.normal(a, var)

                reserve_threshold_estimator[count] = [ob_thre]

                count += 1
                print("threshold", a)
                if abs(min_cost) <= a[0][0] or stage == device_number:
                    done = True
                else:
                    observation_probing = np.append(observation, min_cost)
                    observation_probing = np.append(observation_probing, a)
                    action = RL.choose_action(observation_probing)
                    print("probing type", action)
                    # print("probed action",record_probed_action)

                    # pick up a device from the determined group
                    explore = False
                    for i in range(len(type_device[action])):
                        if type_device[action][i][0] not in device_state.keys():
                            device = type_device[action][i][0]
                            device_ip = type_device[action][i][1]
                            explore = True
                    if not explore:
                        if np.random.uniform() < 0.9:
                            device, device_ip, esti_gain = pick_device(service_amount, service_size, type_device[action],
                                                            record_probed_action, device_state, min_cost, observation)
                        else:
                            tuple = random.sample(type_device[action], 1)[0]
                            device = tuple[0]
                            device_ip = tuple[1]
                            esti_gain = estimate_value(service_amount, service_size, device_state[device], current_cost, observation)

                    print("device", device)
                    record_probed_action[device] = 1
                    # probing device
                    comp, connec, comm, probing_cost = client.probing(device_ip)
                    print("===================")
                    print("info", comp, connec, comm, probing_cost)
                    observation_, reward, best_device_ip, current_cost, min_cost, sum_cost = receive_prob_info_change(observation, service_amount, service_size, comp, connec, comm,
                                                                                                                      probing_cost, action, current_cost, min_cost, best_device_ip, device_ip)
                    device_state = device_state_update(device, device_state, comp, comm, connec, probing_cost)
                    stage += 1
                    prob[stage] = probing_cost + prob[stage - 1]
                    total_cost[stage] = sum_cost
                    if stage > 0:
                        ob_thre = np.append(observation, stage)
                        a = bp.action(ob_thre)

                        # Add exploration noise
                        a = np.random.normal(a, var)

                    observation_probing_ = np.append(observation_, min_cost)
                    observation_probing_ = np.append(observation_probing_, a)
                    # store the probing network
                    if not explore:
                        reward = reward - abs((reward + esti_gain)) + max(0, a - min_cost)
                    #     reward = reward - abs((reward + esti_gain))
                    RL.store_transition(observation_probing, action, reward, observation_probing_)
                    reserve_optimal_cost[count_action] = current_cost

                    count_action += 1
                    # action determination update
                    if (step > 100) and (step % 10 == 0):
                        RL.learn()
                    observation = copy.deepcopy(observation_)

                    step += 1
                if done:

                    print("cost", current_cost)
                    print("stage", stage)
                    if best_device_ip == "local":
                        print("local execution")
                    else:
                        client.transfer(best_device_ip)
                    cost_value_up = sorted(reserve_optimal_cost.items(), key=operator.itemgetter(1))
                    # print("threhold update", reserve_threshold_estimator, "and its len", len(reserve_threshold_estimator) - 1)
                    # print("arrange",cost_value_up)
                    # print(reserve_optimal_cost)
                    # print(reserve_threshold_estimator)
                    # print('---------------------------------------------------')
                    if len(cost_value_up) > 0:
                        bp.store_transition(reserve_threshold_estimator[0][0], min_cost)
                        # print("============================================")
                        # print("reward", min_cost)
                        # print("state", reserve_threshold_estimator[0][0])
                        for i in range(len(reserve_threshold_estimator) - 1):
                                # re = min(cost_value_up[i][1], reserve_optimal_cost[i])
                                # re = reserve_optimal_cost[i]
                                total_cost.pop(0)
                                re = min(total_cost) + prob[i + 1]
                                bp.store_transition(reserve_threshold_estimator[i+1][0], re)
                                # if stage > 3:
                                #     print("reward", re)
                                #     print("state", reserve_threshold_estimator[i+1][0])


            step_ep += 1
            # threshold estimator update after a whole episode
            if (step_ep > 100) and (step_ep % 15 == 0):
                var *= 0.90  # decay the action randomness
                bp.train()

if __name__ == '__main__':
    main()
