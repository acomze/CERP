import numpy as np
import math
# from sklearn import preprocessing
from collections import defaultdict

# the number of device by D2D
M = 12

# the number of edge by cellular/wifi

E = 4

# the number of cloud
C = 1

app_k = 3
connection_k = 3

Total_node = 16


unit_cost = 1

MAXSTEP = M+E



class Environment():
    def __init__(self,):
        read_com = open('simuData/edge_capacity.txt', 'r')
        read_D2Dtrans = open('simuData/edgerate.txt', 'r')
        read_uplink = open('simuData/uplinkrate.txt', 'r')
        read_backhaul = open('simuData/backhaul.txt', 'r')
        read_probing_1 = open('simuData/probe_cost1.txt', 'r')
        read_probing_2 = open('simuData/probe_cost2.txt', 'r')
        read_probing_3 = open('simuData/probe_cost3.txt', 'r')
        read_probing_4 = open('simuData/probe_cost4.txt', 'r')
        read_node = open('simuData/count.txt', 'r')
        avai = open('simuData/avai.txt', 'r')
        read_lamda1 = open('simuData/lamda1.txt', 'r')
        read_lamda2 = open('simuData/lamda2.txt', 'r')

        self.probe1 = read_probing_1.readlines()
        self.probe2 = read_probing_2.readlines()
        self.probe3 = read_probing_3.readlines()
        self.probe4 = read_probing_4.readlines()
        self.lamda1 = read_lamda1.readlines()
        self.lamda2 = read_lamda2.readlines()


        self.Dtrans_rate = read_D2Dtrans.readlines()
        self.Dsigpower = 0.2
        self.uplink_rate = read_uplink.readlines()
        self.bakchaul = read_backhaul.readlines()
        self.Csigpower = 0.6
        self.CPUpower = 0.9
        self.computing_capacity = read_com.readlines()
        # available type
        # self.n_actions = app_k*connection_k
        self.n_actions = 4
        # dimension: device type * network * probed/unprobed
        # self.features_space = np.ones(app_k*connection_k + 2)
        self.features_space = np.ones(4 + 2)
        self.n_features = len(self.features_space)
        self.type_record = defaultdict(list)
        # RRT 100ms
        self.h_RRT = 0.1
        self.count = read_node.readlines()
        self.avaible = avai.readlines()



    def step(self, node, action, task_size, input_data, u, episode, min_cost):
        #D2D offloading
        if node < M:
            # 1MB
            computing_delay = task_size / float(self.computing_capacity[episode*(Total_node) + node])
            transmission_delay = input_data / (float(self.Dtrans_rate[episode*(Total_node) + node])*8)  # + output_data / self.Dtrans_rate
            delay = computing_delay + transmission_delay
            power_com = transmission_delay * self.Dsigpower
            # (receive_power[node] + send_power) * transmission_delay
            cost = self.features_space[4] * delay + self.features_space[5] * power_com
            # print(self.distance)
            if float(self.Dtrans_rate[episode*(Total_node) + node]) > 7.5:
                probing_cost = unit_cost * float(self.probe1[episode*(Total_node) + node]) * input_data / 2.475
            else:
                probing_cost = unit_cost * float(self.probe2[episode*(Total_node) + node]) * input_data / 2.475

            self.features_space[action] -= 1
            # self.features_space[action + app_k* connection_k] += 1

            # print("comp", computing_delay)
            # print('trans', transmission_delay)
            # print("time", delay)
            # print("power", power_com)
            # print("probe", probing_cost)


            current_cost = u + min(0, cost-min_cost) + probing_cost

            gain = min(0, cost-min_cost) + probing_cost
            communication_type = 0

            min_cost = min(min_cost, cost)
            done = False
            # print("best cost", current_cost)
            # print("mini", min_cost)
            # print("===================================================")


        elif node < M + E:
            computing_delay = task_size / float(self.computing_capacity[episode * (Total_node) + node])
            transmission_delay = input_data / (float(self.Dtrans_rate[episode*(Total_node) + node]) * 8)  # + output_data / self.Dtrans_rate
            delay = computing_delay + transmission_delay
            power_com = transmission_delay * self.Csigpower
            # (receive_power[node] + send_power) * transmission_delay
            cost = self.features_space[4] * delay + self.features_space[5] * power_com
            # print(self.distance)
            if float(self.Dtrans_rate[episode*(Total_node) + node]) > 1:
                probing_cost = unit_cost * float(self.probe3[episode*(Total_node) + node]) * input_data / 2.475
            else:
                probing_cost = unit_cost * float(self.probe4[episode*(Total_node) + node]) * input_data / 2.475
            self.features_space[action] -= 1
            # self.features_space[action + app_k * connection_k] += 1
            communication_type = 1

            # print("comp",computing_delay)
            # print('trans', transmission_delay)
            # print("time", delay)
            # print("power", power_com)
            # print("probe", probing_cost)


            current_cost = u + min(0, cost - min_cost) + probing_cost

            gain = min(0, cost - min_cost) + probing_cost

            min_cost = min(min_cost, cost)
            done = False

        #     done = False
        else:
            done = True
        s_ = self.features_space

        return s_, -gain, min_cost, current_cost, done, float(self.computing_capacity[episode*(Total_node) + node]), float(self.Dtrans_rate[episode*(Total_node) + node]), communication_type, probing_cost, cost

    def render(self):
        # time.sleep(0.01)
        self.update()

    def reset(self, episode):
        self.features_space = np.zeros(4 + 2)
        self.type_record = defaultdict(list)
        count = int(self.count[episode])
        record_probed_action = np.ones(Total_node)

        num = np.array([int(x) for x in self.avaible[episode].split(',')])
        for i in range(len(num)):
            # category
            cop = float(self.computing_capacity[episode * (Total_node) + num[i]])
            com = float(self.Dtrans_rate[episode*(Total_node) + num[i]])
            record_probed_action[num[i]] = 0

            if num[i] < 6:
                self.features_space[0] += 1
                self.type_record[0].append(num[i])
            elif num[i] < 12:
                self.features_space[1] += 1
                self.type_record[1].append(num[i])
            elif num[i] < 14:
                self.features_space[2] += 1
                self.type_record[2].append(num[i])
            else:
                self.features_space[3] += 1
                self.type_record[3].append(num[i])

            # time classification
            # if cop < 3 and com < 2:
            #     self.features_space[0] += 1
            #     self.type_record[0].append(num[i])
            # elif cop < 5 and com < 2:
            #     self.features_space[1] += 1
            #     self.type_record[1].append(num[i])
            # elif cop < 10 and com < 2:
            #     self.features_space[2] += 1
            #     self.type_record[2].append(num[i])
            # elif cop < 3 and com < 10:
            #     self.features_space[3] += 1
            #     self.type_record[3].append(num[i])
            # elif cop < 5 and com < 10:
            #     self.features_space[4] += 1
            #     self.type_record[4].append(num[i])
            # elif cop < 10 and com < 10:
            #     self.features_space[5] += 1
            #     self.type_record[5].append(num[i])
            # elif cop < 3 and com < 15:
            #     self.features_space[6] += 1
            #     self.type_record[6].append(num[i])
            # elif cop < 5 and com < 15:
            #     self.features_space[7] += 1
            #     self.type_record[7].append(num[i])
            # else:
            #     self.features_space[8] += 1
            #     self.type_record[8].append(num[i])



        lambda1 = float(self.lamda1[episode])
        lambda2 = float(self.lamda2[episode])
        self.features_space[4] = lambda1
        self.features_space[5] = lambda2

        return self.features_space, self.type_record, count, record_probed_action

