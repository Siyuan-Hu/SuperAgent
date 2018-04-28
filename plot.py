import matplotlib.pyplot as plt

index = 0
step = 100
update = []
reward1 = []
reward2 = []
std = []
sample = 1

file = open("log.txt", "r")
while True:
	line = file.readline()
	if line:
		if line[0] == '-' or (line[0]>='0' and line[0]<='9'):
			if index % 2 == 0:
				data = line.split()
				reward1.append(float(data[0]))
				update.append(step*index/2)				

			if index % 2 == 1:
				data = line.split()
				reward2.append(float(data[0]))
			index += 1
	else:
		break

file.close()


plt.plot(update, reward1, c='b', label='Actor Mimic for InvertedPendulum-v2')
plt.plot(update, reward2, c='r', label='Actor Mimic for InvertedDoublePendulum-v2')

plt.legend(loc='best')
plt.ylabel('average reward on 20 episodes')
plt.xlabel('updates')
# plt.errorbar(update, reward, yerr=std, fmt='o')
plt.grid()
plt.show()