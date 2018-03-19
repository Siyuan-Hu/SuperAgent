import matplotlib.pyplot as plt

index = 0
step = 1
update = []
reward = []

file = open("log.txt", "r")
while True:
	line = file.readline()
	if line:
		if line[0:3] == "('E":
			# print(line[31:-2])
			reward.append(float(line[31:-2]))
			update.append(step*index)
			index += 1
			# print float(line[27:])
	else:
		break

file.close()

# print(reward)

plt.plot(update, reward, c='b', label='Actor Mimic for Acrobot-v1')

plt.legend(loc='best')
plt.ylabel('average reward on 20 episodes')
plt.xlabel('update to Q-network')
plt.grid()
plt.show()
