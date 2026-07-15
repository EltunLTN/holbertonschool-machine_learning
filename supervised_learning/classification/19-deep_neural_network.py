import matplotlib.pyplot as plt

data1 = [1, 2, 3, 4, 5]
data2 = [2, 3, 4, 5, 6]

# Multiple datasets, each can have its own color
plt.hist([data1, data2], color=['red', 'blue'])

plt.show()
