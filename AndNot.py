import numpy as np

input_values = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

print("Input Table \n",input_values)

weights = [1, -1]

dot_product = input_values @ weights

threshold = (2*1) - 1
print("Dot Product ", dot_product)
print("x1\tx2\tY")

for i in range(len(dot_product)):
    if dot_product[i] >= 1:
        print(input_values[i][0], "\t", input_values[i][1], "\t", "1")
    else:
        print(input_values[i][0], "\t", input_values[i][1], "\t", "0")
