# Generate ANDNOT function using McCulloch-Pitts neural net by a python program

weights = [1, -1]
threshold = 1


def get_activation(input):
    weighted_sum = (input[0] * weights[0]) + (input[1] * weights[1])
    if weighted_sum >= threshold:
        return 1
    return 0


intput = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]

for lst in intput:
    print(fr'x1 : {lst[0]} x2 : {lst[1]} Y -> {get_activation(lst)}')
