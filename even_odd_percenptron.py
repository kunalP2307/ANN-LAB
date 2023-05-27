weights = [0, 0, 0, 0, 0, 1]
bias = 0
threshold = 1


def get_activation(input):
    f_x = 0
    for i in range(len(weights)):
        f_x += input[i] * weights[i]

    f_x += bias
    if f_x >= threshold:
        return 1
    return 0


if __name__ == '__main__':

    input = {0: [1, 1, 0, 0, 0, 0],
             1: [1, 1, 0, 0, 0, 1],
             2: [1, 1, 0, 0, 1, 0],
             3: [1, 1, 0, 0, 1, 1],
             4: [1, 1, 0, 1, 0, 0],
             5: [1, 1, 0, 1, 0, 1],
             6: [1, 1, 0, 1, 1, 0],
             7: [1, 1, 0, 1, 0, 1],
             8: [1, 1, 1, 0, 0, 0],
             9: [1, 1, 1, 0, 0, 1]}

    for dec, _bin in input.items():
        activation = get_activation(_bin)
        if activation == 1:
            print(fr'{dec} -> Odd')
        else:
            print(fr'{dec} -> Even')
