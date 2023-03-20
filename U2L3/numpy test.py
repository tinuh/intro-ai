import numpy as np

# create a one dimensional array
one_dimensional_array = np.array([1.2, 2.4, 3.5, 4.7, 6.1, 7.2, 8.3, 9.5])
print(one_dimensional_array)

# create a two dimensional array
two_dimensional_array = np.array([[6, 5], [11, 7], [4, 8]])
print(two_dimensional_array)

# Create a 2 by 3 array of all zeros and print the array.
zero_array = np.zeros((2,3))
print(zero_array)

# Create a 4 by 6 array of all ones and print the array.
one_array = np.zeros((4,6))
print(one_array)

# populate an array with sequences
sequence_of_integers = np.arange(5, 12)
print(sequence_of_integers)

random_integers_between_50_and_100 = np.random.randint(low=50, high=101, size=(6))
print(random_integers_between_50_and_100)

random_floats_between_0_and_1 = np.random.random([6])
print(random_floats_between_0_and_1)

random_Maatrix = np.random.random((2, 3))
print(random_Maatrix)

random_floats_between_2_and_3 = random_floats_between_0_and_1 + 2.0
print(random_floats_between_2_and_3)

random_integers_between_150_and_300 = random_integers_between_50_and_100 * 3
print(random_integers_between_150_and_300)

feature = np.arange(6, 21)
print(feature)

label = 3 * feature + 4
print(label)

# create a noise array of values between -2 and 2
noise = (np.random.random([15]) - 0.5) * 4
print(noise)

# add the noise to the label
label = label + noise
print(label)

np.random.seed(21)
data = np.random.randint(1, 500, size=(20, 5))

print(data)

# normalize the first and third columns
data[:, 0] = (data[:, 0] - np.amin(data[:, 0])) / (np.amax(data[:, 0]) - np.amin(data[:, 0]))
data[:, 2] = (data[:, 2] - np.amin(data[:, 2])) / (np.amax(data[:, 2]) - np.amin(data[:, 2]))

print(data)