import numpy as np
from numpy.random import random
import matplotlib.pyplot as plt
import scipy
from PIL import Image

"""1-Dimensional Arrays (Vectors)"""
my_array = np.array([1.1, 9.2, 8.1, 4.7])
# print(type(my_array))
# print(my_array.shape)

# print(my_array[2])
# print(my_array.ndim)

"""2-Dimensional Arrays (Matrices)"""
array_2d = np.array([[1, 2, 3, 9],
                     [5, 6, 7, 8]])
# print(f'array_2d has {array_2d.ndim} dimensions')
# print(f'Its shape is {array_2d.shape}')
# print(type(array_2d.shape))
# print(f'It has {array_2d.shape[0]} rows and {array_2d.shape[1]} columns')
# print(array_2d)
# print(array_2d.ndim)

"""access the 3rd value in the 2nd row"""
# print(type(array_2d[1, 2]))
# print(array_2d[1, 2])

"""access an entire row and all the values"""
# print(type(array_2d[0, :]))
# print(array_2d[0, :])

"""access an entire column and all the values"""
# print(type(array_2d[:, 0]))
# print(array_2d[:, 0])

"""N-Dimensional Arrays (Tensors)"""
mystery_array = np.array([[[0, 1, 2, 3],
                           [4, 5, 6, 7]],
                          [[7, 86, 6, 98],
                           [5, 1, 0, 4]],
                          [[5, 36, 32, 48],
                           [97, 0, 27, 18]]])
# print(f'We have {mystery_array.ndim} dimensions')
# print(f'The shape is {mystery_array.shape}')

# print(type(mystery_array))
# print(mystery_array)

"""access the last value from the tensor"""
# print(mystery_array[-1, -1, -1])
# print(mystery_array[2, 1, 3])

"""access the last row of the last matrix"""
# print(mystery_array[-1, -1, :])
# print(mystery_array[2, 1, :])

"""access the first column of the tensor"""
# print(mystery_array[:, :, 0])

"""Use .arange() to create a vector a with values ranging from 10 to 29."""
a = np.arange(10, 30)
# print(a)

"""Create an array containing only the last 3 values of a"""
# print(a[-3:])

"""Create a subset with only the 4th, 5th, and 6th values"""
# print(a[3:6])

"""Create a subset of a containing all the values except for the first 12"""
# print(a[12:])

"""Create a subset that only contains the even numbers (i.e, every second number)"""
# print(a[::2])

"""Reverse the order of the values in a, so that the first element comes last:"""
# print(a[::-1])
# print(np.flip(a))

"""Print out all the indices of the non-zero elements in this array: [6,0,9,0,0,5,0]"""
b = np.array([6, 0, 9, 0, 0, 5, 0])
# nz_indices = b[np.nonzero(b)]
# nz_indices = b[b.astype(bool)]
nz_indices = b[b != 0]
# print(type(nz_indices))
# print(nz_indices)

"""Use NumPy to generate a 3x3x3 array with random numbers"""
z = random((3, 3, 3))
# print(z.shape)
# print(z)

"""Use .linspace() to create a vector x of size 9 with values spaced out evenly between 0 to 100 (both included)."""
x = np.linspace(0, 100, num=9)
# print(x.shape)
# print(x)

"""Use .linspace() to create another vector y of size 9 with values between -3 to 3 (both included). Then plot x and y on a line chart using Matplotlib."""
y = np.linspace(-3, 3, num=9)

# plt.plot(x, y)
# plt.show()

"""Use NumPy to generate an array called noise with shape 128x128x3 that has random values. Then use Matplotlib's .imshow() to display the array as an image."""
noise = random((128, 128, 3))
# print(noise.shape)

# plt.imshow(noise)
# plt.show()

"""Linear Algebra with Vectors"""
v1 = np.array([4, 5, 2, 7])
v2 = np.array([2, 1, 3, 3])

# Python Lists vs ndarrays
list1 = [4, 5, 2, 7]
list2 = [2, 1, 3, 3]

# print(v1 + v2)
# print(list1 + list2)

"""Broadcasting and Scalars"""
arrays = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8]])

# print(arrays + 10)
# print(array_2d * 5)

"""Matrix Multiplication with @ and .matmul()"""
a1 = np.array([[1, 3],
               [0, 1],
               [6, 2],
               [9, 7]])

b1 = np.array([[4, 1, 3],
               [5, 8, 5]])

# print(f'{a1.shape}: a has {a1.shape[0]} rows and {a1.shape[1]} columns.')
# print(f'{b1.shape}: b has {b1.shape[0]} rows and {b1.shape[1]} columns.')
# print('Dimensions of result (n,k),(k,m)->(n,m): (4x2)*(2x3)=(4x3)')

"""Let's multiply a1 with b1. Looking at the wikipedia example above, work out the values for c12 and c33 on paper. Then use the .matmul() function or the @ operator to check your work."""
c = np.matmul(a1, b1)
# c = a1 @ b1
# print(f"Matrix c has {c.shape[0]} rows and {c.shape[1]} columns.")
# print(c)

"""Manipulating Images as ndarrays"""
img = scipy.datasets.face()
# plt.imshow(img)
# plt.show()

"""What is the data type of img? Also, what is the shape of img and how many dimensions does it have? What is the resolution of the image?"""
# print(type(img))
# print(img.ndim)
# print(img.shape)
# print(img)

"""Convert the image to black and white. The values in our img range from 0 to 255."""
# Divide all the values by 255 to convert them to sRGB, where all the values are between 0 and 1.
grey_vals = np.array([0.2126, 0.7152, 0.0722])
# print(grey_vals.shape)

# Next, multiply the sRGB array by the grey_vals to convert the image to grey scale.
srgb_array = img / 255
# grey_img = np.matmul(srgb_array, grey_vals)
grey_img = srgb_array @ grey_vals
# print(grey_img)

# Finally use Matplotlib's .imshow() together with the colormap parameter set to gray cmap=gray to look at the results.
# plt.imshow(grey_img, cmap="gray")
# plt.show()

"""Examples for numpy.flip() and numpy.rot90()"""
test = np.array([[1, 3], [0, 1], [6, 2], [9, 7]])
# print(test)

flip_ind = test[::-1]
# print(flip_ind)

flip_num = np.flip(test)
# print(flip_num)

rotate = np.rot90(test)
# print(rotate)

"""Flip the grayscale image upside down"""
# plt.imshow(np.flip(grey_img), cmap="gray")
# plt.show()

"""Rotate the colour image"""
# plt.imshow(np.rot90(img))
# plt.show()

"""Invert (i.e., solarize) the colour image. To do this you need to converting all the pixels to their "opposite" value, so black (0) becomes white (255)."""
solar_img = 255 - img
# plt.imshow(solar_img)
# plt.show()

"""Use your Own Image!"""
file_name = "yummy_macarons.jpg"
my_img = Image.open(file_name)
img_array = np.array(my_img)
# print(img_array)
# print(img_array.ndim)
# print(img_array.shape)

# plt.imshow(img_array)
# plt.show()

# plt.imshow(255 - img_array)
# plt.show()

