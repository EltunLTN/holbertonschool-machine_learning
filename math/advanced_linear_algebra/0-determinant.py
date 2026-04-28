def determinant(matrix):
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")
    if len(matrix[0]) == 0:
        return 1
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
    if len(matrix) == 3:
        new = matrix[0][0]*(matrix[1][1]*matrix[2][2]-matrix[1][2]*matrix[2][1]) - matrix[0][1]*(matrix[1][0]*matrix[2][2]-matrix[2][0]*matrix[1][2]) + matrix[0][2]*(matrix[1][0]*matrix[2][1]-matrix[2][0]*matrix[1][1])
        return new
    
mat0 = [[]]
mat1 = [[5]]
mat2 = [[1, 2], [3, 4]]
mat3 = [[1, 1], [1, 1]]
mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
mat5 = []
mat6 = [[1, 2, 3], [4, 5, 6]]

print(determinant(mat0))
print(determinant(mat1))
print(determinant(mat2))
print(determinant(mat3))
print(determinant(mat4))
try:
    determinant(mat5)
except Exception as e:
    print(e)
try:
    determinant(mat6)
except Exception as e:
    print(e)