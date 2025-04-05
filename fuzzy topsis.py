import pandas as pd
import numpy as np

class FuzzyNumber:
    def __init__(self, l, m, u):
        self.l = l
        self.m = m
        self.u = u

    def __repr__(self):
        return f"({self.l}, {self.m}, {self.u})"

def fuzzy_topsis(matrix, weights):
    
    norm_matrix = np.zeros(matrix.shape, dtype=object)
    for j in range(matrix.shape[1]):
        col_max = max(matrix[:, j], key=lambda x: x.u).u
        col_min = min(matrix[:, j], key=lambda x: x.l).l
        for i in range(matrix.shape[0]):
            norm_matrix[i, j] = FuzzyNumber(
                (matrix[i, j].l - col_min) / (col_max - col_min),
                (matrix[i, j].m - col_min) / (col_max - col_min),
                (matrix[i, j].u - col_min) / (col_max - col_min)
            )

    # Calculate the weighted normalized decision matrix
    weighted_matrix = np.zeros(matrix.shape, dtype=object)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            weighted_matrix[i, j] = FuzzyNumber(
                norm_matrix[i, j].l * weights[j],
                norm_matrix[i, j].m * weights[j],
                norm_matrix[i, j].u * weights[j]
            )

    # Determine the fuzzy positive ideal solution (FPIS) and fuzzy negative ideal solution (FNIS)
    fpis = [FuzzyNumber(max(weighted_matrix[:, j], key=lambda x: x.u).u,
                        max(weighted_matrix[:, j], key=lambda x: x.m).m,
                        max(weighted_matrix[:, j], key=lambda x: x.l).l) for j in range(matrix.shape[1])]
    fnis = [FuzzyNumber(min(weighted_matrix[:, j], key=lambda x: x.u).u,
                        min(weighted_matrix[:, j], key=lambda x: x.m).m,
                        min(weighted_matrix[:, j], key=lambda x: x.l).l) for j in range(matrix.shape[1])]

    # Calculate the distance of each alternative from FPIS and FNIS
    def distance(a, b):
        return np.sqrt((a.l - b.l)**2 + (a.m - b.m)**2 + (a.u - b.u)**2)

    distances_to_fpis = np.zeros(matrix.shape[0])
    distances_to_fnis = np.zeros(matrix.shape[0])
    for i in range(matrix.shape[0]):
        distances_to_fpis[i] = np.sum([distance(weighted_matrix[i, j], fpis[j]) for j in range(matrix.shape[1])])
        distances_to_fnis[i] = np.sum([distance(weighted_matrix[i, j], fnis[j]) for j in range(matrix.shape[1])])

    # Calculate the relative closeness to the ideal solution
    closeness = distances_to_fnis / (distances_to_fpis + distances_to_fnis)

    # Rank the alternatives based on their closeness values
    ranked_alternatives = np.argsort(closeness)[::-1]  # Sort in descending order

    return closeness, ranked_alternatives

# Example usage with the provided data
matrix = np.array([
    [FuzzyNumber(10, 61, 69), FuzzyNumber(18, 137, 3)],
    [FuzzyNumber(11, 122, 54), FuzzyNumber(24, 142, 6)],
    [FuzzyNumber(12, 75, 57), FuzzyNumber(30, 196, 5)],
    [FuzzyNumber(13, 80, 77), FuzzyNumber(55, 247, 1)]
])
weights = [0.5, 0.5]

closeness, ranked_alternatives = fuzzy_topsis(matrix, weights)

# Output the closeness values and ranks
print("Closeness to ideal solution:", closeness)
print("Ranked alternatives (0-indexed):", ranked_alternatives)

# Output the alternatives in their ranked order.
print("\nAlternatives ranked by closeness:")
for rank, alt_index in enumerate(ranked_alternatives):
    print(f"Rank {rank + 1}: Alternative {alt_index + 1}")