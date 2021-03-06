# import tkinter as tk
import numbers
import numpy as np
import sympy as sp
import qmcpy as qp
import math as math
import itertools as itt
import scipy.special
import matplotlib.pyplot as plt
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, implicit_multiplication)
from sympy.utilities.lambdify import lambdify
import scipy.optimize as scp
import collections
import pandas as bearcats
import tkinter as tk
from tkinter import messagebox as mbox


from scipy.integrate import newton_cotes

import time as time


# Four different, one-dimensional kind of quadratures with uniform input and output structure
# Notes regarding the quadratures:
#      We chose to use the trapezoidal quadrature, the Newton-Cotes quadrature (deterministic)
#           the Monte Carlo quadrature (nested and not nested) and the quasi Monte Carlo quadrature (random)
#      Regarding the amount of points used for the approximation we use m = 2^q points, for all quadratures,
#      because this makes it easier to use nested data.

# For the self written methods, the package abbreviations were removed.

# Trapezoidal quadrature

# Method gives back the data used for one-dimensional trapezoidal quadrature
def one_dim_trapezoidal(q: int, a=0, b=1):
    """
    Trapezoidal quadrature

    This function computes the nodes and the weights for the 1D trapezoidal quadrature.

    Args:
        q (int): Degree of quadrature. The number of points used for the quadrature then is 2^q
        a (int, optional): Lower boundary of integral. Defaults to 0.
        b (int, optional): Upper boundary of integral. Defaults to 1.

    Raises:
        Exception: a and b need to be real numbers.
        Exception: b as the upper boundary has to be bigger than a.
        Exception: q needs to be an integer.

    Returns:
        [array, array]: nodes and the weights for the 1D trapezoidal quadrature of degree q.
    """

    # Boundaries of interval need to be numbers and b needs to be bigger than a.
    if not ((type(a) == int or type(a) == float) and (type(b) == int or type(b) == float)):
        raise Exception("a and b need to be real numbers.")

    if b <= a:
        raise Exception("b as the upper boundary has to be bigger than a.")

    # error if m is not a integer because of def.
    if (not isinstance(q, int)) and (not issubclass(type(1, np.interger))):
        raise Exception("q needs to be an integer.")

    m = 2 ** q  # because we have m + 1 points we calculate - 1 to get 2^m points

    points = a + np.dot(range(0, m + 1), (b - a) / m)

    weights = np.ones(m + 1)
    weights[1:m] = 2
    weights = weights * (b - a) / (2 * m)
    return [points, weights]


# Newton-Cotes quadrature
# Method giving back the data needed for Newton-Cotes quadrature

def one_dim_newton_cotes(q: int, a=0, b=1):
    """
    Newton-Cotes quadrature

    This function computes the nodes and weights for the 1D Newton-Cotes quadrature.
    This is a program helping me getting into programming. On the internet better implementation of
    this very basic and not very precise quadrature. Apart from this, the quadrature will be used for the
    testing of the viewer at the start. !! Effectively we have the open Newton-cotes formula of the degree (2^q - 1)!!

    Args:
        q (int): Degree of quadrature.
        a (int, optional): Lower boundary of integral. Defaults to 0.
        b (int, optional): Upper boundary of integral. Defaults to 1.

    Raises:
        Exception: a and b need to be real numbers.
        Exception: b as the upper boundary has to be bigger than a.
        Exception: q needs to be an integer.
        Exception: Please choose q < 6 for the Newton-Cotes approximation to avoid numerical errors.

    Returns:
        [array, array]: nodes and weights for the 1D Newton-Cotes quadrature of degree q.
    """

    # Boundaries of interval need to be numbers and b needs to be bigger than a.
    if not ((type(a) == int or type(a) == float) and (type(b) == int or type(b) == float)):
        raise Exception("a and b need to be real numbers.")

    if b <= a:
        raise Exception("b as the upper boundary has to be bigger than a.")

    # error if m is not a integer because of def.
    if (not isinstance(q, int)) and (not issubclass(type(1, np.interger))):
        raise Exception("q needs to be an integer.")

    # Error if q is getting to high, because calculations are getting to costly and numeric errors can occur.
    if q > 5:
        raise Exception("Please choose for the a q < 6 for the Newton-Cotes approximation. (Numerical errors)")

    m = 2 ** q  # because we have m + 1 points we calculate - 1 to get 2^m points

    points = a + np.dot(range(0, m + 1), (b - a) / m)

    # External methode is used, because own implementation slower by about one order.
    weights, error = newton_cotes(len(points) - 1, 1)
    weights = weights * (b - a) / m
    return [points, weights]


# Monte Carlo method
# Method gives back the data used for one-dimensional Monte Carlo quadrature


def monte_carlo_quad(q: int, a=0, b=1):
    """
    Monte Carlo quadrature

    This function computes the nodes and weights for the 1D Monte carlo quadrature.

    Args:
        q (int): degree of approximation. Number of nodes, m = 2^(q-1)+1.
        a (int, optional): Lower boundary of integral. Defaults to 0.
        b (int, optional): Upper boundary of integral. Defaults to 1.

    Raises:
        Exception: a and b need to be real numbers.
        Exception: b as the upper boundary has to be bigger than a.
        Exception: q needs to be an integer.

    Returns:
        [array, array]: nodes and weights for the 1D Monte carlo quadrature of degree q.
    """

    # Boundaries of interval need to be numbers and b needs to be bigger than a.
    if not ((type(a) == int or type(a) == float) and (type(b) == int or type(b) == float)):
        raise Exception("a and b need to be real numbers.")

    if b <= a:
        raise Exception("b as the upper boundary has to be bigger than a.")

    if (not isinstance(q, int)) and (not issubclass(type(1, np.interger))):
        raise Exception("q needs to be an integer.")

    m = 2 ** q + 1  # number of points where to evaluate function

    # m times a random point is saved in the list

    points = np.array(np.random.uniform(a, b, m))

    weights = 1 / m * np.ones(m)
    return [points, weights]


# Quasi-Monte Carlo quadrature
# Method giving back the data used for one-dimensional quasi Monte-Carlo quadrature

def qmc_quad(q: int, a=0, b=1):
    """
    Quasi-Monte Carlo quadrature

    This function computes the nodes and weights for the 1D Quasi Monte carlo quadrature using lattice rules.

    Args:
        q (int): degree of approximation. Number of nodes, m = 2^q.
        a (int, optional): Lower boundary of integral. Defaults to 0.
        b (int, optional): Upper boundary of integral. Defaults to 1.

    Raises:
        Exception: a and b need to be real numbers.
        Exception: b as the upper boundary has to be bigger than a.
        Exception: q needs to be an integer.

    Returns:
        [array, array]: nodes and weights for the 1D QMC quadrature of degree q.
    """

    # Boundaries of interval need to be numbers and b needs to be bigger than a.
    if not ((type(a) == int or type(a) == float) and (type(b) == int or type(b) == float)):
        raise Exception("a and b need to be real numbers.")

    if b <= a:
        raise Exception("b as the upper boundary has to be bigger than a.")

    # error if m is not a integer because of def.
    if (not isinstance(q, int)) and (not issubclass(type(1, np.interger))):
        raise Exception("q needs to be an integer.")

    # number of points used.
    m = 2 ** q + 1

    # Construction of one-dimensional Lattice. After this calling of m points in it
    qmc_lattice = qp.Lattice(dimension=1)

    # For q = 20 the limit of points are reached, for which can be generated by the methode.
    if q < 20:
        # Easy methode to make one-dimensional array out of array returned n-dim. array.
        points = np.append(qmc_lattice.gen_samples(m), [])
    else:
        # every time a new lattice needs to be called, because otherwise the same points are generated again.
        qmc_lattice = qp.Lattice(dimension=1)
        points = qmc_lattice.gen_samples(2 ** 19 + 1)
        for k_1 in range(2 ** (q - 19) - 1):
            points = np.append(points, qmc_lattice.gen_samples(2 ** 19))

    weights = np.ones(m) / m
    return [points, weights]


# Method calculating the Smolyak algorithm

def smolyak(f, weights: list, points: list, q: int, dim: int, is_nested=False) -> numbers:
    """
    The Smolyak algorithm

    Method to compute an approximation of the integral of a function using the Smolyak algorithm.

    Args:
        f (function): Callable sympy function.
        dim (int): Dimension of the function f.
        weights (list): Numpy array with saved 1D weights corresponding to the points.
        points (list): Points on which the function is evaluated for the 1D integral approx.
        q (int): Degree of approximation.
        is_nested (boolean, optional): Defines, if old evaluations are reused. Default is False.

    Raises:
        Exception: The list of points and weights given for the Smolyak have incorrect lengths.
        Exception: The dimensions of the 1D gridpoints and weights do not match.
        Exception: Something went wrong with the usage of the nested information.
        Exception: Up to now the input vectors should be one dimensional but are not.

    Returns:
        float: result of approximation
    """

    #  Error, if the lists for weights and points are not q - d + 1 entries long.
    if len(weights) != (q - dim + 1) or len(points) != (q - dim + 1):
        raise Exception("The list given to the Smolyak-Methode are not sufficiently long.")

    # We first want to find all elements in the set Q(q,d) as e. g. defined in Def. 3.4 page 11 Bachelor´s thesis
    # finds all combinations in the list rng with length d. This coincides with comb. of one-dim. approx. used for
    # Smolyak-alg.

    # For higher dimension a Method was written scaling in a nicer way.
    if dim < 4 and q < dim * 2:
        # list with all integers bigger than zero smaller than q+1 repeated d times
        rng = list(range(max(1, q - dim + 1))) * dim
        rng = [x + 1 for x in rng]

        # We need to reduce the number of objects that need to be permuted, because otherwise we get into fatal
        # runtimes for higher dimensions. We only exclude the more obvious reductions
        # upper end: we know that we only need one value with of  q - dim + 1 for q - dim + 1 > 1.
        # We know as well that we need two elements of time q - dim and so on.
        for k_1 in range(1, max(rng) + 1):

            if k_1 > 1:
                tup_sum = 0
                times_needed = 0
                while tup_sum < q + 1:
                    times_needed = times_needed + 1
                    tup_sum = k_1 * times_needed + (dim - times_needed)
                for k_2 in range(dim - times_needed + 1):
                    rng.remove(k_1)

            if k_1 < np.floor(q / dim) and dim * k_1 < q - dim:
                rng.remove(k_1)

        # One might optimize by calling permutation of dimension d for possible combinations,
        # but implementation takes long

        possible_combinations = list(set(i for i in itt.permutations(rng, dim) if (q - dim) < sum(i) < q + 1))
        possible_combinations = sort_tuples(possible_combinations, q)
    else:
        possible_combinations = find_all_sensible_combinations(dim, q)
    number_approx = len(possible_combinations)
    Smolyak_result = []  # Predefining return value for reasons of visibility and reviewing results
    function_evaluations = []  # This list is only used for the saving of old evaluations, if the information
    # is nested, but the editor, does not feel safe, if it is defined in an if-clause.

    # Main part of the Methode: Here first an d dimensional gird is made out of the one dimensional vectors
    #                           Then the summands are successively added to the result.
    for i in range(number_approx):

        if isinstance(points[0], np.ndarray) and len(points[0].shape) == 1:

            # the next two steps could be combined in one line.
            # what is done is, that first the tuple at position i is taken from the list of the set I.
            # Then the regarding vectors are taken from the list with the weight and points vectors and
            # are combined to a d dimensional point set

            # gridpoints and the weight vectors are put into separate lists to make it easier to
            # make a meshgrid
            current_tuple = possible_combinations[i]
            current_points = []
            current_weights = []
            for j in range(len(current_tuple)):
                ''' It turned out that calling repeatedly generating random points makes the not converge. 
                    I do not know  why. Especially I did not find an obvious  reason.

                if quadrature == "Monte Carlo (non-nested)":
                    current_points.append(monte_carlo_quad(int(current_tuple[j]))[0])

                elif quadrature == "Quasi-Monte Carlo":
                    current_points.append(qmc_quad(int(current_tuple[j]))[0])
                    print(current_points[-1])

                else:
                '''
                current_points.append(points[current_tuple[j] - 1])
                current_weights.append(weights[current_tuple[j] - 1])

            # meshgrid yields a np.array with all coordinates needed for alg
            # coordinate and respective weight could be found in the same places in the array
            meshgrid_points = np.array(np.meshgrid(*current_points))
            meshgrid_weights = np.array(np.meshgrid(*current_weights))

            if meshgrid_points[0].shape != meshgrid_weights[0].shape:
                # information for later.
                raise Exception("The dimension of the one dimensional gridpoints and weights do not match for the "
                                "vector i = " + str(i))

            # In this if-clause we calculate the intermediate-results using old evaluations. This is only possible , if
            #  the information is nested.
            if is_nested and i > 0 and dim < 5:
                # First we need to  find the already used tuple in Q(d,q), with the most similar evaluations used.
                # For reasons of simplicity, we already have ordered the tuples.
                nearest_index = find_next_tuples_index(current_tuple, possible_combinations[:i])
                old_evals = function_evaluations[nearest_index]  # Grid with the most reusable data

                # Now we have to find the positions  of the reusable data in the different arrays.
                current_shape = meshgrid_points.shape[1:]
                old_shape = old_evals.shape

                # These are returned by this function
                positions_needed_evals, positions_old_evals, position_weights_of_saved_eval = slice_find(current_shape,
                                                                                                         old_shape)
                # The old and new function values are combined to an array saving the full information.
                function_values = np.zeros(current_shape)

                # used old evaluations
                used_old_evals = old_evals[tuple(positions_old_evals)]
                function_values[tuple(position_weights_of_saved_eval[1:])] = used_old_evals

                # new evals
                new_evals = f(*meshgrid_points[tuple(positions_needed_evals)])
                if np.isscalar(new_evals):
                    new_evals = new_evals * np.ones(meshgrid_points[tuple(positions_needed_evals)].shape[1:])

                function_values[tuple(positions_needed_evals[1:])] = new_evals
                # This information needs to be saved for later calculations
                function_evaluations.append(function_values)

                if sum_tuple_shape(used_old_evals.shape, new_evals.shape) != function_values.shape:
                    raise Exception("Something went wrong with the usage of the nested information."
                                    "We have:\n dim. total_points: " + str(meshgrid_weights.shape) + "\n" +
                                    "dim. points used: " + str(positions_old_evals.shape) + "\n"
                                                                                            "dim. points new: " + str(
                                                                                        positions_needed_evals.shape))

            else:

                function_values = f(*meshgrid_points)

                if np.isscalar(function_values):
                    function_values = function_values * np.ones(meshgrid_points.shape[1:])

                # If we take the first evaluations for the nested information, nothing can be reused, but we need the
                if is_nested and i == 0:
                    function_evaluations.append(function_values)

            current_weights = np.prod(meshgrid_weights, axis=0)

            current_factor = (-1) ** (q - sum(current_tuple)) * scipy.special.binom(dim - 1, q - sum(
                current_tuple))
            Smolyak_result.append(current_factor * np.sum(np.multiply(current_weights, function_values)))

        else:
            raise Exception("Up to now the input vectors should be one dimensional but are not.")

    return np.sum(Smolyak_result)


def sum_tuple_shape(old_shape, new_shape):
    """
    Sum up tuples

    Function summing up the dimension of the old information used in the way necessary here.
    This means that if the dimension is the same, this value is saved. In other case, the values are summed up.
    Example: (33,3) + (32,3)  = (65,3).

    Args:
        old_shape (tuple): Dimension of the used information.
        new_shape (tuple): Dimension of the used information.

    Raises:
        Exception: The input data needs to have the same lengths.

    Returns:
        tuple: Combined shape of information.
    """

    dim = len(old_shape)

    # first check if input data reasonable
    if len(new_shape) != dim:
        raise Exception("The input data needs to have the same length")

    combined_shape = np.zeros(dim)

    # run through dimension to add them up
    for k_1 in range(dim):
        if old_shape[k_1] == new_shape[k_1]:
            combined_shape[k_1] = old_shape[k_1]
        else:
            if old_shape[k_1] - 1 != int(new_shape[k_1]):
                raise Exception("The information used in the project does not have this form.")
            else:
                combined_shape[k_1] = int(old_shape[k_1] + new_shape[k_1])

    return tuple(combined_shape)


def sort_tuples(list_of_tuples, q: int):
    """
    Sort tuples

    We need this function to use the advantages of nested information, for this it is easier to first need to sort the list of
    tuples in a way, that we first evaluate the function in a systematical way. This sorting is done here.
    Starting  with the first dimension we sort the list in a rising order.

    This means: input = [(2,1,4),(1,2,2),(1,1,1),(2,1,2)] --> [(1,1,1),(1,2,2),(2,1,2),(2,1,4)]

    Args:
        list_of_tuples (list): List of tuples that needs to be sorted in a specific order.
        q (int): Degree of approximation

    Returns:
        list: Sorted list of tuples.
    """

    # we need a list, because otherwise in the first run, we would access tuples.
    dim = len(list_of_tuples[0])
    tuple_save = [list_of_tuples]

    for k_1 in range(dim):

        former_length = len(tuple_save)
        tuple_list = []

        for k_2 in range(former_length):
            for k_3 in range(1, q - dim + 2):

                bucket = [x for x in tuple_save[k_2] if x[k_1] == k_3]
                if len(bucket) > 0:
                    tuple_list.append(bucket)

        tuple_save = tuple_list
    tuple_save = [single_tuple for sublist in tuple_save for single_tuple in sublist]

    return tuple_save


def find_next_tuples_index(current_tuple, tuple_list):
    """
    Determine index of the matrix with the most reusable information

    Function to find the function evaluation matrix with the most reusable function values

    Args:
        current_tuple (tuple): Tuple saving the degree of approximation currently used for the approximation
        tuple_list (list): list of tuples for which the evaluations are stored.

    Returns:
        integer: The index of the evaluations with the most reusable evaluations.
    """

    dim = len(current_tuple)
    diff = [[current_tuple[k_1] - k_2[k_1] for k_1 in range(dim)] for k_2 in tuple_list]
    direct_smaller = [1 if (collections.Counter(k_1)[0] == dim - 1 and collections.Counter(k_1)[1] == 1) else 0 for k_1
                      in diff]
    if collections.Counter(direct_smaller)[1] == 0:
        neighbour = [1 if (collections.Counter(k_1)[0] == dim - 2 and collections.Counter(k_1)[1] == 1 and
                           collections.Counter(k_1)[-1] == 1) else 0 for k_1 in diff]
        index_evals = neighbour.index(1)
        if isinstance(index_evals, list):
            index_evals = index_evals[0]
    else:
        index_evals = direct_smaller.index(1)
        if isinstance(index_evals, list):
            index_evals = index_evals[0]
    return index_evals


def slice_find(current_shape, nearest_shape):
    """
    Determine the position of the reusable information in the matrix.

    Function returning slices that make it possible to find old evaluations can be used
    for the calculation of the current summand and the points for that further evaluation are needed.
    Args:
        current_shape (tuple): Tuple storing the number of evaluations used for the current approximation.
        nearest_shape (tuple): Tuple storing the number of evaluations used for a former evaluation
        open_quad (boolean, optional): Only defined for example 2. If this is true, the borders of the interval are not
            used, default to False.
    Raises:
        Exception: A unforeseen combination of values occurred.
    Returns:
        [list1, list2, list3]: Three d dimensional lists of slices are returned. In these the slice at the i-th position
            coincide with the points used in this dimension.
        list1: Positions at which a new evaluation is needed.
        list2: Positions at which old evaluations can be used.
        list3: Positions of the weights multiplied with the old evaluations.
    """

    positions_old_evals = []
    positions_needed_evals = [slice(0, len(nearest_shape))]
    position_weights_of_saved_evals = [slice(0, len(nearest_shape))]

    for tuple_dim in range(len(current_shape)):
        if current_shape[tuple_dim] == nearest_shape[tuple_dim]:
            positions_old_evals.append(slice(0, current_shape[tuple_dim]))
            positions_needed_evals.append(slice(0, current_shape[tuple_dim]))
            position_weights_of_saved_evals.append(slice(0, current_shape[tuple_dim]))

        if current_shape[tuple_dim] < nearest_shape[tuple_dim] - 1:
            positions_old_evals.append(slice(0, nearest_shape[tuple_dim], 2))
            positions_needed_evals.append(slice(0, current_shape[tuple_dim] + 1))
            position_weights_of_saved_evals.append(slice(0, current_shape[tuple_dim] + 1))

        if current_shape[tuple_dim] > nearest_shape[tuple_dim] + 1:
            positions_old_evals.append(slice(0, nearest_shape[tuple_dim] + 1))
            positions_needed_evals.append(slice(1, current_shape[tuple_dim], 2))
            position_weights_of_saved_evals.append(slice(0, current_shape[tuple_dim], 2))

    return [positions_needed_evals, positions_old_evals, position_weights_of_saved_evals]


"""
The three methods have the following function:
    find_all_sensible_combinations: Runs through the combination. The choice what to do is made in next step. 
            The vector suiting the requirements are saved and afterwards converted into tuples.
    next_step: If the value in the current position is smaller than q-dim+2, one is added. If not, 
            the next value meeting this condition one is added. The values in the lower positions is set to 1.
    find_right position: Find next position where one needs to be added

    Simple example of how order of vectors: (dim = 2, q = 3)
            a = (1,1) -> add 1 at pos. 0 -> a = (2,1) -> sum(a) == q -> add 1 to pos 2, set pos  0 to 1 -> 
            -> a = (1,2) -> break, because a == (1, q-dim+1)

"""


def next_step(current_vector: np.array, pointer_current_position: int, max_value: int):
    """
    Faster combinatorics

    Function returning, were and what to change in the vectors determining, what one dimensional degree of
    approximation should be applied. (Internal function)

    Args:
        current_vector (np.array):
            full actual vector. (Could be reduced in most cases)
        pointer_current_position (int):
            position currently worked at
        max_value(int):
            necessary to decide when to skip to the next position
    Returns:
        [int, int, int]: The position at which the value needs to be incremented, the next value and a pointer showing,
        in which dimension the next action needs to be performed.
    """

    # Finds the highest position, that needs to be changed.
    if sum(current_vector) < max_value + len(current_vector):
        pointer_next_position = find_right_position(current_vector, pointer_current_position, max_value)
    else:
        pointer_next_position = find_right_position(current_vector, pointer_current_position + 1, max_value)

    # If value in current pos. still is small enough, one can be added
    if pointer_next_position == pointer_current_position:
        next_values = current_vector[pointer_current_position] + 1
        positions_to_change = pointer_current_position
        pointer_next_step = positions_to_change

    # If not, the next position with a sufficiently low value, The lower positions are set to 1.
    else:
        positions_to_change = slice(0, (pointer_next_position + 1))
        next_values = np.ones((pointer_next_position + 1), dtype=int)
        next_values[-1] = current_vector[pointer_next_position] + 1
        pointer_next_step = 0

    return [positions_to_change, next_values, pointer_next_step]


def find_right_position(current_vector, pointer_current_position, max_value):
    """
    Faster combinatorics

    Decides, where to add one.

    Args:
        current_vector (np.array):
            Current combination of degrees of approximation.
        pointer_current_position (int):
            index of dimension, in which needs  to be changed next.
        max_value (int):
            upper bound of degrees of approximation needed in set.
    Returns:
        pointer_current_position (int):
            highest position that needs to be changed in next step
    """

    while current_vector[pointer_current_position] == max_value:
        pointer_current_position += 1

    return pointer_current_position


def find_all_sensible_combinations(dim: int, q: int):
    """
    Faster combinatorics

    Function finding all tuples of combinations of one dim approximations necessary to calc. Smol. alg..

    Args:
        dim (int):
            dimension of vectors
        q (int):
            maximal cumulative sum

    Returns:
        list: List of all combination of degrees of approximation needed for later calculations.
    """

    if q < dim:
        raise Exception("q needs to be  at least equal to the dimension of the vector.")

    # Initialize variables used.
    vector = np.ones(dim, dtype=int)
    max_value = (q - dim + 1)
    save_vector = vector
    current_pointer = 0

    # Defining stop conditions. (vector == [1,1, ..., 1, q-dim+1], because afterwards, no vector could meet
    # requirements.)
    stop_vector = np.ones(dim, dtype=int)
    stop_vector[-1] = q - dim + 1

    while any(vector != stop_vector):
        # Changes to commit.
        positions_to_change, next_values, pointer_current_position = next_step(vector, current_pointer, max_value)

        # Commit changes
        vector[positions_to_change] = next_values

        # Further check, if vector is needs to be used.
        if sum(vector) < (q + 1):
            save_vector = np.c_[save_vector, vector]

    # Correct side effect. (I didn't find an other fast way to prevent this, apart from defining a buffer variable.)
    if len(save_vector.shape) == 2:
        save_vector[0, 0] = 1
        save_vector = [tuple(save_vector[:, k_1]) for k_1 in range(save_vector.shape[1])]
    else:
        save_vector[0] = 1
        save_vector = [tuple(save_vector)]

    # The vectors formerly stored in array are converted into tuples.

    return save_vector


# Error estimation for Smolyak algorithm
# For me the main problem here is the estimation of the estimated error of the probabilistic quadratures.
# Error estimation for Smolyak algorithm
# For me the main problem here is the estimation of the estimated error of the probabilistic quadratures.
# Method calculating the error estimation
def error_smolyak(f, quadrature: str, variables, q: int, function_string_error: str, a=0, b=1, give_parameters=False):
    """
   Error estimation

   This function returns error-estimation for Smolyak algorithm.

   Args:
       f (function): callable sympy function
       quadrature (str): quadrature used
       variables (sympy.variables): sympy variables of function
       q (int): degree of approximation
       function_string_error (str): string of function. (Needed, because analytically deriving strings is easier than
       deriving functions)
       a (float, optional): lower limit of interval, default to 0.
       b (float, optional): upper limit of interval, default to 1.
       give_parameters (boolean, optional): Boolean defined to get the parameters of the error estimation instead of
       the error estimation, default set to False.
   Raises:
       Exception: If the quadrature chosen is not implemented
       Exception: If deterministic quadratures are used and the function can not be derived sufficiently many times.

   Returns:
       float: error of estimation
   """

    # We need to check, whether quadrature is a valid option.
    dim = len(variables)
    valid_options = {"Newton-Cotes", "Trapezoidal", "Monte Carlo (nested)",
                     "Monte Carlo (non-nested)", "Quasi-Monte Carlo"}
    if quadrature not in valid_options:
        raise Exception("the quadrature chosen is not valid.")

    # Predefining the estimations is the easiest way to make the variables global
    B, C, D, E = [1, 1, 1, 1]

    # Now we have to make an estimation for more or all  options

    # For this option we need to find error = |f''|_max (b-a)³/12
    if quadrature == "Trapezoidal" or "Newton-Cotes":
        # We calculate |f''|_max. Hence, we first take the derivative of the function in the different dimensions
        # Then we try to find the maximum of this derivative.
        # !!This is done numerically, such that the result might be wrong !!
        # The highest maximum then is chosen as our one dimensional error estimation e.

        maximum_of_f = 0  # Here we save current maximal absolute value of f

        diff_options = list(itt.product(variables, repeat=2))  # List of options, how you can derive.
        derivations = list()
        for i in range(len(diff_options)):
            derivations.append(sp.diff(function_string_error, diff_options[i][0], diff_options[i][1]))

        # Here we want to check, if the function is 2 times steady differentiable and if make the functions callable.
        callable_derivation = []

        for i in range(len(diff_options)):
            for j in range(i):
                if collections.Counter(diff_options[i]) == collections.Counter(diff_options[j]):
                    if derivations[i] != derivations[j]:
                        raise Exception('The function is not two times steady differentiable. Hence we can not find '
                                        'estimation for error')

                    else:
                        if derivations[i] == 0:
                            maximum_of_f = 0
            if collections.Counter(diff_options[i]) != collections.Counter(diff_options[:i]):
                transformations = standard_transformations + (implicit_multiplication,)
                function = parse_expr(str(derivations[i]), transformations=transformations)
                variable_tuple = tuple(variables)
                callable_derivation.append(lambdify([variable_tuple], -abs(function)))

        # Now we find a maximum. Here we only use 3 evaluations for every dimension, because this part of the
        # calculation should not need to much
        for i in range(len(diff_options)):
            if not (diff_options[i] in diff_options[:i]):

                bound_points = []
                for j in range(dim):
                    bound_points.append((a, b))

                random_start_points = a + (b - a) * np.random.rand(3 * dim, dim)
                for k_1 in range(len(callable_derivation)):
                    for k_2 in range(len(random_start_points)):

                        search_max = scipy.optimize.minimize(callable_derivation[k_1], random_start_points[k_2],
                                                             bounds=bound_points)
                        max_point = -callable_derivation[k_1](search_max.x)
                        if search_max.success and (max_point > maximum_of_f):
                            maximum_of_f = max_point

        one_d_max = maximum_of_f * ((b - a) ** 3) / 12

        # Defining variables used for error estimation
        if quadrature == "Trapezoidal":
            B = 1
            C = one_d_max
            D = (1 / 2) ** 2
            E = C * ((D + 1) / D)
        else:
            B = 1
            C = one_d_max
            D = (b - a) / (2 ** 2 * 3 ** 2)  # Estimation of ((a-b)/(2n))**(2*n+1) * 1/(2n+1)! ... with n  >= 2.
            E = C * ((D + 1) / D)

    # We calculate the error for all probabilistic quadratures using a qmcpy lattice, because the variance of this
    # result is not that high.
    if quadrature == "Quasi-Monte Carlo" or quadrature == "Monte Carlo (nested)" \
            or quadrature == "Monte Carlo (non-nested)":
        Lattice = qp.Lattice(dimension=dim)
        points = a + (b - a) * np.array(Lattice.gen_samples(128 * dim))
        sample = [f(*points[k_1]) for k_1 in range(len(points))]
        sigma = np.sqrt(np.var(sample))

        # Here we choose the 95% error-level of 2 standard deviations
        B = 1
        C = 2 * sigma / np.sqrt(3)
        D = 1 / np.sqrt(2)
        E = C * ((D + 1) / D)

    # Now we calculate the error of the Smolyak algorithm
    # Different for nested and not nested
    error = C * (B ** (dim - 1)) * (D ** (q - dim + 1)) * np.sum([(((E * D) / B) ** j) *
                                                                  scipy.special.binom(q - dim + j, j) for j in
                                                                  range(dim)])

    if give_parameters:
        return [B, C, D, E, max(B / D, E)]
    return error


def epsilon_error(function_string: str, variable_string: str, quadrature: str, epsilon: numbers):
    """
    Epsilon cost

    Function calculating the epsilon error.

    Args:
        function_string (str): string of function
        variable_string (str): string of variables
        quadrature (str): string of one-dimensional quadrature to use
        epsilon (float): upper bound of error
        nested (boolean, optional): Decides if the equation for nested or not nested information is used, default is
        False.
        open_quad (boolean, optional): Decides if a open or closed quadrature is used, default is
        False.
    Raises:
        Exception: Exception: If the quadrature chosen is not implemented.

    Returns:
        [float, float]: The estimated error and cost estimation.
    """

    # Error handling for quadrature
    valid_options = {"Newton-Cotes", "Trapezoidal", "Monte Carlo (nested)",
                     "Monte Carlo (non-nested)", "Quasi-Monte Carlo"}
    if quadrature not in valid_options:
        raise Exception("the quadrature chosen is not valid.")

    # We need a callable function to calculate the error constants
    f, variables, function_string = rewrite_function(function_string, variable_string)
    dim = len(variables)

    # Error constants are called here
    B, C, D, E, H = error_smolyak(f, quadrature, variables, len(variables) + 2,
                                  function_string, give_parameters=True)
    # For this project we already know that these cost constants can be taken
    # F_0 = 2
    # F = 2

    # Now we implement the theorem 15.6 of Tractability of Multivariate Problems pages 348 and 349
    # alpha = np.log(F) / np.log(1 / D)

    if quadrature == "Monte Carlo (non-nested)" or quadrature == "Quasi-Monte Carlo":
        # alpha_3 = alpha + 1
        # alpha_2 = ((np.e * F_0 ** (1 / (alpha + 1)) * H ** (alpha / (alpha + 1))) / ((np.e - 1) * np.log(1 / D)))
        # alpha_1 = alpha_2 * np.log(((np.e * H) / (np.log(1 / D))))
        # alpha_0 = (dim / (2 * np.pi * (dim - 1))) ** ((alpha + 1) / 2) * ((F_0 * F ** 2) / ((F - 1) * D))
        h = ((np.e * H) / (np.log(1 / D))) * (C / epsilon * np.sqrt(dim / (2 * np.pi * (dim - 1)))) ** (1 / (dim - 1))

    else:
        # alpha_3 = alpha / 2 + 1
        # alpha_2 = ((F_0 * (F - 1)) / F) ** (2 / (alpha + 2)) * ((np.e ** 2) / ((np.e - 1) * np.log(1 / D))) * \
        #          (C / D) ** ((2 * alpha) / (2 + alpha))
        # alpha_1 = alpha_2 * np.log(((np.e * C ** 2) / (2 * D ** 2 * np.log(1 / D))))
        # alpha_0 = (dim / (2 * np.pi * (dim - 1))) ** ((alpha + 2) / 4) * ((F_0 * F) / (D ** 2))
        h = ((np.e * C ** 2) / (2 * D ** 2 * np.log(1 / D))) * \
            ((C ** 2) / epsilon ** 2 * np.sqrt(dim / (2 * np.pi * (dim - 1)))) ** (1 / (dim - 1))

    # For the calculation of t, a series can be used.Often the real value does  not change.
    delta = 1
    t_0 = np.e / (np.e - 1) * np.log(h)
    t_last = t_0
    t_k = t_0

    # Break off rule delta < 0.01 is arbitrary, but should normally work
    while delta > 0.01:
        t_k = np.log(h * t_last)
        delta = t_k - t_last
        t_last = t_k

    q_epsilon = math.floor(t_k * (dim - 1) / (np.log(1 / D)))

    return q_epsilon


# Cost estimation
# Due to the similar structure of the implementation of the quadratures, it is possible to estimate the cost in a  very
# homogeneous way.
# Method calculating the cost
def cost_smolyak(q: int, dim: int, nested=True):
    """
    Cost calculation

    Function giving back an estimation of the cardinality of the information needed for the algorithm.
    Due to the definition of the. sequences of the one dimensional algorithms used, the cost could be calculated
    relatively quickly

    Args:
        q (int): Degree of approximation
        dim (int): Dimension
        nested (boolean, optional): Decides if information is nested, default True.

    Returns:
        float: estimated cost of approximation.
    """
    # We first estimate m_i = 2 ** i < 2 (2 ** i - 1) for all i > 0
    F_0 = 2
    F = 2

    # With p = 1/2 the condition m_i = O(D**(-(1/p))) also is fulfilled for all quadratures used.

    # We take the least exact  estimation, because this is used in calculation of $epsilon$-cost
    if nested:
        m_q_d = F_0 ** dim * (F - 1 / F) ** (dim - 1) * F ** q * scipy.special.binom(q - 1, dim - 1)
    else:
        m_q_d = F_0 ** dim * F ** (q + 1) / (F - 1) * scipy.special.binom(q - 1, dim - 1)

    return m_q_d


# Controller of project
# Controller (could also be called for calculation)
# For this we first need to parse the string of the function and make it callable using Sympy methods.
def rewrite_function(function_string_rewrite: str, variables_string_rewrite: str, a=0, b=1) -> list:
    """
    Rewrite string to function

    This function converts strings to Sympy objects.

    Args:
        function_string_rewrite (str): function to be approximated
        variables_string_rewrite (str): variables used in function
        a (float): lower border of interval
        b (float): upper border of interval

    Raises:
        Exception: If no variable defined.
        Exception: If the set of variables used in function and defined in variable  string do not coincide.

    Returns:
        [sympy.function, sympy.variables, string]: function and variable used for approximation. String rewritten
            with standardized variables.
    """

    # Firstly we avoid the problem due to different specific notation of ^ in python

    function_string_rewrite = function_string_rewrite.replace("^", "**")

    # We convert variable string into list of the different variables
    # Then we cut of all blank spaces and convert it into a Symbol used in the SymPh package

    variables_string_rewrite = variables_string_rewrite.replace(" ", "")
    variables = variables_string_rewrite.strip(')(').split(',')
    variables_string_rewrite = variables_string_rewrite.strip(')(').split(',')

    # Throw exception if no variables entered
    if len(variables) == 0:
        raise Exception("There at least has to be one variable.")

    # The next step is to make the function readable, such that if x and y are variables, xy is
    # read as x*y.
    # Furthermore we make all make all write all variables as x_i, i in N
    for f in range(len(variables)):
        function_string_rewrite = function_string_rewrite.replace(str(variables_string_rewrite[f]), "x_" + str(f) + " ")
        variables[f] = sp.Symbol("x_" + str(f))

    # We convert the string into a function relying on the package sympy
    # After this we make it easier and (much) faster to call the function by
    # using the lambdify methods.
    transformations = standard_transformations + (implicit_multiplication,)

    function_string_rewrite = parse_expr(function_string_rewrite, transformations=transformations)
    # Here the function could be evaluated by the substitute  function
    # This is very costly for longer functions and many evaluations
    # Therefore we make it callable by entering a tuple of values

    variable_tuple = tuple(variables)
    function = lambdify(variable_tuple, function_string_rewrite)

    # Last test whether the function is evaluable with given variables
    test_value = function(*tuple((a + 0.5 * (b - a)) * np.ones(len(variable_tuple))))
    if not (isinstance(test_value, float) or isinstance(test_value, int)):
        raise Exception("Not all variables were given. Please enter full set of variables.")

    return [function, variables, function_string_rewrite]


def controller_smolyak(function_string_control: str, variables_string_control: str, quadrature: str, q: int, a=0, b=1,
                       no_error=False):
    """
    Controller of project

    This function gets a function and a few settings and return an approximation, an error-estimation and a
    cost-estimation of the integral of this function.

    Args:
        function_string_control (str): string in which the function is defined
        variables_string_control (str): string for safe use, to be sure that right variables are used in function
        quadrature (str): variable with 4 options to choose quadrature that should be used for approx
        q (int): degree of approximation
        no_error (boolean, optional): Decides, if error should be calculated. (Sometimes the time can be reduced
            relevantly.) Default is set to False.
        a (float, optional): lower limit of interval, default set to 0.
        b (float, optional): upper limit of interval, default set to 1.

    Raise:
        Exception: If the type of quadrature chosen is not defined.

    Returns:
        [float, float, float]: result of approximation, upper bound of error of approximation and upper bound of cost.
    """

    f, variables, function_string_control = rewrite_function(function_string_control, variables_string_control)

    dim = len(variables)  # dimension of function

    list_of_weights = []
    list_of_points = []

    # Now we apply the quadrature chosen by user
    # !! If quadratures is changed or extended in interface, please options here !!

    # Because the points used for the deterministic quadratures are not changed, it is faster to call them
    # at the beginning before starting the Smolyak alg.
    if quadrature == "Newton-Cotes":

        # For every one dimensional degree of approx.  a weights and points vector
        for i in range(1, (q - dim + 2)):
            points, weights = one_dim_newton_cotes(i, a, b)
            list_of_points.append(points)
            list_of_weights.append(weights)
        # With these vectors the  methode smolyak is  called. This calculates an approx. using the Smolyak-alg.
        Smolyak_result = smolyak(f, list_of_weights, list_of_points, q, dim, is_nested=True)

    elif quadrature == "Trapezoidal":
        for i in range(1, (q - dim + 2)):
            points, weights = one_dim_trapezoidal(i, a, b)
            list_of_points.append(points)
            list_of_weights.append(weights)
        Smolyak_result = smolyak(f, list_of_weights, list_of_points, q, dim, is_nested=True)

    elif quadrature == "Monte Carlo (nested)" or quadrature == "Monte Carlo (non-nested)":
        for i in range(1, (q - dim + 2)):

            # So far the user  chose the M-C quad. to be nested repeatedly the old points are reused to construct new
            # vector of points.
            if quadrature == "Monte Carlo (nested)" and i > 1:
                m = 2 ** (i - 1) + 1
                points_new, weights = monte_carlo_quad(i - 1, a, b)

                # Here a we produce the same structure like in the deterministic quadratures. With respect to the
                # quadrature before every second point is new.
                points = np.zeros(2 * m - 1)
                points[slice(0, len(points), 2)] = list_of_points[-1]

                points[slice(1, len(points), 2)] = points_new[:-1]

                # The last point needs to be cut off, because otherwise we would use 2 ** i + 2, not 2 ** i +1 points
                weights = 1 / (2 * m - 1) * np.ones(2 * m - 1)
            else:
                points, weights = monte_carlo_quad(i, a, b)

            list_of_points.append(points)
            list_of_weights.append(weights)

        if quadrature == "Monte Carlo (nested)":
            Smolyak_result = smolyak(f, list_of_weights, list_of_points, q, dim, is_nested=True)
        else:
            Smolyak_result = smolyak(f, list_of_weights, list_of_points, q, dim)

    elif quadrature == "Quasi-Monte Carlo":
        for i in range(1, (q - dim + 2)):
            points, weights = qmc_quad(i, a, b)
            list_of_points.append(points)
            list_of_weights.append(weights)

        Smolyak_result = smolyak(f, list_of_weights, list_of_points, q, dim)

    else:
        raise Exception("Something went wrong with list of possible quadratures.")

    # calculation of error estimation
    # In some situation the calculation of the error takes much longer than the Smolyak algorithm. For these we
    # can turn of the calculation of the error.
    if not no_error:
        error = error_smolyak(f, quadrature, variables, q, function_string_control)
    else:
        error = "None"

    # calculation of cost estimation
    # algorithms chosen in a way that only exceptions Quasi-Monte carlo and Monte Carlo non-nested.
    # Here non  nested  information need to be used
    if quadrature == "Quasi-Monte Carlo" or quadrature == "Monte Carlo (non-nested)":
        cost = cost_smolyak(q, dim, False)
    else:
        cost = cost_smolyak(q, dim)

    # giving back results that should be displayed
    return [Smolyak_result, error, cost]


# A methode for plotting
# Needs to be modified to make plotting more flexible and design nicer
def modified_scatter_plot(x, y, title="Gridpoints used for Smolyak-algorithm", input_list=False, a=0, b=1):
    """
    Scatter plot

    Function for quick application some basic modifications to a scatter plot of the gridpoints
    used for the Smolyak algorithm
    Args:
        x (list): list of x-values of points
        y (list): list of y-values of points
        title (str): title of plot
        input_list (boolean, optional): Boolean saying, whether input is list. If True, the points in different
            lists are plotted in different colours, default is False.
        a (float, optional): lower border of interval
        b (float, optional): upper border of interval

    Returns:
        None
    """

    len_input = 1  # number of objects saved in list
    if input_list:
        len_input = len(x)
        if len_input != len(y):
            raise Exception("If x and y are lists, these need to have the same length ")

    # First we prevent some obvious errors
    # x and y need to have the same length
    if input_list:
        for i in range(len_input):
            if x[i].shape != y[i].shape:
                raise Exception("The x and y dimension of the points need to shape. Problems in entry number " + str(i))

    else:
        if x.shape != y.shape:
            raise Exception("The x and y dimension of the points need the same shape")

    if input_list:

        # list of letters encoding colours
        color_code = ["b", "r", "b", "g", "c", "m", "y"]
        for i in range(len_input):
            plt.scatter(x[i], y[i], color=color_code[i % len(color_code)])
    else:
        plt.scatter(x, y, color="b")

    # Some cosmetic changes in plot
    plt.xlabel("x-values", fontsize=16, fontweight="bold")
    plt.ylabel("y-value", fontsize=16, fontweight="bold")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(a - (b - a) / 10, b + (b - a) / 10)
    plt.ylim(a - (b - a) / 10, b + (b - a) / 10)
    plt.title(title, fontsize=20, fontweight="bold")


# Comment: A better data_structure might easily be found.
def calculate_stat_data(function_string_stat_calc: str, var_string: str, option_list_quad: str,
                        filename: str, degree_of_app_range: list, repetitions=2 ** 10, damping=True,
                        damping_exp=1.5):
    """
    Calculate the estimation many times

    This function calculates and saves the result, error and runtime Smolyak approximation for the quadratures
    given in option_list several times.

    Args:
        function_string_stat_calc (string): function for which the integral should be approximated.
        var_string (sting): variables of function for which the integral should be approximated.
        option_list_quad (list): list with the quadratures that should be used.
        filename (string): name of the file in which you want to save the data
        degree_of_app_range (list): list with degrees of approximation, for which data should be generated
            repetitions: Times the results  are generated.
        repetitions (int): Times the results  are generated.
        damping (boolean, optional): boolean saying, whether the number of repetition should decrease according to the
            degree of approximation. Depending on the range of degrees of approximation and the computer recommendable,
            default is True.
        damping_exp (float): damping exponent of the degree of approximation, if damping is true.

    Raises:
        Exception: If one of the quadrature is not defined.
        Exception: If the filename is not valid
        Exception: If degree_of_app_range does not have the right length.

    Returns:
        None
    """

    possible_options = ["Newton-Cotes",
                        "Trapezoidal",
                        "Monte Carlo (nested)",
                        "Monte Carlo (non-nested)",
                        "Quasi-Monte Carlo"]

    # Some error handling
    if not (set(option_list_quad) <= set(possible_options)):
        raise Exception("Please check whether all options called are valid quadratures implemented.")

    if not ("." in filename):
        filename = filename + ".csv"
    elif len(filename.split(".")) > 2:
        raise Exception("There seem to be to many dots in the filename.")

    if degree_of_app_range[1] > 21 and damping and repetitions > 10 ** 3:
        print("Ok, you are the boss. Hopefully your computational power is sufficient. ")
    if not (isinstance(degree_of_app_range, list)) and len(degree_of_app_range) != 2:
        raise Exception("The range of the degree of approximation needs to be list of the length 2.")

    # Lists in which all the data is saved.
    result_save = []
    error_save = []
    run_time_save = []

    # We have different lists for different quadrature options
    for i in range(len(option_list_quad)):
        result_list = []
        error_list = []
        time_list = []

        # In the lists for the options the approximations for the quadratures are saved.
        for q in range(degree_of_app_range[0], degree_of_app_range[1]):
            result_quad = []
            error_quad = []
            run_time = []

            for k_1 in range(int(repetitions / q ** damping_exp)):
                start = time.time()  # Time before calculation

                # Calling the function
                result, error, cost = controller_smolyak(function_string_stat_calc, var_string,
                                                         option_list_quad[i], q)
                stop = time.time()  # End time of calculation

                result_quad.append(result)
                error_quad.append(error)
                run_time.append(stop - start)

            time_list.append(run_time)
            result_list.append(result_quad)
            error_list.append(error_quad)

        result_save.append(result_list)
        error_save.append(error_list)
        run_time_save.append(time_list)

        # We want to save the data in a data file

    result_write = result_save
    error_write = error_save
    time_write = run_time_save

    # We need to append some clearly identifiable elements to some lists, so that all lists have the same length.
    for k_1 in range(len(result_save)):
        for k_2 in range(len(result_save[k_1])):
            for k_3 in range(len(result_save[k_1][k_2]), len(result_save[k_1][0])):
                result_write[k_1][k_2].append("")
                error_write[k_1][k_2].append("")
                time_write[k_1][k_2].append("")

    # The saving is done using specific methods of the pandas package. (a bearcat is a panda)
    save_data = bearcats.DataFrame()
    save_data["results"] = result_write
    save_data["errors"] = error_write
    save_data["time_write"] = time_write
    save_data.to_csv(filename, index=False)
    print("ready")


# After saving the data in a .csv file we  need to reload it to the memory.
# !! If other characteristics are saved by the method calculate_stat_data is changed, here the characteristics called
# also need to be changed!!
def load_stat_data(filename: str):
    """
    Load data calculated by calculate_stat_data

    Function loading the formerly saved data to the memory. !! This method only should be used for the
    calculate_stat_data, because the data structure is relatively specific.
    Args:
        filename: string of name of file that should be loaded.

    Raises:
        Exception: If the file name has the wrong format.

    Returns:
        List: Contains the results of the approximation.
    """

    if not ("." in filename):
        filename = filename + ".csv"
    elif len(filename.split(".")) > 2:
        raise Exception("There seem to be to many dots in the filename.")

    data_nested = bearcats.read_csv(filename)

    # The data is read as string. We convert it to a list
    string_opt = ["results", "errors", "time_write"]

    # The was the data is saved, we get it back as long string of numbers. For this we need to split the string up
    #  with respect to commas and square brackets. It  turned out, that this code gets  this done.
    results = []
    for k_1 in range(len(string_opt)):
        results.append([])
        for k_2 in range(len(data_nested[string_opt[k_1]])):
            results[k_1].append([])
            data_string = data_nested[string_opt[k_1]][k_2].split("[")
            data_string = [data.replace("]", "") for data in data_string]
            data_string = [data for data in data_string if len(data) > 10]
            data_string = [data.split(", ") for data in data_string]
            for k_3 in range(len(data_string)):
                data_string_lists = [data for data in data_string[k_3] if data != "''" and data != ""]
                data = [float(data) for data in data_string_lists]
                results[k_1][k_2].append(data)

    return results


def View():
    """
    View

    View of project.

    Returns:
      None
    """

    root = tk.Tk()
    root.title("Application for calculation of integral with Smolak algorithm")
    root.geometry("800x300")

    # Label for description of functionality of viewer
    description = tk.Label(root, text="To  use the application you need first need to enter a function in the "
                                      "entry.\n"
                                      "After this the tuple of variables and in the end the quadrature you want to be "
                                      "used.\n"
                                      "The calculation starts after clicking start.", anchor="w", justify="left")
    description.grid(row=0, column=0, sticky="w", columnspan=3)

    #
    # Column for entry of function
    func_label = tk.Label(root, text="Please enter function: ", justify="center", anchor="w", pady=10, padx=10)
    func_label.grid(row=1, column=0, sticky="w")

    func_entry = tk.Entry(root, width=50, borderwidth=5)
    func_entry.grid(row=1, column=1, sticky="e", padx=20, pady=20, columnspan=2)
    func_entry.insert(0, "2 x y")

    # Column for entry of variables
    var_label = tk.Label(root, text="Please enter variables: ", justify="center", anchor="w", pady=10, padx=10)
    var_label.grid(row=2, column=0, sticky="w")

    var_entry = tk.Entry(root, width=50, borderwidth=5)
    var_entry.grid(row=2, column=1, sticky="e", padx=20, pady=20, columnspan=2)
    var_entry.insert(0, "(x , y)")

    # Drop down menu for quadrature

    # list of quadratures available
    option_list = ["Newton-Cotes",
                   "Trapezoidal",
                   "Monte Carlo (nested)",
                   "Monte Carlo (non-nested)",
                   "Quasi-Monte Carlo"]
    option = tk.StringVar()
    option.set(option_list[0])
    which_quad = tk.OptionMenu(root, option, *option_list)
    which_quad.grid(row=3, column=2)

    degree_label = tk.Label(root, text="Please choose degree of approximation: ", justify="center", anchor="w",
                            pady=10, padx=10)
    degree_label.grid(row=3, column=0, sticky="w")

    degree_of_approx = tk.Entry(root, width=10, borderwidth=5)
    degree_of_approx.grid(row=3, column=1)
    degree_of_approx.insert(0, 2)

    # !! Here we also could  think about a button making it possible to define the epsilon-error!!

    # Start button
    def start():
        # read out entries
        function_string = func_entry.get()
        variables_string = var_entry.get()
        quadrature = option.get()
        degree = int(degree_of_approx.get())

        # We want to open an error box, if the degree of approximation is not big enough. Hence, we need to
        # know the dimension of the function. Here we get this by counting the commas.
        if degree < (variables_string.count(",") + 1):
            mbox.showerror("Input error", "The degree of the approximation needs to be at least as high as the "
                                          "dimension of m the function. \n\nIn this case this would be q = "
                           + str(variables_string.count(",") + 1) + ".")

        else:
            # ask controller for benchmarks of approximation
            result_approx, error, cost = cont.controller_smolyak(function_string, variables_string, quadrature, degree)

            # Create new window where results are shown
            result_window = tk.Toplevel(root)
            result_window.title("Result")
            result_window.geometry("800x200")

            # Label for result of approx.
            result_label = tk.Label(result_window, text="The result of the approximation is: " + str(result_approx),
                                    justify="center", anchor="e", pady=10, padx=10)

            result_label.grid(row=0, column=0, columnspan=2)

            # Label for error
            error_label = tk.Label(result_window, text="Estimated error: " + str(error), justify="center", anchor="w",
                                   pady=10, padx=10)
            error_label.grid(row=1, column=0)

            # Label for cost estimation
            cost_label = tk.Label(result_window, text="Estimated number of evaluations: " + str(cost), justify="center",
                                  anchor="w", pady=10, padx=10)
            cost_label.grid(row=1, column=2)

            # Quit button of result window
            quit_button_result = tk.Button(result_window, text="Quit", command=root.destroy, width=10, borderwidth=5,
                                           padx=5, pady=5)
            quit_button_result.grid(row=2, column=2)

    #  Start button
    start_button = tk.Button(root, text="Start", command=start, width=10, borderwidth=5, padx=5, pady=5)

    start_button.grid(row=4, column=0)

    # Quit button
    quit_button = tk.Button(root, text="Quit", command=root.destroy, width=10, borderwidth=5, padx=5, pady=5)
    quit_button.grid(row=4, column=2)

    root.mainloop()
