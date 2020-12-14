import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_solution, calculate_happiness, \
    calculate_stress_for_room, calculate_happiness_for_room
import sys
import random
import glob
from random import sample
from os.path import basename, normpath
from math import ceil, exp

def solve(G, s):
    """
    Args:
        G: networkx.Graph
        s: stress_budget
    Returns:
        D: Dictionary mapping for student to breakout room r e.g.
        {0:2, 1:0, 2:1, 3:2}
        k: Number of breakout rooms
    Notes:
        This function was designed to be a Sequential Variable Neighborhood
        Descent approximation algorithm.
        This is an NP-hard problem which makes finding a correct solution
        computationally difficult.
        However, by employing certain measures, this algorithm can
        ensure a solution T that approximates the optimal solution T'.
    """

    initial_solution, k = create_initial_mapping(G, s)
    D, k = seqVND(G, s, initial_solution, k)

    return D, k


def create_initial_mapping(G, s):
    """
    Args:
        G: networkx.Graph
        s: stress_budget
    Returns:
        D: Dictionary mapping for student to breakout room r e.g.
        {0:2, 1:0, 2:1, 3:2}
        k: Number of breakout rooms
    Notes:
        This function greedily chooses an initial solution to the problem by
        placing the lowest stress students with each other.
    """

    num_students = len(G.nodes())
    D = {}
    k = num_students

    # Sort all edges by increasing stress value.
    edges_by_stress= sorted(G.edges(data=True), key=lambda t: t[2].get(
        'stress', 1))

    # Place pairs of students into their own room if they have a low enough
    # stress value.
    room_counter = 0
    for edge in edges_by_stress:
        if edge[0] not in D and edge[1] not in D:
            if edge[2].get('stress') <= (s / num_students):
                D[edge[0]] = room_counter
                D[edge[1]] = room_counter
                room_counter += 1

    # Loop through the students and place them into their own breakout room
    # if they weren't placed in one previously.
    for i in range(num_students):
        if i not in D:
            D[i] = room_counter
            room_counter += 1

    return D, k


def seqVND(G, s, D, k):
    """
    Args:
        G: networkx.Graph
        s: stress_budget
        D: Dictionary mapping for student to breakout room r e.g.
        {0:2, 1:0, 2:1, 3:2}
        k: Number of breakout rooms
    Returns:
        D: Dictionary mapping for student to breakout room r e.g.
        {0:2, 1:0, 2:1, 3:2}
        k: Number of breakout rooms
    Notes:
        This function perfroms Sequential Variable Neighborhood Descent
        using different neighborhoods. Currently there are three neighborhods.
        The first neighborhood corresponds to all valid single student room
        changes. The second neighborhood corresponds to all valid pairwise
        student room swaps. The third neighborhood corresponds to all valid
        concurrent two person moves.
    """

    # Store the current happiness rating
    curr_happiness_rating = calculate_happiness(D, G)

    # Initialize our neighborhood variables
    total_neighborhoods = 3
    current_neighborhood = 1

    # Perform Sequential Varaible Neighborhood Descent
    while current_neighborhood <= total_neighborhoods:
        if current_neighborhood == 1:
            potential_changes = move_neighborhood(G, s, D, k)
            best_neighborhood_change, best_neighborhood_happiness = local_search_move(D, G, potential_changes)
        elif current_neighborhood == 2:
            potential_changes = swap_neighborhood(G, s, D, k)
            best_neighborhood_change, best_neighborhood_happiness = local_search_swap(D, G, potential_changes)
        else:
            potential_changes = move2_neighborhood(G, s, D, k)
            best_neighborhood_change, best_neighborhood_happiness = local_search_move2(D, G, potential_changes)
        if best_neighborhood_happiness > curr_happiness_rating:
            D = best_neighborhood_change
            k = len(set(D.values()))
            curr_happiness_rating = calculate_happiness(D, G)
            current_neighborhood = 1
        else:
            current_neighborhood += 1

    return D, k


def local_search_move(D, G, potential_changes):
    """
    Args:
        D: Dictionary mapping for student to breakout room r e.g.
        {0:2, 1:0, 2:1, 3:2}
        G: networkx.Graph
        potential_changes: 2D List of all potential moves.
    Returns:
        best_change: Dictionary mapping for student to breakout room r e.g.
        {0:2, 1:0, 2:1, 3:2}
        best_happiness: Correspond happiness value for best_change
    Notes:
        This function will perform a local search of all potential moves,
        applying the optimal move, and returning the correspond mapping
        and happiness rating.
    """

    room_to_s = {}
    for m, v in D.items():
        room_to_s.setdefault(v, []).append(m)

    best_change = D.copy()
    best_happiness = 0
    best_move = []

    for change in potential_changes:
        move(change[0], change[1], best_change)
        change_happiness = calculate_happiness(best_change, G)
        if change_happiness > best_happiness:
            best_happiness = change_happiness
            best_move = change
        move(change[0], D[change[0]], best_change)
    if len(best_move) > 0:
        move(best_move[0], best_move[1], best_change)

    return best_change, best_happiness


def local_search_swap(D, G, potential_changes):
    """
    Args:
        D: Dictionary mapping for student to breakout room r e.g.
        {0:2, 1:0, 2:1, 3:2}
        G: networkx.Graph
        potential_changes: 2D List of all potential moves.
    Returns:
        best_change: Dictionary mapping for student to breakout room r e.g.
        {0:2, 1:0, 2:1, 3:2}
        best_happiness: Correspond happiness value for best_change
    Notes:
        This function will perform a local search of all potential swaps,
        applying the optimal swap, and returning the correspond mapping
        and happiness rating.
    """

    room_to_s = {}
    for m, v in D.items():
        room_to_s.setdefault(v, []).append(m)

    best_change = D.copy()
    best_happiness = 0
    best_swap = []

    for change in potential_changes:
        student1, student2 = change[0], change[1]
        swap(student1, student2, best_change)
        change_happiness = calculate_happiness(best_change, G)
        if change_happiness > best_happiness:
            best_happiness = change_happiness
            best_swap = change
        swap(student1, student2, best_change)
    if len(best_swap) > 0:
        swap(best_swap[0], best_swap[1], best_change)

    return best_change, best_happiness

def local_search_move2(D, G, potential_changes):
    """
    Args:
        D: Dictionary mapping for student to breakout room r e.g.
        {0:2, 1:0, 2:1, 3:2}
        G: networkx.Graph
        potential_changes: 2D List of all potential moves.
    Returns:
        best_change: Dictionary mapping for student to breakout room r e.g.
        {0:2, 1:0, 2:1, 3:2}
        best_happiness: Correspond happiness value for best_change
    Notes:
        This function will perform a local search of all potential two
        student moves, applying the optimal move, and returning the
        correspond mapping and happiness rating.
    """

    room_to_s = {}
    for m, v in D.items():
        room_to_s.setdefault(v, []).append(m)

    best_change = D.copy()
    best_happiness = 0
    best_move = []

    for change in potential_changes:
        move2(change[0][0], change[1][0], change[0][1], change[1][1],
              best_change)
        change_happiness = calculate_happiness(best_change, G)
        if change_happiness > best_happiness:
            best_happiness = change_happiness
            best_move = change
        move2(change[0][0], change[1][0], D[change[0][0]], D[change[1][0]],
              best_change)
    if len(best_move) > 0:
        move2(change[0][0], change[1][0], change[0][1], change[1][1],
              best_change)

    return best_change, best_happiness

def is_valid_swap(student1, student2, D, rooms_to_s, k, G, s):

    """
    Args:
        student1: the first student
        student2: the second student
        D: Dictionary mapping for student to breakout room r e.g.
        {0:2, 1:0, 2:1, 3:2}
        rooms_to_s: Dictionary mapping breakout room to a list of
        students currently in that room e.g. {0:[1, 2], 1:[0, 3]}
        k: number of breakout rooms
        G: networkx.Graph
        s: stress budget
    Returns:
        bool: whether we can swap the students' breakout rooms
    Notes:
        This function checks if swapping two students' rooms follows the
        stress guidelines
    """

    # Store the students' current rooms.
    student_one_room, student_two_room = D[student1], D[student2]

    # Check if the students are already in the same room.
    if student_one_room == student_two_room:
        return False

    # Check if both students are currently alone (thus a swap will not change
    # the happiness or stress and is irrelevant).
    if len(rooms_to_s[student_one_room]) == 0 and len(rooms_to_s[
                                                          student_two_room]) == 0:
        return False


    # Move the first student into a copy of the second student's room,
    # excluding the second student.
    student_one_moved = rooms_to_s[student_two_room].copy()
    student_one_moved.remove(student2)
    student_one_moved.append(student1)

    # Move the second student into a copy of the first student's room,
    # excluding the first student.
    student_two_moved = rooms_to_s[student_one_room].copy()
    student_two_moved.remove(student1)
    student_two_moved.append(student2)

    # Store the stress guideline.
    max_stress = s/k

    # Check if swapping the students was valid.
    if calculate_stress_for_room(student_one_moved, G) > max_stress:
        return False
    if calculate_stress_for_room(student_two_moved, G) > max_stress:
        return False

    return True



def is_valid_move(student, room, D, rooms_to_s, k, G, s):
    """
    Args:
        student: the student to be moved
        room: the room for the student to move to
        D: Dictionary mapping for student to breakout room r e.g.
        {0:2, 1:0, 2:1, 3:2}
        rooms_to_s: Dictionary mapping breakout room to a list of
        students currently in that room e.g. {0:[1, 2], 1:[0, 3]}
        k: number of breakout rooms
        G: networkx.Graph
        s: stress budget
    Returns:
        bool: whether we can move the student to the room
    Notes:
        This function checks if moving a student to a room follows the stress
        guidleines.
    """

    student_room = D[student]

    # Check if we are trying to move the student to his own room.
    if student_room == room:
        return False

    # Check if the student is alone in his current room.
    alone = False
    if len(rooms_to_s[student_room]) == 1:
        alone = True

    # Move the student to a copy of the new room.
    student_moved = rooms_to_s[room].copy()
    student_moved.append(student)

    # Check if moving the student to the room was valid
    if alone:
        max_stress = s / (k - 1)
        if calculate_stress_for_room(student_moved, G) > max_stress:
            return False
    else:
        max_stress = s / k
        if calculate_stress_for_room(student_moved, G) > max_stress:
            return False

    return True


def is_valid_move2(student1, student2, room1, room2, D, rooms_to_s, k, G, s):
    """
    Args:
        student1: the first student to be moved
        student2: the second student to be moved
        room1: the room for the first student to move to
        room2: the room for the second student to move to
        D: Dictionary mapping for student to breakout room r e.g.
        {0:2, 1:0, 2:1, 3:2}
        rooms_to_s: Dictionary mapping breakout room to a list of
        students currently in that room e.g. {0:[1, 2], 1:[0, 3]}
        k: number of breakout rooms
        G: networkx.Graph
        s: stress budget
    Returns:
        bool: whether we can move the students to the rooms
    Notes:
        This function checks if moving twp student to rooms follows the stress
        guidleines.
    """

    student1_room = D[student1]
    student2_room = D[student2]

    # Check if we are trying to move student1 to his own room.
    if student1_room == room1:
        return False

    # Check if we are trying to move student2 to his own room.
    if student2_room == room2:
        return False

    # Check if student1 is alone in his current room.
    if len(rooms_to_s[student1_room]) != 1:
        return False

    # Check if student2 is alone in his current room.
    if len(rooms_to_s[student2_room]) != 1:
        return False

    # Check if we are moving them to the same room.
    same_room = False
    if room1 == room2:
        same_room = True

    if same_room:
        students_moved = rooms_to_s[room1].copy()
        students_moved.extend([student1, student2])
        max_stress = s / (k - 2)
        if calculate_stress_for_room(students_moved, G) > max_stress:
            return False
        return True
    else:
        # Move student1 to a copy of their new room.
        student1_moved = rooms_to_s[room1].copy()
        student1_moved.append(student1)

        # Move student2 to a copy of their new room.
        student2_moved = rooms_to_s[room2].copy()
        student2_moved.append(student2)

        # Check if moving the students to their new rooms was valid
        max_stress = s / (k - 2)
        if calculate_stress_for_room(student1_moved, G) > max_stress:
            return False
        if calculate_stress_for_room(student2_moved, G) > max_stress:
            return False

        return True


def swap_neighborhood(G, s, D, k):
    """
    Args:
        G: networkx.Graph
        s: stress_budget
        D: Dictionary mapping for student to breakout room r e.g.
        {0:2, 1:0, 2:1, 3:2}
        k: Number of breakout rooms
    Returns:
        valid_swaps: A 2D list corresponding to all valid student room swaps.
        Each element of the list is a two element list. The
        first element is a student and the second element is another student
        whose room they can swap with. e.g. [[0, 1], [1,3]]
    Notes:
        This function determines all valid pairwise swaps given a mapping of
        student to breakout room.
    """

    num_students = len(G.nodes())
    valid_swaps = []

    # Create a dictionary which maps breakout room to a list of students who are
    # currently in that room.
    room_to_s = {}
    for m, v in D.items():
        room_to_s.setdefault(v, []).append(m)

    # Loop through all the students and check which other student's they can
    # potentially swap rooms with.
    for i in range(num_students):
        j = i + 1
        for j in range(num_students):
            if is_valid_swap(i, j, D, room_to_s, k, G, s):
                valid_swaps.append([i, j])

    return valid_swaps



def move_neighborhood(G, s, D, k):
    """
    Args:
        G: networkx.Graph
        s: stress_budget
        D: Dictionary mapping for student to breakout room r e.g.
        {0:2, 1:0, 2:1, 3:2}
        k: Number of breakout rooms
    Returns:
        valid_moves: A 2D list corresponding to all valid moves from the given
        mapping D. Each element of the list is a two element list. The
        first element is a student and the second element is a room they
        can move to. e.g. [[0, 1], [1,3]]
    Notes:
        This function determines all valid single student room changes given
        a mapping of students to breakout room.
    """

    num_students = len(G.nodes())
    valid_moves = []
    rooms_in_use = set(D.values())


    # Create a dictionary which maps breakout room to a list of students who are
    # currently in that room.
    room_to_s = {}
    for m, v in D.items():
        room_to_s.setdefault(v, []).append(m)


    # Loop through all the students and check which rooms they can
    # potentially move to.
    for student in range(num_students):
        for room in rooms_in_use:
            if is_valid_move(student, room, D, room_to_s, k, G, s):
                valid_moves.append([student, room])

    return valid_moves


def move2_neighborhood(G, s, D, k):
    """
    Args:
        G: networkx.Graph
        s: stress_budget
        D: Dictionary mapping for student to breakout room r e.g.
        {0:2, 1:0, 2:1, 3:2}
        k: Number of breakout rooms
    Returns:
        valid_moves: A 2D list corresponding to all valid moves from the given
        mapping D. Each element of the list is a two element list. The
        first element is a student and the second element is a room they
        can move to. e.g. [[0, 1], [1,3]]
    Notes:
        This function determines all valid two student room changes given
        a mapping of students to breakout room.
    """

    num_students = len(G.nodes())
    valid_moves = []
    rooms_in_use = set(D.values())


    # Create a dictionary which maps breakout room to a list of students who are
    # currently in that room.
    room_to_s = {}
    for m, v in D.items():
        room_to_s.setdefault(v, []).append(m)


    # Loop through all the students and check which rooms they can
    # potentially move to.
    for student1 in range(num_students):
        student2 = student1 + 1
        for student2 in range(num_students):
            for room1 in rooms_in_use:
                for room2 in rooms_in_use:
                    if is_valid_move2(student1, student2, room1, room2, D,
                                      room_to_s, k, G,
                                      s):
                        valid_moves.append([[student1, room1], [student2,
                                                                room2]])

    return valid_moves


def swap(student1, student2, D):

    """
    Args:
        student1: first student to be swapped
        student2: second student to be swapped
        D: Dictionary mapping for student to breakout room r e.g.
        {0:2, 1:0, 2:1, 3:2}
    Returns:
        Nothing
    Notes:
        This function moves the first student into the second student's
        room and the second student into the first student's room.
    """

    D[student2], D[student1] = D[student2], D[student1]

    return



def move(student, room, D):

    """
    Args:
        student: the student to be moved
        room: the room the student will be moved into
        D: Dictionary mapping for student to breakout room r e.g.
        {0:2, 1:0, 2:1, 3:2}
    Returns:
        Nothing
    Notes:
        This function moves a student into a certain room.
    """

    D[student] = room

    return

def move2(student1, student2, room1, room2, D):

    """
    Args:
        student1: the first student to be moved
        student2: the second student to be moved
        room1: the room the first student will be moved into
        room2: the room the first student will be moved into
        D: Dictionary mapping for student to breakout room r e.g.
        {0:2, 1:0, 2:1, 3:2}
    Returns:
        Nothing
    Notes:
        This function moves two students into rooms.
    """

    D[student1] = room1
    D[student2] = room2

    return


# Here's an example of how to run the solver.

# Usage: python3 solver.py test.in

if __name__ == '__main__':
     assert len(sys.argv) == 2
     path = sys.argv[1]
     G, s = read_input_file(path)
     D, k = solve(G, s)
     assert is_valid_solution(D, G, s, k)
     print(D)
     print("Total Happiness: {}".format(calculate_happiness(D, G)))
     write_output_file(D, 'outputs/test.out')


# For testing a folder of inputs to create a folder of outputs, you can use glob
#if __name__ == '__main__':
#     inputs = glob.glob('inputs/*')
#     for input_path in inputs:
#         output_path = 'outputs/' + basename(normpath(input_path))[:-3] + '.out'
#         print(input_path)
#         G, s = read_input_file(input_path, 100)
#         D, k = solve(G, s)
#         assert is_valid_solution(D, G, s, k)
#         happiness = calculate_happiness(D, G)
#         write_output_file(D, output_path)



