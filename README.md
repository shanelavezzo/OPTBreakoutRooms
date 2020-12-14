Problem Statement: Consider the following NP-Hard problem. Place n students 
into Zoom breakout rooms. For each pair of students i and j, there is one value h<sub>i,j</sub> quantifying 
how much happiness these two students give each other and one value s<sub>i,
j</sub> quantifying how much stress
they give each other. The total happiness value of a room H<sub>room</sub> is 
the sum of the happiness values h<sub>i,j</sub> of every student
pair in that room, and the total stress value of a room S<sub>room</sub> is the 
sum of the stress values s<sub>i,
j</sub> of every student pair in
that room. The total stress must be low enough so that it does not 
surpass S<sub>max</sub> / k in each room, where k is the number of breakout 
rooms you choose to open. The goal is to maximize total happiness across all 
rooms, while keeping the total stress below the threshold in each room.

Input Parameters:\
• n = Number of students in the class\
• h<sub>i,j</sub> = Happiness student i and j give each other\
• s<sub>i,j</sub> = Stress student i and j induce on each other\
• S<sub>max</sub>  = Maximum total stress across all breakout rooms

Main code can be found in solver.py.