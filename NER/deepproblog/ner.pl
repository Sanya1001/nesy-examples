nn(person_net1, [PersonEmb], P, [0,1]) :: is_real_person1(PersonEmb, P).
nn(person_net2, [PersonEmb], P, [0,1]) :: is_real_person2(PersonEmb, P).
nn(person_net3, [PersonEmb], P, [0,1]) :: is_real_person3(PersonEmb, P).

nn(work_net1, [PersonEmb, LocEmb], P, [0, 1]) :: work_in1(PersonEmb, LocEmb, P).
nn(work_net2, [PersonEmb, LocEmb], P, [0, 1]) :: work_in2(PersonEmb, LocEmb, P).
nn(work_net3, [PersonEmb, LocEmb], P, [0, 1]) :: work_in3(PersonEmb, LocEmb, P).

% The following predicates are used to define the constraints
% and are not part of the neural network
and(X, Y, Z) :- Z is X * Y.
or(X, Y, Z) :- Z is X + Y.

constraint1(P1, P2, L1, L2, P3, L3, Z) :- is_real_person1(P1, Z1), work_in1(P1, L1, W1), 
    is_real_person2(P2, Z2), work_in2(P2, L2, W2), and(Z1, W1, T1), and(Z2, W2, T2), and(T1, T2, Z).

constraint2(P1, P2, L1, L2, P3, L3, Z) :- is_real_person2(P2, Z2), work_in2(P2, L2, W2),
    is_real_person3(P3, Z3), work_in3(P3, L3, W3), and(Z2, W2, T2), and(Z3, W3, T3), or(T2, T3, Z).

check(P1, P2, P3, L1, L2, L3, F) :- constraint1(P1, P2, L1, L2, P3, L3, Z1), constraint2(P1, P2, L1, L2, P3, L3, Z2), and(Z1, Z2, F).