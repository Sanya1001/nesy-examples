nn(property_net1,[X],Y,[0, 1]) :: is_condition1_nn(X,Y).
nn(property_net2,[X],Y,[0, 1]) :: is_condition2_nn(X,Y).

nn(relation_net1,[X, Y],Z,[0, 1]) :: is_relation1_nn(X,Y,Z).
nn(relation_net2,[X, Y],Z,[0, 1]) :: is_relation2_nn(X,Y,Z).

andL(X,  Y,  Z,  K) :- K is X * Y * Z.

inference_1_1_1(X, Y, Z) :- is_condition1_nn(X, Z1), is_relation1_nn(X, Y, Z2), is_condition1_nn(Y, Z3), andL(Z1, Z2, Z3, Z).
inference_1_1_2(X, Y, Z) :- is_condition1_nn(X, Z1), is_relation1_nn(X, Y, Z2), is_condition2_nn(Y, Z3), andL(Z1, Z2, Z3, Z).
inference_1_2_1(X, Y, Z) :- is_condition1_nn(X, Z1), is_relation2_nn(X, Y, Z2), is_condition1_nn(Y, Z3), andL(Z1, Z2, Z3, Z).
inference_1_2_2(X, Y, Z) :- is_condition1_nn(X, Z1), is_relation2_nn(X, Y, Z2), is_condition2_nn(Y, Z3), andL(Z1, Z2, Z3, Z).

inference_2_1_1(X, Y, Z) :- is_condition2_nn(X, Z1), is_relation1_nn(X, Y, Z2), is_condition1_nn(Y, Z3), andL(Z1, Z2, Z3, Z).
inference_2_1_2(X, Y, Z) :- is_condition2_nn(X, Z1), is_relation1_nn(X, Y, Z2), is_condition2_nn(Y, Z3), andL(Z1, Z2, Z3, Z).
inference_2_2_1(X, Y, Z) :- is_condition2_nn(X, Z1), is_relation2_nn(X, Y, Z2), is_condition1_nn(Y, Z3), andL(Z1, Z2, Z3, Z).
inference_2_2_2(X, Y, Z) :- is_condition2_nn(X, Z1), is_relation2_nn(X, Y, Z2), is_condition2_nn(Y, Z3), andL(Z1, Z2, Z3, Z).



