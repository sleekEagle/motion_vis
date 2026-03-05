# motion_vis
bugs:

1. insert and deletion test for structure is not performed when single structure
2. the pair analysis is buggy when there are 2 important pairs
3. there seems to be a bug in the motion filling funtions. there were two 8s in one example solution
4. insertion and deletion tests for important and unimportant frames are suspicious. 
5. i got 
d['pair_analysis']['clustered_ids']
{'1': [1, 1, 1], '5': [5, 5, 5, 5, 5], '8': [8], '9': [9, 9, 9, 9, 9, 9], '15': [15]}
This is wrong right ? there are not enough 1's
have to look in motion_analysis.py
6. insertion and deletion tests are similar to their random counterpart. maybe our frame importance methods are not that good. also the algorithm to sort the importance of the motion pairs.

