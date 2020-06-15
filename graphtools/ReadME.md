## Graphtools
Aim of this part is to:
1. Collect the connectomes from all subjects in the form of graph <br>
        a. Currently in the form of a 2D array (number of subjects x (upper triangular dimension*3))
2. Perform classification/regression for all subjects

## File Descriptions
1. classification.py: uses different classification algorithms and makes a report with different metrics
2. regression.py: stacking regressor
3. big5personality.py: processes the labels provided by the unrestricted files in the HCP connectome. 
Makes profile reports based on pandas profiling module.
4. processing.py: Generate histogram plots for fscores (input to classification task) and
pearson correlation coefficients (for the regression task). The structure is as follows: <br>
    a. x axis presents the score: fscore/correlation 
    b. y axis presents the number of edges that give this particular fscore/correlation
5. node_edge_convert_solver.py: Try to input and work with different types of nodes and edges for input to
the java solver. 
6. metrics.py: fscore pandas implementation and miscellaneous metrics for further usage. 
7. reafiles.py: Read the labels for all the subjects who connectivity matrices have been computed using the 
preprocessing pipeline. 