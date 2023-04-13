package org.karazin;

import mpi.*;

import java.util.Arrays;

public class Main {

    public static void main(String[] args) throws MPIException {
        MPI.Init(args);

        int rank = MPI.COMM_WORLD.Rank();
        int size = MPI.COMM_WORLD.Size();
        int root = 0;

        // Graph representation (adjacency matrix)
        int[][] graph = {
                {0, 2, 0, 6, 0},
                {2, 0, 3, 8, 5},
                {0, 3, 0, 0, 7},
                {6, 8, 0, 0, 9},
                {0, 5, 7, 9, 0}
        };

        // Array to keep track of selected vertices
        boolean[] selected = new boolean[graph.length];

        // Initialize selected array
        Arrays.fill(selected, false);

        // Minimum key values to keep track of edges
        int[] key = new int[graph.length];

        // Initialize key values to positive infinity
        Arrays.fill(key, Integer.MAX_VALUE);

        // Mark root as selected and set its key value to 0
        selected[root] = true;
        key[root] = 0;

        // Number of vertices in each process
        int n = graph.length / size;

        // Array to store the edges of the MST
        int[][] mstEdges = new int[graph.length - 1][2];

        // Parallel Prim's algorithm
        for (int count = 0; count < graph.length - 1; count++) {
            // Find the minimum key value vertex not yet selected
            int minKey = Integer.MAX_VALUE;
            int minKeyVertex = -1;
            for (int i = rank * n; i < (rank + 1) * n; i++) {
                if (!selected[i] && key[i] < minKey) {
                    minKey = key[i];
                    minKeyVertex = i;
                }
            }

            // Allreduce to find the global minimum key value and its vertex
            int[] globalMinKey = new int[1];
            int[] globalMinKeyVertex = new int[1];
            MPI.COMM_WORLD.Allreduce(new int[]{minKey}, 0, globalMinKey, 0, 1, MPI.INT, MPI.MIN);
            MPI.COMM_WORLD.Allreduce(new int[]{minKeyVertex}, 0, globalMinKeyVertex, 0, 1, MPI.INT, MPI.MIN);

            minKey = globalMinKey[0];
            minKeyVertex = globalMinKeyVertex[0];

            // Mark the selected vertex
            selected[minKeyVertex] = true;

            // Update the key values of adjacent vertices
            for (int i = rank * n; i < (rank + 1) * n; i++) {
                if (graph[minKeyVertex][i] != 0 && !selected[i] && graph[minKeyVertex][i] < key[i]) {
                    key[i] = graph[minKeyVertex][i];
                }
            }

            // Gather the updated key values to all processes
            MPI.COMM_WORLD.Allgather(key, rank * n, n, MPI.INT, key, rank * n, n, MPI.INT);

            // Gather the selected vertices to all processes
            boolean[] globalSelected = new boolean[graph.length];
            MPI.COMM_WORLD.Allgather(selected, rank * n, n, MPI.BOOLEAN, globalSelected, rank * n, n, MPI.BOOLEAN);
            // Update the selected array with the gathered values
            System.arraycopy(globalSelected, 0, selected, 0, graph.length);

            // Find the minimum weight edge from selected vertices
            int[] minEdge = new int[3];
            minEdge[0] = Integer.MAX_VALUE;
            for (int u = 0; u < graph.length; u++) {
                if (selected[u]) {
                    for (int v = 0; v < graph.length; v++) {
                        if (!selected[v] && graph[u][v] != 0 && graph[u][v] < minEdge[0]) {
                            minEdge[0] = graph[u][v];
                            minEdge[1] = u;
                            minEdge[2] = v;
                        }
                    }
                }
            }

            // Allreduce to find the global minimum weight edge
            int[] globalMinEdge = new int[3];
            MPI.COMM_WORLD.Allreduce(minEdge, 0, globalMinEdge, 0, 3, MPI.INT, MPI.MIN);

            // Add the minimum weight edge to the MST
            mstEdges[count][0] = globalMinEdge[1];
            mstEdges[count][1] = globalMinEdge[2];
        }

        // Print the edges of the MST
        if (rank == 0) {
            System.out.println("Edges of Minimum Spanning Tree:");
            for (int i = 0; i < mstEdges.length; i++) {
                System.out.println(mstEdges[i][0] + " - " + mstEdges[i][1]);
            }
        }

        MPI.Finalize();
    }
}


/**
 * import mpi.*;
 *
 *
 *
 * This program uses MPJ to parallelize Prim's algorithm for finding the minimum spanning tree of a graph.
 * It uses a master-worker approach, where the master process (rank 0) coordinates the algorithm and
 * the worker processes (ranks 1 to size-1) perform the computation in parallel. The graph is represented
 * as an adjacency matrix, and the minimum spanning tree is also represented as an adjacency matrix.
 * The selected vertices are synchronized using MPJ's `Bcast` method, which broadcasts the selected array
 * from the master process to all other processes. The minimum weight edge is found in parallel by each process,
 * and the result is synchronized using `Bcast` as well. Finally, the minimum spanning tree is printed by the master process.

 * Note: This code assumes that you have set up MPJ correctly in your environment and have compiled
 * and run it with MPJ's `mpjrun` command. Also, this code assumes that the graph is connected and undirected,
 * and the edge weights are represented as integers in the `graph` array. You may need to modify the code
 * to suit your specific use case or input format.

 *
 * public class ParallelPrims {
 *
 *     private static final int MASTER_RANK = 0;
 *
 *     public static void main(String[] args) throws MPIException {
 *         MPI.Init(args);
 *
 *         int rank = MPI.COMM_WORLD.Rank();
 *         int size = MPI.COMM_WORLD.Size();
 *
 *         // Number of vertices in the graph
 *         int numVertices = 5;
 *
 *         // Graph represented as an adjacency matrix
 *         int[][] graph = {
 *             {0, 2, 0, 6, 0},
 *             {2, 0, 3, 8, 5},
 *             {0, 3, 0, 0, 7},
 *             {6, 8, 0, 0, 9},
 *             {0, 5, 7, 9, 0}
 *         };
 *
 *         // Minimum spanning tree represented as an adjacency matrix
 *         int[][] mst = new int[numVertices][numVertices];
 *
 *         // Array to keep track of selected vertices
 *         boolean[] selected = new boolean[numVertices];
 *
 *         // Initialize selected array
 *         if (rank == MASTER_RANK) {
 *             selected[0] = true;
 *             for (int i = 1; i < numVertices; i++) {
 *                 selected[i] = false;
 *             }
 *         }
 *
 *         // Synchronize selected array across all processes
 *         MPI.COMM_WORLD.Bcast(selected, 0, numVertices, MPI.BOOLEAN, MASTER_RANK);
 *
 *         int numSelected = 1;
 *         int selectedVertex = 0;
 *         int minWeight = Integer.MAX_VALUE;
 *
 *         // Loop until all vertices are selected
 *         while (numSelected < numVertices) {
 *             // Find the minimum weight edge that connects a selected vertex to an unselected vertex
 *             for (int i = 0; i < numVertices; i++) {
 *                 if (selected[i]) {
 *                     for (int j = 0; j < numVertices; j++) {
 *                         if (!selected[j] && graph[i][j] > 0 && graph[i][j] < minWeight) {
 *                             minWeight = graph[i][j];
 *                             selectedVertex = i;
 *                         }
 *                     }
 *                 }
 *             }
 *
 *             // Broadcast the selected vertex and its corresponding minimum weight to all processes
 *             int[] broadcastData = {selectedVertex, minWeight};
 *             MPI.COMM_WORLD.Bcast(broadcastData, 0, 2, MPI.INT, MASTER_RANK);
 *
 *             // Update the minimum spanning tree
 *             if (rank != selectedVertex && rank == MASTER_RANK) {
 *                 mst[selectedVertex][rank] = minWeight;
 *                 mst[rank][selectedVertex] = minWeight;
 *                 selected[selectedVertex] = true;
 *                 numSelected++;
 *             } else if (rank == selectedVertex) {
 *                 selected[rank] = true;
 *                 numSelected++;
 *             }
 *
 *             // Synchronize selected array across all processes
 *             MPI.COMM_WORLD.Bcast(selected, 0, numVertices, MPI.BOOLEAN, MASTER_RANK);
 *
 *             // Reset minWeight for the next iteration
 *             minWeight = Integer.MAX_VALUE;
 *         }
 *
 *         // Print the minimum spanning tree
 *         if (rank == MASTER_RANK) {
 *             System.out.println("Minimum Spanning Tree:");
 *             for (int i = 0; i < numVertices; i++) {
 *                 for (int j = 0; j < numVertices; j++) {
 *                     System.out.print(mst[i][j] + " ");
 *                 }
 *                 System.out.println();
 *             }
 *              }
 *
 *     MPI.Finalize();
 * }
 * }
 */