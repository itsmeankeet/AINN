### Lab Report: K-Means Clustering

**Title:** K-Means Clustering

**Objective:** To implement and understand the K-Means clustering algorithm using Python and scikit-learn.

**Theory:**  
K-Means clustering is an unsupervised machine learning algorithm used to partition data into K clusters. It works iteratively to assign data points to clusters based on their proximity to the cluster centroids. This method minimizes the sum of squared distances between data points and their respective cluster centers. K-Means is widely used due to its simplicity and efficiency.

**Steps:**
1. Generate 2D random data points using `numpy`.
2. Offset the data points into three distinct groups to form clusters.
3. Import the `KMeans` class from `scikit-learn`.
4. Initialize the K-Means model with 3 clusters.
5. Fit the model to the generated data and predict cluster assignments.
6. Use `matplotlib` to visualize the data points and cluster centers.

**Questions and Answers:**
1. **How does the choice of the number of clusters affect the results of K-Means?**  
   The number of clusters determines how the algorithm partitions the data. Too few clusters may oversimplify the data, while too many can overfit, capturing noise.

2. **What are the strengths and limitations of K-Means clustering?**  
   **Strengths:** Simple, fast, and efficient for large datasets.  
   **Limitations:** Sensitive to the initial choice of centroids and requires specifying the number of clusters.

3. **How can you determine the optimal number of clusters for a given dataset?**  
   Techniques like the Elbow Method and Silhouette Score can help determine the optimal number of clusters.

4. **In what real-world scenarios might K-Means clustering be useful?**  
   Customer segmentation, image compression, anomaly detection, and document clustering.

