from sklearn.cluster import KMeans


class KMeansClustering:
    def __init__(self, n_clusters,
                 n_jobs=-2, verbose=1, random_state=2019):
        self.n_clusters = n_clusters
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

        self.kmeans = KMeans(n_clusters=self.n_clusters,
                             init='k-means++',
                             n_jobs=self.n_jobs, verbose=self.verbose,
                             random_state=self.random_state)

    def fit(self, input_data):
        self.kmeans.fit(input_data)

    def predict(self, input_data):
        return self.kmeans.predict(input_data)
