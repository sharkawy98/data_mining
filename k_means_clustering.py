import numpy as np

class KMeans:
    def __init__(self, file_name, n_clusters):
        self.labels = None
        self.attributes  = None
        self.centroids = None
        self.clusters = {}

        self.read_data(file_name)
        self.create_clusters(n_clusters)
        self.set_initial_centroids(n_clusters)
    #-------------------------------------------------------

    def read_data(self, file_name):
        data = np.loadtxt(file_name, dtype=str, delimiter=',', skiprows=1)
        self.labels = data[:, 0]
        self.attributes = data[:, 1:].astype('int32') 
    #-------------------------------------------------------
    
    def create_clusters(self, n_clusters):
        for n in range(n_clusters):
            self.clusters[n] = {'labels': [], 'attributes': []}
    #-------------------------------------------------------

    def set_initial_centroids(self, n_clusters):
        rand_idxs = np.random.choice(self.attributes.shape[0], n_clusters, replace=False)
        self.centroids = self.attributes[rand_idxs]
    #-------------------------------------------------------

    def get_distance(self, record, centroid):
        return np.sum(np.abs(record - centroid))
    #-------------------------------------------------------

    def get_new_centroids(self):
        new_centroids = np.zeros(self.centroids.shape)
        counter = 0
        for cluster in self.clusters.values():
            new_centroids[counter] = np.mean(cluster['attributes'], axis=0)
            counter += 1
        return new_centroids
    #-------------------------------------------------------

    def build_clusters(self):
        while True:
            for cluster in self.clusters.values():
                cluster['labels'] = []
                cluster['attributes'] = []
                
            for record_idx, record in enumerate(self.attributes):
                distances = []
                for centroid in self.centroids:
                    distances.append(self.get_distance(record, centroid))
                min_dist_cluster = np.argmin(distances)
                self.clusters[min_dist_cluster]['labels'].append(self.labels[record_idx])
                self.clusters[min_dist_cluster]['attributes'].append(record)

            new_centroids = self.get_new_centroids()
            
            if(np.array_equal(self.centroids, new_centroids)):
                break  # if no change in clusters

            self.centroids = new_centroids
   	#-------------------------------------------------------

    def print_results(self):
        for cluster in self.clusters:
            labels = self.clusters[cluster]['labels']
            print('Cluster', cluster, 'data:')
            print('-'*15)
            print(labels)
            print()
      

#-------------------------------------------------------
k = int(input('Enter your K: '))
k_means = KMeans('data/sales.csv', n_clusters=k)
k_means.build_clusters()
k_means.print_results()
