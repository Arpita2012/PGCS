import os 
import sys
import pickle
import numpy as np
import torch 
from  iqa_architecture.liqe  import *
from tqdm import tqdm
import math
from sklearn.manifold import TSNE
import multiprocessing
from sklearn.cluster import KMeans




class CoveragePrecedencePartitioning:

    def __init__(self, n_partitions=8, random_state=None):
        self.n_partitions = n_partitions
        self.random_state = random_state
        self.labels_ = None

    def fit(self, X ,random_state=None ):
        # Dummy fit method that assigns random partition labels
        np.random.seed(random_state)
        # self.labels_ = np.random.randint(0, self.n_partitions, size=X.shape[0])

        N = X.shape[0]

        # idx = list(range(X.shape[0]))

        # print(idx)


        squared_norms = torch.sum(X ** 2, dim=1).reshape(-1, 1)  # shape (10000, 1)

        # Step 2: Use broadcasting to compute squared Euclidean distances
        # d(x, y)^2 = ||x||^2 + ||y||^2 - 2 * x @ y
        distances_squared = squared_norms + squared_norms.T - 2 * X @ X.T

        # Step 3: Take the square root to get Euclidean distances
        distances = torch.sqrt(distances_squared)

        # Replace any potential negative values with 0 (due to numerical precision issues)
        distances = torch.clamp(distances, min=0)


        #replace all diagonal elements with 0
        distances.fill_diagonal_(0)

        
        # Now `distances` contains the Euclidean distances between all rows
        print("Euclidean : " , distances.shape)  # Output shape should be (10000, 10000)
        print(distances[0:5, 0:5])

        # for each row in distances  --- replace values in following way
        # replaces values with digit 1 to n_partitions
        # such that first min (N/n_partitions) values are replaced with 1, next min (N/n_partitions) values are replaced with 2 and so on
        # where N is the number of elements in the row
        # and n_partitions is the number of partitions
        
        partition_size = N // self.n_partitions

        for i in range(N):
            sorted_indices = torch.argsort(distances[i])
            for j in range(self.n_partitions):
                start_idx = j * partition_size
                end_idx = (j + 1) * partition_size if j != self.n_partitions - 1 else N
                distances[i, sorted_indices[start_idx:end_idx]] = j + 1

        print("Euclidean : " , distances.shape)  # Output shape should be (10000, 10000)
        print(distances[0:5, 0:5])
        
        #sum all the rows  and save in a torch tensor
        sum_distances = torch.sum(distances, dim=0)
        print("Sum distances : " , sum_distances.shape)  # Output shape should be (10000, 10000)
        print(sum_distances[0:15])

        # there are two ways from here
        # way 1 we spread the high coverage points across all the paritions
        #way 2 we just select wweighted selection from each paritiion -- summ all coverage and then normalize to get weights

        #trying way 1 first
        # assign partion ids [1 to K ] to each element based on the sum of distances
        # starting from minimum element to maximum element -- assign 1 to n_partitions
        # where n_partitions is the number of partitions
        sorted_indices = torch.argsort(sum_distances)
        self.labels_ = np.zeros(N, dtype=int)
       
        print(sorted_indices[0:15])
        print(sum_distances[sorted_indices[0:15]])
        i=0;
        for p in range (0,partition_size):
            
            self.labels_[sorted_indices[i:i+self.n_partitions]] = np.arange(0,self.n_partitions)
            i+=self.n_partitions
        ## remaining elements
        self.labels_[sorted_indices[i:]] = np.arange(0,self.n_partitions) [:(N-i)]


        print("Partitioning done")
        print(self.labels_.shape)
        print(self.labels_[sorted_indices[-15:]])




    def fit_predict(self, X):
        self.fit(X)
        return self.labels_



class PGCS:
   
    def __init__(self, imageEncoder=None,  projectionOperator=None,m=None, partioningAlgo=None, K=None):
        """
        Initializes the class with the given projection operator, parameter m, and partitioning algorithm.
        Args:
            imageEncoder: Image Encoder to be used to extract image embeddings
            projectionOperator: An instance of the Non Linear  projection operator to be used.
            m: A parameter used in the projection operator.
            partioningAlgo: The partitioning algorithm to be used.
            K: The number of partitions to be used.
        """
        self.projectionOperator = projectionOperator
        self.m = m
        self.partioningAlgo = partioningAlgo
        self.K = K
        self.imageEncoder = imageEncoder

        if self.projectionOperator=="TSNE":
            self.projectionOperator = TSNE(n_components=self.m)
        else:
            raise ValueError("Undefined Projection Operator")
        
        if self.partioningAlgo=="Kmeans":
            self.partioningAlgoObj = KMeans(n_clusters=self.K, random_state=0)
        
        elif self.partioningAlgo=="RandomPartitioning":
            self.partioningAlgoObj = RandomPartitioning(n_partitions=self.K, random_state=0)

        elif self.partioningAlgo=="CoveragePrecedencePartitioning":
            self.partioningAlgoObj = CoveragePrecedencePartitioning(n_partitions=self.K, random_state=0)

        # elif self.partioningAlgo=="WeightedCoveragePrecedencePartitioning":
        #     self.partioningAlgoObj = WeightedCoveragePrecedencePartitioning(n_partitions=self.K, random_state=0)
        else:
            
            raise ValueError("Undefined partitioning Algorithm")

        if self.imageEncoder=="liqe":
            self.imageEncoderObj = LIQEExtractor()
        else:
            raise ValueError("Undefined Image Encoder ")




    def count_partition_elements(self,partition_labels_):
        # Count the number of elements in each partition
        unique_partitions, counts = np.unique(partition_labels_, return_counts=True)
        partition_counts = dict(zip(unique_partitions.tolist(), counts.tolist()))
        return partition_counts
  

    def get_median(self,features, targets):
        # get the median feature vector of each class  IDEALLY This function is getting class center
    
        num_classes = len(np.unique(targets, axis=0))
     
        prot = np.zeros((num_classes, features.shape[-1]), dtype=features.dtype)
      
        for i in range(num_classes):
            cc =  np.median(features[(targets == i).nonzero(), :].squeeze(), axis=0, keepdims=False)
            prot[i] = cc
        return prot


    def get_distance(self,features, labels):
        
        prots = self.get_median(features, labels)
        prots_for_each_example = np.zeros(shape=(features.shape[0], prots.shape[-1]))
        
        num_classes = len(np.unique(labels))
        for i in range(num_classes):
            prots_for_each_example[(labels==i).nonzero()[0], :] = prots[i]
        distance = np.linalg.norm(features - prots_for_each_example, axis=1)
        
        return distance

    def get_features_liqe( self, trainloader, config):
       
        
        self.imageEncoderObj = self.imageEncoderObj.to(config.device)

        targets, features = [], []
        # i=0;
        for data in tqdm(trainloader):

            img = data['d_img_org'].cuda()
            target = data['score']

            targets.extend(target.numpy().tolist())
            img = img.to(config.device)
            feature = self.imageEncoderObj(img).detach().cpu().numpy()
            
            
            features.extend([feature[i] for i in range(feature.shape[0])])


        
        features = np.array(features)
        targets = np.array(targets)
        
        #converting to classification problem
        targets = np.floor(targets * 10).astype(int)
        targets=targets.squeeze(-1) #removing all singleton elements
        print("Features and targets obtained")
        print(features.shape)
        print(targets.shape)
        return features, targets  


    def select_idx( self, distance,config,targets, partition_ids):
        
        prune_ids = []
        selected_ids = []

        unique_labels = np.unique(partition_ids)  #100

        
        per_bin_selection_num_list=[]
        num_cls = len(unique_labels)
        for C in unique_labels:
            # print(f"partition {C} has {np.sum(partition_ids==C)} samples")


            Class_C_ids = np.where(partition_ids == C)[0]
            # print(Class_C_ids)
            distance_C = distance[Class_C_ids]

            sorted_idx = distance_C.argsort()

            len_dist_C = len(distance_C)
         
            per_bin_selection_num_list.append(math.ceil(len_dist_C * (config.coreset_size/len(distance))))
        # print("per_bin_selection_num", per_bin_selection_num_list)




        per_bin_selection_num_list = np.array(per_bin_selection_num_list)


        
        # print("per_bin_selection_num", per_bin_selection_num_list)
        # print("sum", per_bin_selection_num_list.sum())

        

        # Assuming per_bin_selection_num_list is a numpy array
        extra_samples = per_bin_selection_num_list.sum() - config.coreset_size

        # Calculate the total number of elements in all bins
        total_elements = per_bin_selection_num_list.sum()

        # Calculate the proportion of elements in each bin
        proportions = per_bin_selection_num_list / total_elements

        # Calculate the number of extra samples to prune from each bin
        extra_samples_to_prune = np.floor(proportions * extra_samples).astype(int)

        # Adjust the bins by pruning the extra samples
        for i in range(len(per_bin_selection_num_list)):
            per_bin_selection_num_list[i] -= extra_samples_to_prune[i]
            extra_samples -= extra_samples_to_prune[i]

        # If there are still extra samples to prune, distribute them one by one
        while extra_samples > 0:
            max_idx = np.argmax(per_bin_selection_num_list)
            per_bin_selection_num_list[max_idx] -= 1
            extra_samples -= 1

        # print("Adjusted per_bin_selection_num", per_bin_selection_num_list)
        # print("Adjusted sum", per_bin_selection_num_list.sum())

        if per_bin_selection_num_list.sum() != config.coreset_size:
            print("Error: Coreset size is not equal to the sum of per_bin_selection_num_list")
           
            sys.exit(0)


    

    
        for C in unique_labels:
            # print(f"Class {C} has {np.sum(partition_ids==C)} samples")

            Class_C_ids = np.where(partition_ids == C)[0]
            # print(Class_C_ids)
            distance_C = distance[Class_C_ids]

            sorted_idx = distance_C.argsort()

            len_dist_C = len(distance_C)
          
            per_bin_selection_num =   per_bin_selection_num_list[C] 
            # print("per_bin_selection_num", per_bin_selection_num)

            median = len_dist_C // 2
            low_idx = median - (per_bin_selection_num // 2)
            high_idx = low_idx + per_bin_selection_num 

        

            # Get original indices from the sorted indices
            original_indices = Class_C_ids[sorted_idx]
            # print("Original indices sorted by distance:", original_indices)


            # Prune ids (indices to be pruned)
            prune_ids.extend(original_indices[:low_idx])
            prune_ids.extend(original_indices[high_idx:])

            # Selected ids (indices to be selected as coreset)
            selected = original_indices[low_idx:high_idx]
            selected_ids.extend(selected)

        return  np.array(selected_ids)



    def get_coreset(self,train_loader, config):

    
        #features/dataset/imageEncoder
        feature_save_folder = os.path.join(config.feature_save_path, config.dataset_name, config.imageEncoder);
        if not os.path.exists(feature_save_folder):
            os.makedirs(feature_save_folder)
        

        if not os.path.isfile(os.path.join(feature_save_folder , config.feature_save_file)):
            print("Features file does not exist..extarcting features")
            features, targets = self.get_features_liqe( train_loader , config)
            with open(os.path.join(feature_save_folder , config.feature_save_file), "wb") as file:
                pickle.dump((features, targets), file)
        
        else:
            print("Features file already exist")
            with open(os.path.join(feature_save_folder , config.feature_save_file), "rb") as f:
                features, targets = pickle.load(f)
        
        # print("Features shape" , features.shape)
        # print("targets shape" , targets.shape)
        print("Features and targets obtained")

        
        #############################################################################################################
      

        # Save partition IDs to a file
        partition_save_path = os.path.join(config.partition_save_path, config.dataset_name, config.partioningAlgo+ str(config.partion_count) ,config.projectionOperator + str(config.prjection_hyper))
        if not os.path.exists(partition_save_path):
            os.makedirs(partition_save_path)

        if not os.path.isfile(os.path.join(partition_save_path , config.partition_save_file)):
            # print("partition file does not exist..extarcting partitions")

            print("Applying Projection Operator....")
            features_after_projection = self.projectionOperator.fit_transform(features)
            features_tensor = torch.from_numpy(features_after_projection)

            print("Applying Partitioning Algorithm....")
            self.partioningAlgoObj.fit(features_tensor)
            partition_ids = self.partioningAlgoObj.labels_

            with open(os.path.join(partition_save_path, config.partition_save_file), "wb") as file:
                pickle.dump(partition_ids, file)

    
        else:
            print("partitioning file already exist")
            with open(os.path.join(partition_save_path , config.partition_save_file), "rb") as f:
                partition_ids = pickle.load(f)

        # print("partition labels:",partition_ids)
        # print("partition labels shape:",partition_ids.shape)

        # partition_counts = self.count_partition_elements(partition_ids)
        # print("Number of elements in each partition:", partition_counts)

        #############################################################################################################

     
        distance_file_path = os.path.join(config.index_folder, config.coreset_method, config.dataset_name, "distances")

        if not os.path.exists(distance_file_path):
            os.makedirs(distance_file_path)
        
        name_distance_file_con =  os.path.join(distance_file_path , config.partioningAlgo + "_"+ str(config.partion_count) +"_" + config.projectionOperator + "_" +str(config.prjection_hyper) +  config.distance_file_name  ) 

        if not os.path.isfile(name_distance_file_con):
            print("Distance file does not exist..extarcting distances")
            
            distance = self.get_distance(features, partition_ids) #get distance from partition center, instead of targets we use partition_ids

            with open(name_distance_file_con, "wb") as file:
                pickle.dump(distance, file)
        else:
            print("Distance file already exist")
            with open(name_distance_file_con, "rb") as f:
                distance = pickle.load(f)

        # print("Distance obtained")
        # print("Distance shape" , distance.shape)

        # print("max distance ", distance.max())
        # print("min distance ", distance.min())
        
        ids = self.select_idx(  distance,config,targets, partition_ids)
        
        # print("Selected ids ", ids)
        # print("distance ", distance[ids])
        # print("targets ", targets[ids])
        # print("partition_ids ", partition_ids[ids])
        # print("shape ", ids.shape)
        
        

        return ids

