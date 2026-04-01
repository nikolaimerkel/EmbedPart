import numpy as np  
import random
import time
import torch
class Migration:
    
    def random_migration(vid2pid, num_parts):
        
        
        print("min partition", np.min(vid2pid))
        print("max partition", np.max(vid2pid))
        def hamming_distance(a, b):
            return bin(a ^ b).count('1')

        # Function to generate the ordering of integers [0, 15] based on Hamming distance
        def generate_ordering(num_parts):
            integers = np.arange(num_parts)  # Generate integers from 0 to 15
            ordering_dict = {}

            for integer in integers:
                # Calculate Hamming distances from the current integer to all others
                distances = np.array([hamming_distance(integer, other) for other in integers])
                
                # Sort the integers based on their Hamming distances, excluding the current integer
                ordered_indices = np.argsort(distances)
                ordered_integers = integers[ordered_indices]
                
                # Exclude the current integer from its own ordering
                ordering_dict[integer] = np.array(ordered_integers[ordered_integers != integer])
            
            return ordering_dict
        # Generate and print the orderings
        ordering = generate_ordering(num_parts)
       # for key, value in ordering.items():
        #    print(f"{key}: {value}")
        
        
    
        unique_bins, counts = np.unique(vid2pid, return_counts=True)
        threshold = int(len(vid2pid) / num_parts)
        print(f"threshold: {threshold}, num vertices {len(vid2pid)}, num Parts = {num_parts}")
        pid2place = [threshold] * num_parts
        pid2overload = [0] * num_parts
        print("unique_bins", unique_bins)
        print("counts", counts)
        print("balance nefore migration", np.max(counts) / (len(vid2pid)/num_parts))
        num_vertices_to_migrate = 0
        pids_to_migrate = []
        for i in range(len(unique_bins)):
            pid = unique_bins[i]
            load = counts[i]
            if load <= threshold:
                pid2place[pid] = int(threshold - load)
            else:
                num_vertices_to_migrate += (load - threshold) 
                pids_to_migrate.append(pid)
                pid2overload[pid] = int(load - threshold )
                pid2place[pid] = 0
        print("num_vertices_to_migrate in percent",num_vertices_to_migrate / len(vid2pid))           
        print("pid2place",pid2place)
        pids = []

        print("pids",pids)      

        
        vid2pid_migrated = []
        for i in range(len(vid2pid)):
            if pid2overload[vid2pid[i]] > 0:
                current_pid = vid2pid[i]
                for potential_pid in ordering[current_pid]:
                    if pid2place[potential_pid] >= 0:
                        vid2pid_migrated.append(potential_pid)
                        pid2place[potential_pid] = pid2place[potential_pid] -1
                        pid2overload[current_pid] = pid2overload[current_pid] -1
                        break

            
 #               vid2pid_migrated.append(random.choices(pids, probs)[0])
  #              pid2overload[vid2pid[i]] = pid2overload[vid2pid[i]] -1
 
            else:
                vid2pid_migrated.append(vid2pid[i])
                
                
        print("length", len(vid2pid_migrated) )  
        return np.array(vid2pid_migrated)
    
    
    
    
    def random_migration_old(vid2pid, num_parts):
        
        unique_bins, counts = np.unique(vid2pid, return_counts=True)
        threshold = int(len(vid2pid) / num_parts)
        print(f"threshold: {threshold}, num vertices {len(vid2pid)}, num Parts = {num_parts}")
        pid2place = [threshold] * num_parts
        pid2overload = [0] * num_parts
        print("unique_bins", unique_bins)
        print("counts", counts)
        print("balance nefore migration", np.max(counts) / (len(vid2pid)/num_parts))
        num_vertices_to_migrate = 0
        pids_to_migrate = []
        for i in range(len(unique_bins)):
            pid = unique_bins[i]
            load = counts[i]
            if load <= threshold:
                pid2place[pid] = int(threshold - load)
            else:
                num_vertices_to_migrate += (load - threshold) 
                pids_to_migrate.append(pid)
                pid2overload[pid] = int(load - threshold )
                pid2place[pid] = 0
        print("num_vertices_to_migrate in percent",num_vertices_to_migrate / len(vid2pid))           
        print("pid2place",pid2place)
        pids = []
        probs = []
        for i in range(len(pid2place)):
            if pid2place[i] > 0:
                pids.append(i)
                probs.append(pid2place[i] / num_vertices_to_migrate)
                
        print("pids",pids)      
        print("probs", probs)    
        print(len(pids), len(probs), len(pids_to_migrate))
        print(np.sum(probs))  
        
        vid2pid_migrated = []
        for i in range(len(vid2pid)):
            if vid2pid[i] in pids_to_migrate and pid2overload[vid2pid[i]] > 0:
                vid2pid_migrated.append(random.choices(pids, probs)[0])
                pid2overload[vid2pid[i]] = pid2overload[vid2pid[i]] -1
            else:
                vid2pid_migrated.append(vid2pid[i])
        return np.array(vid2pid_migrated)





    import numpy as np

    def migrate_training_balance_fast(vid2pid, num_parts, graph, train_balance, vertex_balance, distances=None):
        t1 = time.time()

        
        
        """
        Rebalances training and non-training vertices across partitions.
        Optimized for speed via NumPy vectorization.

        Args:
            vid2pid (np.ndarray): Initial partition assignments.
            num_parts (int): Number of partitions.
            graph (DGLGraph): Graph with 'train_mask' in ndata.
            train_balance (float): Target training balance factor.
            vertex_balance (float): Target vertex balance factor.

        Returns:
            np.ndarray: Balanced partition assignments.
        """
        def rebalance_phase(vid2pid, mask, num_parts, balance_factor, phase_name="",distances=None):
           # Make a copy of the current node-to-partition mapping to avoid modifying the original
            new_vid2pid = vid2pid.copy()

            # Get the indices of nodes where 'mask' is True (i.e., nodes to consider for migration)
            node_ids = np.where(mask)[0]

            # Get the partition IDs for the selected nodes
            pids = new_vid2pid[node_ids]

            # Count how many nodes are assigned to each partition (current load per partition)
            current_load = np.bincount(pids, minlength=num_parts)

            # Calculate the maximum allowed nodes per partition (with some slack, controlled by balance_factor)
            limit = int((balance_factor * len(node_ids)) / num_parts) + 1

          #  print("limit per partition in phase", phase_name, limit)
           # print("Current load before phase", phase_name, current_load)
           # print("Balance before phase", phase_name, np.max(current_load)/np.mean(current_load))
            
            # Calculate how many nodes each partition is over the limit
            pid2over = current_load - limit

            # Calculate how many nodes each partition is under the limit
            pid2under = limit - current_load

            # Find the indices of partitions that are overloaded (more nodes than allowed)
            overloaded_pids = np.where(pid2over > 0)[0]

            # Find the indices of partitions that are underloaded (fewer nodes than allowed)
            underloaded_pids = np.where(pid2under > 0)[0]

            # Get the underload amounts for underloaded partitions as float (for probability calculation)
            underload_probs = pid2under[underloaded_pids].astype(np.float32)

            # Sum of underload amounts (used for normalization)
            underload_probs_sum = underload_probs.sum()

            # If there is no underloaded partition, return the current mapping (already balanced)
            if underload_probs_sum == 0:
                return new_vid2pid  # Already balanced

            # Normalize underload amounts to get probabilities for assigning nodes to underloaded partitions
            underload_probs /= underload_probs_sum

            # For each overloaded partition, try to migrate nodes out
            for pid in overloaded_pids:
                # Find nodes in this partition that are candidates for migration
                
                candidates = node_ids[new_vid2pid[node_ids] == pid]
                
                # Number of nodes to move out of this partition
                num_to_move = pid2over[pid]
           #     print("Number of candidates in partition", pid, "in phase", phase_name, len(candidates), "need to move", num_to_move)
                # If there are nodes to move and candidates available
                if num_to_move > 0 and len(candidates) > 0:
                    # Randomly select nodes to migrate (up to num_to_move or available candidates)
                    selected = None
                    
                    if distances is None:
                        print("No distances provided, using random selection")
                        selected = np.random.choice(candidates, size=min(num_to_move, len(candidates)), replace=False)
                    else: 
                        candidates_distances = distances[candidates]
                        print("Using distances to select nodes for migration")
                        topk_values, topk_indices = torch.topk(candidates_distances.squeeze(), num_to_move, largest=True)
                     #   print("number of topk indices", len(topk_indices), "num to move", num_to_move, "candidates", len(candidates), "largest", torch.max(topk_indices), "smallest", torch.min(topk_indices))
                        selected = candidates[topk_indices.cpu().numpy()]
                        
                    print("Migrating", len(selected), "nodes from partition", pid, "in phase", phase_name, "from candidates", len(candidates))
                    # Randomly assign selected nodes to underloaded partitions, weighted by underload probabilities
                    target_pids = np.random.choice(underloaded_pids, size=len(selected), p=underload_probs)

                    # Update the mapping to reflect the new partition assignments
                    new_vid2pid[selected] = target_pids

        
                    np.add.at(pid2under, target_pids, -1)
                    # Update the list of underloaded partitions (remove those that are no longer underloaded)
                    underloaded_pids = underloaded_pids[pid2under[underloaded_pids] > 0]
                    # Mask for partitions still underloaded
                    mask = pid2under[underloaded_pids] > 0
                    # Update underloaded_pids again (redundant, but ensures correctness)
                    underloaded_pids = underloaded_pids[mask]
                    # Recalculate underload probabilities
                    underload_probs = pid2under[underloaded_pids].astype(np.float32)

                    # If no more underloaded partitions, stop migrating
                    if underload_probs.sum() == 0:
                        break
                    # Normalize probabilities again
                    underload_probs /= underload_probs.sum()

            # Return the updated node-to-partition mapping
            return new_vid2pid

        train_mask = graph.ndata['train_mask'].cpu().numpy().astype(bool)
        val_mask = graph.ndata['val_mask'].cpu().numpy().astype(bool)
        remaining_mask = ~train_mask & ~val_mask
    
        # Phase 1: Balance training vertices
        
        # only the distances of the training vertices
       
       # print("vertices in Graph", graph.num_nodes())
       # print("number of training vertices", np.sum(train_mask), "number of validation vertices", np.sum(val_mask), "number of remaining vertices", np.sum(remaining_mask))
        distances = -graph.in_degrees().cpu()
        #distances = None
        
        vid2pid = rebalance_phase(vid2pid, train_mask, num_parts, train_balance, phase_name="TRAIN", distances=distances)

        # Phase 2: Balance validation vertices
    
        vid2pid = rebalance_phase(vid2pid, val_mask, num_parts, train_balance, phase_name="VAL",distances=distances)
        
        # Phase 3: Balance remaining vertices (non-training and non-validation)
 
        vid2pid = rebalance_phase(vid2pid, remaining_mask, num_parts, vertex_balance, phase_name="NON-TRAIN-VAL",distances=distances)
        
        t2 = time.time()
        print(f"Migration time: {t2 - t1:.2f} seconds.")
        return vid2pid



    def migrate_training_balance(vid2pid, num_parts, graph, train_balance, vertex_balance):
        """
        Phase 1: Balance training nodes exactly across partitions.
        Phase 2: Balance non-training nodes, ensuring imbalance ≤ max_non_train_balance.

        Args:
            vid2pid (np.ndarray): Partition assignments for each node.
            num_parts (int): Number of partitions.
            graph (DGLGraph): Graph with 'train_mask' in ndata.
            max_non_train_balance (float): Max allowed imbalance for non-training nodes.

        Returns:
            np.ndarray: New partition assignments.
        """
        newVid2pid = vid2pid.copy()
        
        train_mask = graph.ndata['train_mask'].cpu().numpy().astype(bool)
       # print("Debug", train_mask, len(train_mask))
       # print(train_mask.dtype)
        #print(np.unique(train_mask))
        train_nids = np.nonzero(train_mask)[0]  
        train_pids = newVid2pid[train_nids]
        
        non_train_nids = np.nonzero(~train_mask)[0]
       # overlap = np.intersect1d(train_nids, non_train_nids)
        #print(overlap)
        #print("Overlap", len(overlap), len(train_nids), len(non_train_nids))
        #assert np.all((train_mask == True) | (train_mask == False)), "train_mask has non-binary values"
        training_limit = int((train_balance * len(train_nids)) / num_parts)+1
        current_load = np.bincount(train_pids, minlength=num_parts)
        
        print("Limit", training_limit)
        num_vertices_to_migrate = 0
        pid2migrate = [0] * num_parts
        pid2prob = [0] * num_parts
        for i, d in enumerate(current_load):
            if d > training_limit:
                num_vertices_to_migrate += (d - training_limit)
                pid2migrate[i] = int(d - training_limit)
            else:
                pid2prob[i] = int(training_limit - d)
        #pid2prob = pid2prob / num_vertices_to_migrate
        pid2prob = [x / sum(pid2prob) for x in pid2prob]
        print("vertices to migrate", num_vertices_to_migrate)
        print("Current Load", current_load)
        print("pid2migrate", pid2migrate)
        print("pid2prob", pid2prob)    
        
        underloaded_pids = []
        underloaded_pids2probs = []
        
        for i in range(len(pid2prob)):
            if pid2prob[i] > 0:
                underloaded_pids.append(i)
                underloaded_pids2probs.append(pid2prob[i])
        print("underloaded_pids", underloaded_pids)
        print("underloaded_pids2probs", underloaded_pids2probs)
                
        train_nids_copy = train_nids.copy()
        random_permutation = np.random.permutation(train_nids_copy)
        migrated = 0
        for vid in random_permutation:
            if pid2migrate[newVid2pid[vid]] > 0:
                from_pid = newVid2pid[vid]
                newVid2pid[vid] = random.choices(underloaded_pids, underloaded_pids2probs)[0]
                #print("migated", vid, "from", from_pid, "to", newVid2pid[vid])
                pid2migrate[from_pid] = pid2migrate[from_pid] -1
                migrated += 1   
        
        #train_pids = newVid2pid[train_nids]
        #current_load = np.bincount(train_pids, minlength=num_parts)
        #print("After migration of", migrated, "vertices")
        #print("Current Load", current_load)
        #print("Balance", np.max(current_load)/np.mean(current_load))
        
        
        ################# 
        #############
        #########current_load = np.bincount(newVid2pid, minlength=num_parts)
        
        
        print("\n\nstart with NON training vertices\n\n")
        
  
        #print("Overlap", len(overlap), len(train_nids), len(non_train_nids))
        
        vertex_limit = int((vertex_balance * len(newVid2pid)) / num_parts) + 1
        current_load = np.bincount(newVid2pid, minlength=num_parts)
        
        print("Limit", vertex_limit)
        num_vertices_to_migrate = 0
        pid2migrate = [0] * num_parts
        pid2prob = [0] * num_parts
        for i, d in enumerate(current_load):
            if d > vertex_limit:
                num_vertices_to_migrate += (d - vertex_limit)
                pid2migrate[i] = int(d - vertex_limit)
            else:
                pid2prob[i] = int(vertex_limit - d)
        #pid2prob = pid2prob / num_vertices_to_migrate
        pid2prob = [x / sum(pid2prob) for x in pid2prob]
        print("vertices to migrate", num_vertices_to_migrate)
        print("Current Load", current_load)
        print("pid2migrate", pid2migrate)
        print("pid2prob", pid2prob)    
        
        underloaded_pids = []
        underloaded_pids2probs = []
        
        for i in range(len(pid2prob)):
            if pid2prob[i] > 0:
                underloaded_pids.append(i)
                underloaded_pids2probs.append(pid2prob[i])
        print("underloaded_pids", underloaded_pids)
        print("underloaded_pids2probs", underloaded_pids2probs)
                
        
        non_train_nids_copy = non_train_nids.copy()
        print("in theory we have", len(non_train_nids_copy), "non training vertices and need to migrate", num_vertices_to_migrate)
        random_permutation = np.random.permutation(non_train_nids_copy)
        migrated = 0
        for vid in random_permutation:
            if pid2migrate[newVid2pid[vid]] > 0:
                from_pid = newVid2pid[vid]
                newVid2pid[vid] = random.choices(underloaded_pids, underloaded_pids2probs)[0]
                #print("migated", vid, "from", from_pid, "to", newVid2pid[vid])
                pid2migrate[from_pid] = pid2migrate[from_pid] -1
                migrated += 1   
       # print("We migrated ", migrated, "non training vertices")
        
        
        
        
        
      #  non_train_pids = newVid2pid[non_train_nids]
      #  current_load = np.bincount(newVid2pid, minlength=num_parts)
      #  print("Current Load", current_load)
       # print("vertex balance", np.max(current_load)/np.mean(current_load))
                
       # print("\n\nAre training vertices still balanced")
       # train_pids = newVid2pid[train_nids]
       # current_load = np.bincount(train_pids, minlength=num_parts)
       # print("Current Load", current_load)
       # print("Balance train", np.max(current_load)/np.mean(current_load))
        
        

        return newVid2pid