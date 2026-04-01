import argparse
import numpy as np  

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-bin2ascii', 
        type=str, 
        help="each line contains to numbers: new_vid old_vid. This was create by the converter in c++",
        required=True
    )
    
    parser.add_argument(
        '-vid2pid', 
        type=str, 
        help="each line contains one number: pid out put of c++ partitioning",
        required=True
    )
    
    parser.add_argument(
        '-vid2pidOutput', 
        type=str, 
        help="the output vid2pid file which we load into dgl",
        required=True
    )
    
    
    


    return parser



# some notes:
# bin2ascii: new_vid -> old_vid
# vid2pid: new_vid -> pid

# goal old_vid -> pid

if __name__ == "__main__":
    parser = parser()
    args = parser.parse_args()
    
    data = np.loadtxt(args.bin2ascii, dtype=int)
    
    print(f"{args.bin2ascii} contains \n\t shape: {data.shape} \n\t vertices: {len(data)} \n\t max_vid: {np.max(data)} \n\t min_vid: {np.min(data)}")
    

    # Create a NumPy array where index = new_id, value = old_id
    max_new_id = np.max(data[:, 0])  # Find the highest new_id
    max_old_id = np.max(data[:, 1])  # Find the highest old_id
    print(f"max_new_id: {max_new_id} \nmax_old_id: {max_old_id}")
    new2old = np.zeros(max_new_id + 1, dtype=int)  # Preallocate array

    for new_id, old_id in data:
        new2old[new_id] = old_id

    vid2pid = np.loadtxt(args.vid2pid, dtype=int)

    old2pid = np.zeros_like(new2old)
    for new_id, old_id in enumerate(new2old):
        old2pid[old_id] = vid2pid[new_id]


    print(old2pid)  # Output: [1, 0, 2]
 
    with open(f"{args.vid2pidOutput}", "w") as f:
        f.write("\n".join(map(str, old2pid)))