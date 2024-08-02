import os.path
import pandas as pd
from torch_geometric.data import HeteroData
import torch

def generate_supervision_relationships(data_path, userlist_path):
    # Load the data
    data = pd.read_csv(data_path)
    userlist = pd.read_csv(userlist_path)

    # Create a dictionary for quick lookup of supervisors based on user_id
    supervisor_dict = userlist.set_index('user_id')['supervisor'].to_dict()
    UserName_dict = userlist.set_index('employee_name')['user_id'].to_dict()

    # Initialize lists to store head and tail nodes
    head_nodes = []
    tail_nodes = []
    date_IDs = []
    user_IDs = []

    # Iterate through each row in the data dataframe
    for UserDay_index, row in data.iterrows():
        user_ID = row['user_index']
        if user_ID not in supervisor_dict:
            print(f'for {user_ID} cant not find its id in supervisor_dict, please check')
            break
        else:
            SupervisorName = supervisor_dict[user_ID]
            if SupervisorName not in UserName_dict:
                print(f'for {user_ID} cant not find supervisor, please check, now set supervisor_name to xxx')
                supervisor_ID = 'xxx'
            else:
                supervisor_ID = UserName_dict[SupervisorName]
            head_nodes.append(UserDay_index)
            tail_nodes.append(supervisor_ID)
            date_IDs.append(row['date_index'])
            user_IDs.append(user_ID)

    # Create the resulting dataframe
    # UserDay_df = pd.DataFrame({'UserDay_index': head_nodes,
    #     'date_ID': date_IDs,
    #     'user_ID': user_IDs})
    UserDay_df = pd.DataFrame({'UserDay_index': head_nodes})
    UserDay_df = pd.concat([UserDay_df, data[['date_index','user_index','group','label','f1','f2','f3','f4','f5']]],axis=1)

    supervisor_df = pd.DataFrame({'supervisor_id': tail_nodes})
    unique_supervisor_df = supervisor_df.drop_duplicates(subset='supervisor_id').reset_index(drop=True)
    unique_supervisor_df['remapped_index'] = unique_supervisor_df.index
    supervisor_dict = unique_supervisor_df.set_index('supervisor_id')['remapped_index'].to_dict()

    supervision_relation_df = pd.DataFrame({
        'head_index': head_nodes,
        'tail_index': tail_nodes
    })
    supervision_relation_df['tail_index'].replace(supervisor_dict, inplace=True)

    return supervision_relation_df, UserDay_df, unique_supervisor_df

def generate_SameUser_relationships(data_path, userlist_path):
    data = pd.read_csv(data_path)
    userlist = pd.read_csv(userlist_path)

    # Create a dictionary for quick lookup of
    unique_user_df = data.drop_duplicates(subset='user_index').reset_index(drop=True)
    unique_user_df['remapped_index'] = unique_user_df.index
    unique_user_df = unique_user_df[['user_index', 'remapped_index']]
    unique_user_remapping_dict = unique_user_df.set_index('user_index')['remapped_index'].to_dict()

    # Initialize lists to store head and tail nodes
    head_nodes = []
    tail_nodes = []
    date_IDs = []
    user_IDs = []

    # Iterate through each row in the data dataframe
    for UserDay_index, row in data.iterrows():
        user_ID = row['user_index']
        if user_ID in unique_user_remapping_dict:
            head_nodes.append(UserDay_index)
            tail_nodes.append(unique_user_remapping_dict[user_ID])
            date_IDs.append(row['date_index'])
            user_IDs.append(user_ID)

    SameUser_relation_df = pd.DataFrame({
        'head_index': head_nodes,
        'tail_index': tail_nodes
    })

    return SameUser_relation_df, unique_user_df

def generate_share_supervisor_relationships(data_path, userlist_path):
    # Load the data
    data = pd.read_csv(data_path)
    data['remapped_index'] = data.index
    userlist = pd.read_csv(userlist_path)

    # Create a dictionary for quick lookup of supervisors based on user_id
    supervisor_dict = userlist.set_index('user_id')['supervisor'].to_dict()
    UserName_dict = userlist.set_index('employee_name')['user_id'].to_dict()

    # Initialize lists to store head and tail nodes
    head_nodes = []
    tail_nodes = []
    date_IDs = []
    user_IDs = []

    # Iterate through each row in the data dataframe
    for UserDay_index, row in data.iterrows():
        user_ID = row['user_index']
        if user_ID not in supervisor_dict:
            print(f'for {user_ID} cant not find it id in supervisor_dict, please check')
            break
        else:
            SupervisorName = supervisor_dict[user_ID]
            if SupervisorName not in UserName_dict:
                print(f'for {user_ID} cant not find supervisor, please check, now set supervisor_name to xxx')
                supervisor_ID = 'xxx'
            else:
                supervisor_ID = UserName_dict[SupervisorName]
        if supervisor_ID != 'xxx':
            share_supervisor_userlist = userlist[userlist['supervisor']==SupervisorName]['user_id'].to_list()
            share_supervisor_userlist.remove(user_ID) # remove self from the share supervisor list
            tail_nodes_addlist=[] # whole tail nodes list for a userday node.
            for user in share_supervisor_userlist:
                df_for_user = data[data['user_index'] == user]
                if len(df_for_user) > 0:
                    UserDay_list_for_user = df_for_user['remapped_index'].to_list()
                    tail_nodes_addlist = tail_nodes_addlist + UserDay_list_for_user
            if tail_nodes_addlist:
                head_nodes = head_nodes + [row['remapped_index']]*len(tail_nodes_addlist)
                tail_nodes = tail_nodes + tail_nodes_addlist

    share_supervisor_relationship_df = pd.DataFrame({
        'head_index': head_nodes,
        'tail_index': tail_nodes
    })

    return share_supervisor_relationship_df


# Paths to the files
# current_directory = os.path.dirname(os.path.abspath(__file__))
data_path = './CERT4.2/user_feature_label/data-wise_total.csv'
userlist_path = './CERT4.2/userlist.csv'

# Generate the supervisor relationships dataframe
supervision_relationships_df, UserDay_df, unique_supervisor_df = generate_supervision_relationships(data_path, userlist_path)
SameUser_relationships_df, unique_user_df = generate_SameUser_relationships(data_path, userlist_path)
share_supervisor_relationships_df = generate_share_supervisor_relationships(data_path, userlist_path)

#%% Create a HeteroData object
data = HeteroData()

# UserDay nodes feature
data['UserDay'].x = torch.tensor(UserDay_df[['f1', 'f2', 'f3', 'f4', 'f5']].values, dtype=torch.float64)
print(data['UserDay'].x)
data['UserDay'].y = torch.tensor(UserDay_df['label'].values)
data['UserDay'].node_id = torch.tensor(UserDay_df['UserDay_index'].values)
print(data['UserDay'].node_id.shape,data['UserDay'].node_id[-1])

# train-test mask:
data['UserDay'].train_mask = torch.zeros(UserDay_df.shape[0], dtype=torch.bool)
data['UserDay'].test_mask = torch.zeros(UserDay_df.shape[0], dtype=torch.bool)
train_idx = torch.arange(0, 1336)
test_idx = torch.arange(1336, UserDay_df.shape[0])
data['UserDay'].train_mask[train_idx] = True
data['UserDay'].test_mask[test_idx] = True

# supervisor and user node
data['supervisor'].node_id =torch.tensor(unique_supervisor_df['remapped_index'].values)
data['user'].node_id =torch.tensor(unique_user_df['remapped_index'].values)

# relationship
data['UserDay', 'has_supervisor', 'supervisor'].edge_index = torch.transpose(
    torch.tensor(supervision_relationships_df[['head_index','tail_index']].values),0,1) # [2, 595506]
data['UserDay', 'has_SameUser', 'user'].edge_index = torch.transpose(
    torch.tensor(SameUser_relationships_df[['head_index','tail_index']].values),0,1) # [2, 36063]
# Save the heterogeneous graph data
torch.save(data, './data/insider_detection_heterogeneous_graph.pt')

data['UserDay', 'share_supervisor', 'UserDay'].edge_index = torch.transpose(
    torch.tensor(share_supervisor_relationships_df[['head_index','tail_index']].values),0,1)
# Save the heterogeneous graph data
torch.save(data, './data/insider_detection_heterogeneous_graph_3relations.pt')

