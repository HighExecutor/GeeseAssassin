from stable_baselines3.ppo import PPO
import pickle

def ppo2_load():
    orig_model = PPO.load("./../agents/ppo/ppo_v2/ppo_v2.zip")
    orig_model.policy.eval()
    orig_model.policy.to('cpu')
    orig_net = orig_model.policy.state_dict()
    target_dict = dict()
    for key, value in orig_net.items():
        value.to('cpu')
        p_key = None
        if 'features_extractor' in key:
            p_key = key[19:]
        if 'mlp_extractor.shared_net' in key:
            p_key = 'fc2' + key[26:]
        if 'mlp_extractor.policy_net' in key:
            p_key = 'fc3' + key[26:]
        if 'action_net' in key:
            p_key = 'fc4' + key[10:]

        if p_key is not None:
            target_dict[p_key] = value
    return target_dict

def ppo4_load():
    orig_model = PPO.load("./../agents/ppo/ppo_v4/ppo_vec_0.0.4")
    orig_model.policy.eval()
    orig_model.policy.to('cpu')
    orig_net = orig_model.policy.state_dict()
    target_dict = dict()
    for key, value in orig_net.items():
        value.to('cpu')
        p_key = None
        if 'features_extractor' in key:
            p_key = key[19:]
        if 'mlp_extractor.shared_net' in key:
            p_key = 'fc2' + key[26:]
        if 'mlp_extractor.policy_net' in key:
            p_key = 'fc3' + key[26:]
        if 'action_net' in key:
            p_key = 'fc4' + key[10:]

        if p_key is not None:
            target_dict[p_key] = value
    return target_dict


outpath = ".\..\submission\ppo4\\weights.pkl"
target_dict = ppo4_load()
data = pickle.dumps(target_dict)
pass
# with open(outpath, 'wb') as file:
#     pickle.dump(target_dict, file)
