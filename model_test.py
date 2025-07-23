import torch
import numpy as np
from multi_agent_env import SocketMultiAgentEnv
from ding.model import VAC
from ding.policy import PPOPolicy
from easydict import EasyDict

# ===== 설정 =====
MODEL_PATH = r"C:\Users\CIL2\Desktop\DI-engine-main\unreal\learned_models\ppo_model_20250722_002546.pth"
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')

# ===== 환경 초기화 =====
env = SocketMultiAgentEnv({
    'map_size': 4000,
    'max_step': 1000,
    'win_reward': 50,
    'num_detectable': 4,
    'num_agents': 2
})

agent_ids = env.agent_ids
OBS_SHAPE = env.observation_space[agent_ids[0]].shape[0]
ACTION_DIM = env.action_space[agent_ids[0]].n

# ===== PPO 설정 =====
config = dict(
    type='ppo',
    cuda=USE_CUDA,
    multi_gpu=False,
    on_policy=True,
    priority=False,
    priority_IS_weight=False,
    recompute_adv=True,
    action_space='discrete',
    nstep_return=False,
    multi_agent=True,  # ✅ 반드시 True
    transition_with_policy_data=True,

    model=dict(
        obs_shape=OBS_SHAPE,
        action_shape=ACTION_DIM,
        encoder_hidden_size_list=[128, 128, 64],
        actor_head_hidden_size=64,
        actor_head_layer_num=2,
        critic_head_hidden_size=64,
        critic_head_layer_num=2,
        share_encoder=True,
        action_space='discrete',
        activation=torch.nn.ReLU(),
        norm_type='LN',
    ),

    learn=dict(
        epoch_per_collect=8,
        batch_size=64,
        learning_rate=3e-4,
        lr_scheduler=None,
        value_weight=0.5,
        entropy_weight=0.01,
        clip_ratio=0.2,
        adv_norm=True,
        value_norm=True,
        ppo_param_init=True,
        grad_clip_type='clip_norm',
        grad_clip_value=1,
        ignore_done=False,
    ),

    collect=dict(
        unroll_len=1,
        discount_factor=0.99,
        gae_lambda=0.95,
    ),

    eval=dict(
        evaluator=dict(
            eval_freq=1000,
            n_episode=5,
            render=False,
        )
    ),

    other=dict(
        eps=dict(
            type='exp',
            start=1.0,
            end=0.05,
            decay=5000,
        )
    ),
)

cfg = EasyDict(config)

# ===== 모델 및 정책 초기화 + 로드 =====
state = torch.load(MODEL_PATH, map_location=device)
model = VAC(**cfg.model)
policy = PPOPolicy(cfg, model=model)

if 'learn_model' in state:
    policy._learn_model.load_state_dict(state['learn_model'])
    policy._collect_model.load_state_dict(state['collect_model'])
    policy._eval_model.load_state_dict(state['eval_model'])
elif 'model' in state:
    policy._learn_model.load_state_dict(state['model'])
    policy._collect_model.load_state_dict(state['model'])
    policy._eval_model.load_state_dict(state['model'])
else:
    raise RuntimeError("Saved model must contain 'model' or 'learn_model'.")

policy._collect_model.eval()
policy._eval_model.eval()
policy._learn_model.eval()

# ===== 실행 루프 =====
num_episodes = 5

for episode in range(num_episodes):
    print(f"\n=== [EPISODE {episode + 1}] ===")

    obs = env.reset()
    obs_tensor = {
        aid: torch.tensor(o, dtype=torch.float32).unsqueeze(0).to(device)
        for aid, o in obs.items()
    }

    step = 0
    done_all = False
    episode_reward = {aid: 0. for aid in agent_ids}

    while not done_all and step < env.max_step:
        with torch.no_grad():
            output = policy.collect_mode.forward(obs_tensor)

        actions = {aid: output[aid]['action'] for aid in agent_ids}
        logits = {aid: output[aid]['logit'] for aid in agent_ids}
        probs = {aid: torch.softmax(logits[aid], dim=-1).cpu().numpy() for aid in agent_ids}

        timestep = env.step(actions)
        obs_tensor = {
            aid: torch.tensor(o, dtype=torch.float32).unsqueeze(0).to(device)
            for aid, o in timestep['obs'].items()
        }

        for aid in agent_ids:
            episode_reward[aid] += timestep['reward'][aid]

        done = timestep['done']
        done_all = done["__all__"]
        step += 1

        for aid in agent_ids:
            action = actions[aid].item()
            print(f"[{aid}] Step {step} | Action: {action}, Reward: {timestep['reward'][aid]:.2f}, Prob: {np.round(probs[aid], 3)}")

    print(f"[EPISODE {episode + 1} END] Total reward: {episode_reward} | Steps: {step}")

env.close()
