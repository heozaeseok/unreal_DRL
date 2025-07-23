import torch.nn.functional as F
import numpy as np
import torch
from easydict import EasyDict
from ding.model import VAC
from ding.policy import PPOPolicy
from ding.utils import set_pkg_seed
from ding.torch_utils import to_device
from tensorboardX import SummaryWriter
from multi_agent_env import SocketMultiAgentEnv
import matplotlib.pyplot as plt
import random
import os
from datetime import datetime

# ===== 설정 =====
TOTAL_STEPS = 100000
reward_scale = 0.1
win_reward = 50

# ===== 환경 초기화 =====
env = SocketMultiAgentEnv({
    'map_size': 4000,
    'max_step': 1000,
    'win_reward': win_reward,
    'num_detectable': 4,
    'num_agents': 2
})

agent_ids = env.agent_ids
OBS_SHAPE = env.observation_space[agent_ids[0]].shape[0]
ACTION_DIM = env.action_space[agent_ids[0]].n

config = dict(
    type='ppo',
    action_space='discrete',
    cuda=True,
    multi_gpu=False,
    on_policy=True,
    priority=False,
    priority_IS_weight=False,
    multi_agent=True,
    recompute_adv=True,
    transition_with_policy_data=True,
    nstep_return=False,

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
        epoch_per_collect=4,
        batch_size=64,
        learning_rate=3e-4,
        lr_scheduler=None,
        value_weight=0.5,
        entropy_weight=0.02,
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
    other=dict(
        eps=dict(
            type='exp', start=1.0, end=0.05, decay=5000
        )
    )
)
cfg = EasyDict(config)

set_pkg_seed(0, use_cuda=cfg.cuda)
device = torch.device('cuda' if cfg.cuda else 'cpu')

model = VAC(**cfg.model).to(device)
policy = PPOPolicy(cfg, model=model)
writer = SummaryWriter('./tensorlog_ppo_multi')

obs = env.reset()
obs_tensor = {aid: torch.tensor(o, dtype=torch.float32).unsqueeze(0).to(device) for aid, o in obs.items()}

transition_buffer = []
global_step = 0
epsilon = 1.0
epsilon_end = 0.05
epsilon_decay = 30000

episode_rewards_by_agent = {aid: [] for aid in agent_ids}

while global_step < TOTAL_STEPS:
    done = {aid: False for aid in agent_ids}
    done["__all__"] = False
    episode_reward = {aid: 0. for aid in agent_ids}
    episode_step = 0  

    while not done["__all__"]:
        with torch.no_grad():
            collect_output = policy.collect_mode.forward(obs_tensor)

        actions = {}
        for aid in agent_ids:
            if random.random() < epsilon:
                actions[aid] = torch.tensor([env.action_space[aid].sample()], device=device)
            else:
                actions[aid] = collect_output[aid]['action']

        logits = {aid: collect_output[aid]['logit'] for aid in agent_ids}
        values = {aid: collect_output[aid]['value'] for aid in agent_ids}

        print(f"[GLOBAL STEP {global_step}] [EPISODE STEP {episode_step}] Actions: " + 
              ", ".join([f"{aid}: {actions[aid].item()}" for aid in agent_ids]))

        step_result = env.step(actions)
        next_obs_tensor = {
            aid: torch.tensor(o, dtype=torch.float32).unsqueeze(0).to(device)
            for aid, o in step_result['obs'].items()
        }
        done = step_result['done']

        # Step 업데이트
        global_step += 1
        episode_step += 1
        epsilon = max(epsilon_end, epsilon - (1.0 - epsilon_end) / epsilon_decay)

        # 중간 모델 저장
        if global_step % 20000 == 0:  # 원하는 저장 주기 설정 (예: 10000)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = r"C:\Users\CIL2\Desktop\DI-engine-main\unreal\learned_models"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"ppo_model_step{global_step}_{timestamp}.pth")
            torch.save(policy._state_dict_learn(), save_path)
            print(f"[AUTO-SAVED] Step {global_step} → {save_path}")

        #transition 생성
        for aid in agent_ids:

            scaled_reward = step_result['reward'][aid] * reward_scale

            transition = {
                'obs': obs_tensor[aid].squeeze(0),
                'next_obs': next_obs_tensor[aid].squeeze(0),
                'action': actions[aid].squeeze(0),
                'logit': logits[aid].squeeze(0),
                'value': values[aid].squeeze(0),
                'reward': torch.tensor(scaled_reward, dtype=torch.float32),
                'done': torch.tensor(float(done[aid]), dtype=torch.float32),
            }
            transition_buffer.append(transition)
            episode_reward[aid] += scaled_reward

        obs_tensor = next_obs_tensor

        #에피소드 종료 처리
        if done["__all__"]:
            print(f"[EP DONE] Global Step: {global_step}, Rewards: {episode_reward}")
            for aid in agent_ids:
                episode_rewards_by_agent[aid].append(episode_reward[aid])
            obs = env.reset()
            obs_tensor = {
                aid: torch.tensor(o, dtype=torch.float32).unsqueeze(0).to(device)
                for aid, o in obs.items()
            }

    #학습 조건 충족 시
    if len(transition_buffer) >= 1024:
        train_data = policy._get_train_sample(transition_buffer)
        train_data = [
            {
                k: (
                    torch.tensor(v, dtype=torch.float32).to(device)
                    if isinstance(v, (bool, np.bool_)) else
                    v.to(device) if isinstance(v, torch.Tensor) else v
                )
                for k, v in t.items()
            }
            for t in train_data
        ]
        learn_output = policy._forward_learn(train_data)
        print(f"[LEARN] {learn_output}")
        transition_buffer.clear()

#종료
env.close()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = r"C:\Users\CIL2\Desktop\DI-engine-main\unreal\learned_models"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"ppo_model_{timestamp}.pth")

torch.save(policy._state_dict_learn(), save_path)
print(f"[SAVED] {save_path}")

# === 보상 시각화 ===
plt.figure(figsize=(10, 6))
for aid in agent_ids:
    plt.plot(episode_rewards_by_agent[aid], label=f'{aid}')
    # 이동평균
    window = 10
    avg = [np.mean(episode_rewards_by_agent[aid][max(0, i - window + 1):i + 1])
           for i in range(len(episode_rewards_by_agent[aid]))]
    plt.plot(avg, linestyle='--', label=f'{aid} (MA{window})')

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episode Rewards per Agent')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()