import torch
from ding.model import VAC
from ding.policy import PPOPolicy
from socket_env import SocketEnv
from easydict import EasyDict
import numpy as np

# === 환경 설정 ===
env = SocketEnv({
    'map_size': 4000,
    'max_step': 100,
    'win_reward': 5.0,
    'num_detectable': 2
})
OBS_SHAPE = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# === PPO config 설정 ===
config = dict(
    type='ppo',
    cuda=False,
    multi_gpu=False,
    on_policy=True,
    priority=False,
    priority_IS_weight=False,
    recompute_adv=True,
    action_space='discrete',
    nstep_return=False,
    multi_agent=False,
    transition_with_policy_data=True,

    model=dict(
        obs_shape=OBS_SHAPE,
        action_shape=ACTION_DIM,
        encoder_hidden_size_list=[64, 64],
        actor_head_hidden_size=64,
        critic_head_hidden_size=64,
        share_encoder=True,
        action_space='discrete',
    ),

    learn=dict(
        epoch_per_collect=4,
        batch_size=32,
        learning_rate=3e-4,
        lr_scheduler=None,
        value_weight=0.5,
        entropy_weight=0.01,
        clip_ratio=0.15,
        adv_norm=True,
        value_norm=False,
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
            decay=1000,
        )
    ),
)
cfg = EasyDict(config)
device = 'cuda' if cfg.cuda else 'cpu'

# === 모델 불러오기 ===
model = VAC(**cfg.model)
model_path = r"C:\Users\CIL2\Desktop\DI-engine-main\unreal\learned_models\ppo_model_nowall.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# === PPO 정책 초기화 ===
policy = PPOPolicy(cfg, model=model)

# === 평가 루프 ===
obs = env.reset()
obs_tensor = torch.tensor(obs / np.array([4000.0, 4000.0, 400.0] * (OBS_SHAPE // 3)), dtype=torch.float32).unsqueeze(0).to(device)

done = False
episode_reward = 0
step = 0

while not done and step < env.max_step:
    with torch.no_grad():
        obs_dict = {
            'observation': obs_tensor,
            'action_mask': torch.ones((1, ACTION_DIM), dtype=torch.float32).to(device)  # ← bool → float32로 변경
        }
        output = policy._eval_model.forward(obs_dict, mode='compute_actor')
        logits = output['logit']
        action = torch.argmax(logits, dim=-1).item()

        probs = torch.softmax(logits, dim=-1)
        print(f"[DEBUG] logits: {logits.cpu().numpy()} → probs: {probs.cpu().numpy()}")

    env_action = action + 1  # 환경에서 액션 1~N 사용 시 보정
    timestep = env.step(np.array([env_action], dtype=np.int64))

    print(f"[STEP {step}] Action: {env_action}, Reward: {timestep.reward[0]:.2f}")
    episode_reward += timestep.reward[0]
    obs = timestep.obs
    obs_tensor = torch.tensor(obs / np.array([4000.0, 4000.0, 400.0] * (OBS_SHAPE // 3)), dtype=torch.float32).unsqueeze(0).to(device)

    done = timestep.done
    step += 1

print(f"[EVAL DONE] Total Reward: {episode_reward:.2f}, Steps: {step}")
