
'''
obs : 절대좌표로 민맥스스케일링(4000,4000,400)
learn 시점 : 3개의 배치를 모으고, 최소 트랜지션 수 걸어놓음

'''
import numpy as np
import torch
from easydict import EasyDict
from ding.model import VAC
from ding.policy import PPOPolicy
from ding.utils import set_pkg_seed
from ding.torch_utils import to_device, unsqueeze
from tensorboardX import SummaryWriter
from socket_env import SocketEnv
from ding.envs import BaseEnvTimestep

# ===== 설정 =====
TOTAL_STEPS = 50000
BATCH_SIZE = 64
EPOCH_PER_COLLECT = 8
UNROLL_LEN = 1

# ===== 환경 초기화 =====
env = SocketEnv({
    'map_size': 4000,
    'max_step': 1000,
    'win_reward': 5.0,
    'num_detectable': 2
})
OBS_SHAPE = env.observation_space.shape[0]
print(OBS_SHAPE)
ACTION_DIM = env.action_space.n

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
        epoch_per_collect=8, #수집한 데이터를 몇 번 반복해서 학습할지, 데이터를 얼마나 재사용할지
        batch_size=32,
        learning_rate=3e-4,
        lr_scheduler=None,
        value_weight=0.5,
        entropy_weight=0.01, # 탐험정도
        clip_ratio=0.15, # 높을수록 정책이 급변, 기존정책에서 몇퍼센트 변화를 허용하는지, 0.10 -> 10% 변화 허용 그 이상은 X
        adv_norm=True, #false or true / 정규화 여부 중요
        value_norm=False, #false or true / 정규화 여부 중요 / 이거 켜면 nan으로 꽉차는 오류발생생
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

# ===== 초기화 =====
set_pkg_seed(0, use_cuda=cfg.cuda)
obs = env.reset()
model = VAC(**cfg.model)
print(model.actor)
device = 'cuda' if cfg.cuda else 'cpu'
model.to(device)
policy = PPOPolicy(cfg, model=model)
writer = SummaryWriter('./tensorlog_ppo')
first_obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
print("first obs :" , first_obs)

#torch.autograd.set_detect_anomaly(True) 
#오류 찾는 코드?

# ===== PPO 수집 및 학습 =====
global_step = 0
episode_idx = 0
episode_reward = 0
episode_buffer = []  # 여러 에피소드 저장
episode_batch_size = 3  # 몇 개의 에피소드 모아서 학습할지
min_transitions = 0 # 총알데미지 고려해서 최적횟수조정
all_episode_rewards = []  
probs_history = []  # 행동 확률 저장
learn_steps = []    # 학습 시점 저장

while global_step < TOTAL_STEPS:
    trajectory = []
    step = 0
    done = False
    obs_tensor = torch.zeros_like(first_obs)

    while not done and step < env.max_step:
        #obs_tensor = torch.rand(1, OBS_SHAPE)
        #print("obs :", obs_tensor)
        with torch.no_grad():
            output = model(obs_tensor, mode='compute_actor_critic')
            print("Actor logits:", output['logit'])
            print("Value:", output['value'])
            collect_output = policy.collect_mode.forward({0: obs_tensor})[0]      
            print("selected action:", collect_output['action'])

        logits = collect_output['logit']
        logits = torch.clamp(logits, -10, 10)  # 과도한 음수/양수 방지
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs_history.append(probs.squeeze(0).cpu().numpy().tolist())

        print(f"[DEBUG] logits: {logits} → probs: {probs}")

        action = collect_output['action'].item()
        #action = 1 #action = 2 (뒤로가기) 로 고정
        env_action = action + 1
        timestep = env.step(np.array([env_action], dtype=np.int64))

        if np.isnan(timestep.obs).any() or np.isinf(timestep.obs).any():
            print("[ERROR] Invalid obs detected:", timestep.obs)
        if np.isnan(timestep.reward).any() or np.isinf(timestep.reward).any():
            print("[ERROR] Invalid reward detected:", timestep.reward)

        # 위치정보의 다음 관측값을 tensor로 변환하고 변화량 계산
        max_vals = torch.tensor([4000.0, 4000.0, 400.0] * (OBS_SHAPE // 3), device=device)
        obs_tensor_next = timestep.obs / max_vals  # 절대좌표 → [0, 1] 범위로 정규화
        obs_tensor_next_unsqueezed = obs_tensor_next.unsqueeze(0)

        # 보상 및 종료 여부 처리
        scaled_reward = timestep.reward[0] * 0.1  # 보상 스케일링
        reward_tensor = torch.tensor(scaled_reward, dtype=torch.float32)
        done_tensor = torch.tensor(timestep.done, dtype=torch.float32)

        # 환경 결과를 timestep 텐서로 변환
        timestep_tensor = BaseEnvTimestep(
            obs=obs_tensor_next,
            reward=reward_tensor,
            done=done_tensor,
            info=timestep.info
        )

        # transition 처리
        transition = {
            'obs': obs_tensor.squeeze(0),
            'next_obs': obs_tensor_next.squeeze(0),
            'action': collect_output['action'].squeeze(0),
            'logit': collect_output['logit'].squeeze(0),
            'value': collect_output['value'].squeeze(0),
            'reward': reward_tensor,  # 반드시 float tensor
            'done': done_tensor,      # 반드시 float tensor
        }
        trajectory.append(transition)

        trajectory.append(transition)

        print(f"[STEP {global_step}] EP {episode_idx} | Action: {env_action} | Reward: {reward_tensor.item():.2f}")

        # 다음 스텝으로 이동
        obs = timestep.obs
        obs_tensor = obs_tensor_next_unsqueezed  # 변화량 기준 obs 유지
        episode_reward += reward_tensor.item()
        global_step += 1
        step += 1
        done = timestep.done

    print(f"[EP {episode_idx}] Reward: {episode_reward:.2f}, Steps: {step}")
    writer.add_scalar("episode_reward", episode_reward, global_step)

    all_episode_rewards.append(episode_reward)

    episode_buffer.append(trajectory)
    episode_idx += 1
    obs = env.reset()
    first_obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)  # 🔥 중요: 새로운 에피소드의 기준 obs
    obs_tensor = torch.zeros_like(first_obs)  # 변화량은 처음엔 0
    episode_reward = 0

#learn되는 부분분

    if len(episode_buffer) >= episode_batch_size:
        combined_transitions = sum(episode_buffer, [])
        if len(combined_transitions) >= min_transitions:
            train_data = policy._get_train_sample(combined_transitions)

            for t in train_data:
                for key in t:
                    if isinstance(t[key], (bool, np.bool_)):
                        t[key] = torch.tensor(float(t[key]), dtype=torch.float32)

            if len(train_data) > 1:
                print(f">>> [POLICY LEARN] step {global_step} | {len(combined_transitions)} transitions")
                learn_output = policy._forward_learn(train_data)
                learn_steps.append(global_step)
                for name, param in policy._learn_model.named_parameters():
                    if param.grad is not None:
                        print(f"[GRAD] {name} grad norm: {param.grad.norm().item():.6f}")

                print("=== Learn Output ===")
                print(learn_output)
                for stat in learn_output:
                    for k, v in stat.items():
                        writer.add_scalar(k, v, global_step)
        episode_buffer.clear()

# 종료 시 환경 닫기
env.close()

#모델 저장
save_path = "C:/Users/CIL2/Desktop/DI-engine-main/unreal/learned_models/ppo_model.pth"
torch.save(policy._learn_model.state_dict(), save_path)
print(f"[MODEL SAVED] to {save_path}")

# ===== 보상 그래프 =====
import matplotlib.pyplot as plt

episode_rewards = all_episode_rewards 
avg_rewards = []
window_size = 10
for i in range(len(episode_rewards)):
    start = max(0, i - window_size + 1)
    avg_rewards.append(np.mean(episode_rewards[start:i+1]))

plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, label='Episode Reward', alpha=0.5, color='blue')
plt.plot(avg_rewards, label='10-Episode Moving Avg', linewidth=2, color='orange')
plt.title("Episode Rewards over Time (PPO)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ===== 행동 확률 그래프 =====
probs_array = np.array(probs_history)

plt.figure(figsize=(12, 6))
for i in range(probs_array.shape[1]):
    plt.plot(probs_array[:, i], label=f'Action {i}')

for step in learn_steps:
    plt.axvline(x=step, color='gray', linestyle='--', alpha=0.6, label='Learn Step' if step == learn_steps[0] else "")

plt.title("Action Probability over Time")
plt.xlabel("Step")
plt.ylabel("Probability")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
