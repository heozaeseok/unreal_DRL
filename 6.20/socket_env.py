import numpy as np
from gym import spaces
from ding.envs import BaseEnv, BaseEnvTimestep
from socket_server import SocketServer
from ding.utils import ENV_REGISTRY
from ding.torch_utils.data_helper import to_ndarray 

@ENV_REGISTRY.register('socket_env')
class SocketEnv(BaseEnv):
    def __init__(self, cfg):
        self._cfg = cfg
        self._init_flag = False
        self.max_step = cfg.get('max_step', 1000) #에피소드당 최대 스텝수 조정
        self.map_size = cfg.get('map_size', 4000)
        self.win_reward = cfg.get('win_reward', 5.0)
        self.num_detectable = cfg.get('num_detectable', 2)

        obs_dim = 4 + self.num_detectable * 4 
        self._observation_space = spaces.Box(low=-self.map_size, high=self.map_size, shape=(obs_dim,), dtype=np.float32)
        self._action_space = spaces.Discrete(5)

    def reset(self):
        self.server = SocketServer()
        self.server.start()

        self.step_count = 0
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            np.random.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            np.random.seed(self._seed)

        self.server.send({"Order": int(np.random.randint(1, 6))})

        last_data = None
        for _ in range(20):
            data = self.server.receive()
            if data is None:
                continue
            #print("[RESET DEBUG] received:", data)
            #print(f"[CHECK] isDone = {data.get('isDone')} (type: {type(data.get('isDone'))})")
            last_data = data

            # 🔍 다양한 케이스에 대응
            is_done = data.get("isDone")
            if not is_done:
                self.current_obs = self._convert_obs(data)
                self._eval_episode_return = 0.
                return to_ndarray(self.current_obs)

        # fallback: 마지막 수신 데이터를 사용하여 reset 강제 진행
        if last_data:
            print("[RESET WARNING] Using fallback observation due to unexpected isDone value.")
            self.current_obs = self._convert_obs(last_data)
            self._eval_episode_return = 0.
            return to_ndarray(self.current_obs)

        print("[ERROR] No valid reset data received after 20 attempts.")
        return None

    def step(self, action):
        assert isinstance(action, np.ndarray), type(action)
        action = action.item()

        self.server.send({"Order": int(action)})
        data = self.server.receive()
        self.step_count += 1

        if not data:
            return BaseEnvTimestep(to_ndarray(self.current_obs), to_ndarray([0.]), True, {})

        obs = self._convert_obs(data)
        reward = float(data.get("Reward", 0.)) - 0.01
        done = bool(data.get("isDone", False))

        if done:
            if obs[3] > 0:
                reward += self.win_reward
            else:
                reward -= 3.0

        self.current_obs = obs
        self._eval_episode_return += reward

        info = {}
        if done:
            info['eval_episode_return'] = self._eval_episode_return

        return BaseEnvTimestep(
            obs=to_ndarray(obs),
            reward=to_ndarray([reward], dtype=np.float32),
            done=done,
            info=info
        )

    def random_action(self) -> np.ndarray:
        random_action = self._action_space.sample()
        if isinstance(random_action, np.ndarray):
            return random_action
        elif isinstance(random_action, int):
            return to_ndarray([random_action], dtype=np.int64)
        elif isinstance(random_action, dict):
            return to_ndarray(random_action)
        else:
            raise TypeError(
                f"random_action should return int/np.ndarray or dict, but got {type(random_action)}: {random_action}"
            )

    def close(self):
        self.server.close()

    def _convert_obs(self, data):
        def r(x): return round(float(x), 2)

        # 자가 위치 및 체력
        obs = [r(data.get("X", 0)), r(data.get("Y", 0)), r(data.get("Z", 0)), data.get("HP", 0)]

        # 감지된 상대
        detect_list = data.get("DetectActors", [])
        for i in range(self.num_detectable):
            if i < len(detect_list):
                d = detect_list[i]
                obs += [r(d.get("X", 0)), r(d.get("Y", 0)), r(d.get("Z", 0)), d.get("HP", 0)]
            else:
                obs += [0, 0, 0, 0]

        obs = np.array(obs, dtype=np.float32)

        # ✅ 정규화 적용
        # 위치 및 거리 관련 값은 map_size로 정규화 [-1, 1]
        obs[:3] = obs[:3] / self.map_size
        for i in range(self.num_detectable):
            idx = 4 + i * 4  # 시작 인덱스
            obs[idx:idx+3] = obs[idx:idx+3] / self.map_size

        # HP 관련 값은 [0, 100] 범위 가정 → [0, 1]로 정규화
        obs[3] = obs[3] / 100.0
        for i in range(self.num_detectable):
            idx = 4 + i * 4 + 3  # 상대 HP 위치
            obs[idx] = obs[idx] / 100.0

        return np.clip(obs, -1.0, 1.0)

    def seed(self, seed: int = 0, dynamic_seed: bool = False) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def __repr__(self):
        return "SocketEnv()"

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def reward_space(self):
        return spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)
