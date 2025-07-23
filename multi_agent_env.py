from socket_server import SocketServer
from ding.envs import BaseEnv
from ding.utils import ENV_REGISTRY
from ding.torch_utils.data_helper import to_ndarray
from gym import spaces
import numpy as np
import time

@ENV_REGISTRY.register('socket_multi_env')
class SocketMultiAgentEnv(BaseEnv):
    def __init__(self, cfg):
        self._cfg = cfg
        self.max_step = cfg.get('max_step', 1000)
        self.map_size = cfg.get('map_size', 4000)
        self.win_reward = cfg.get('win_reward', 100)
        self.num_detectable = cfg.get('num_detectable', 4)
        self.num_agents = cfg.get('num_agents', 2)
        self.agent_ids = ["AgentCharacter", "AgentCharacter2"]

        obs_dim = 4 + self.num_detectable * 4
        self._observation_space = spaces.Box(low=-self.map_size, high=self.map_size, shape=(obs_dim,), dtype=np.float32)
        self._action_space = spaces.Discrete(6)

    def reset(self):
        print("=== [RESET CALL] ===")
        self.server = SocketServer()
        self.server.start()
        self.step_count = 0
        self.last_done = {aid: False for aid in self.agent_ids}

        self.server.send({
            "Agents": [{"UnitID": aid, "Order": 0, "Pitch": 0, "Yaw": 0, "Roll": 0} for aid in self.agent_ids]
        })

        for attempt in range(20):
            data = self.server.receive()
            if data and "Agents" in data:
                try:
                    obs_dict = {}
                    for agent in data["Agents"]:
                        aid = agent["UnitID"]
                        obs_dict[aid] = to_ndarray(self._convert_obs(agent))
                    self.current_obs = obs_dict
                    self._eval_episode_return = {aid: 0. for aid in self.agent_ids}
                    return obs_dict 
                except Exception as e:
                    print(f"[RESET ERROR] Parsing failed: {e} (attempt {attempt+1})")
            time.sleep(0.1)

        print("[ERROR] No valid data received after 20 attempts.")
        return {aid: to_ndarray(np.zeros(self._observation_space.shape)) for aid in self.agent_ids}

    def step(self, action: dict):
        assert isinstance(action, dict)

        #죽은 에이전트 행동을 항상 0으로 보냄
        self.server.send({
            "Agents": [
                {
                    "UnitID": aid,
                    "Order": 0 if self.last_done.get(aid, False) else int(action[aid].item()),
                    "Pitch": 0,
                    "Yaw": 0,
                    "Roll": 0
                } for aid in self.agent_ids
            ]
        })

        data = self.server.receive()

        if isinstance(data, str) and data.strip() == "EpiDone":
            print("[INFO] EpiDone received. Forcing episode termination.")
            return self._dummy_timestep(done_all=True)

        if not isinstance(data, dict) or "Agents" not in data:
            print("[ERROR] No valid agent data received.")
            return self._dummy_timestep(done_all=True)

        obs, rew, done, info = {}, {}, {}, {}

        for agent in data["Agents"]:
            aid = agent["UnitID"]

            #에이전트 사망 판정
            is_dead = bool(agent.get("isDone", False)) or agent.get("HP", 0) <= 0
            self.last_done[aid] = is_dead

            #관측값 마스킹
            obs_val = self._convert_obs(agent)
            if is_dead:
                obs[aid] = to_ndarray(np.full_like(obs_val, -1.0, dtype=np.float32))
            else:
                obs[aid] = to_ndarray(obs_val)

            #보상, done 처
            reward = float(agent.get("Reward", 0.))
            rew[aid] = reward
            done[aid] = is_dead
            info[aid] = {}
            self._eval_episode_return[aid] += reward

        self.current_obs = obs
        done["__all__"] = all(done[aid] for aid in self.agent_ids)

        if done["__all__"]:
            alive_agents = []
            for agent in data["Agents"]:
                aid = agent["UnitID"]
                if float(agent.get("HP", 0)) > 0:
                    alive_agents.append(aid)

            if len(alive_agents) > 0:
                for aid in self.agent_ids:
                    rew[aid] += self.win_reward
                print("[ENV] Win reward granted to all agents.")

        return {
            'obs': obs,
            'reward': rew,
            'done': done,
            'info': info
        }


    def _dummy_timestep(self, done_all: bool = False):
        dummy_obs = {aid: to_ndarray(self.current_obs.get(aid, np.zeros(self._observation_space.shape)))
                     for aid in self.agent_ids}
        rew = {aid: 0.0 for aid in self.agent_ids}
        done = {aid: done_all for aid in self.agent_ids}
        done["__all__"] = done_all
        info = {aid: {} for aid in self.agent_ids}


        return {
            'obs': dummy_obs,
            'reward': rew,
            'done': done,
            'info': info
        }

    def _convert_obs(self, data):
        self_x = float(data.get("LocX", 0))
        self_y = float(data.get("LocY", 0))
        self_z = float(data.get("LocZ", 0))
        hp = data.get("HP", 0)

        obs = [self_x / 4000, self_y / 4000, self_z / 400, hp / 100]
        fixed_enemy_names = [f"AgentCharacter{i+3}" for i in range(self.num_detectable)]
        obs += [0.0, 0.0, 0.0, 0.0] * len(fixed_enemy_names)

        for enemy in data.get("DetectActors", []):
            name = enemy.get("Name", "")
            if name in fixed_enemy_names:
                idx = fixed_enemy_names.index(name)
                base = 4 + idx * 4
                ex = float(enemy.get("LocX", 0)) / 4000
                ey = float(enemy.get("LocY", 0)) / 4000
                ez = float(enemy.get("LocZ", 0)) / 400
                obs[base:base+4] = [ex, ey, ez, 1.0]

        return np.round(np.array(obs, dtype=np.float32), 3)

    def close(self):
        self.server.close()

    def seed(self, seed: int = 0, dynamic_seed: bool = False) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def random_action(self) -> dict:
        return {aid: np.array([self._action_space.sample()], dtype=np.int64) for aid in self.agent_ids}

    def __repr__(self):
        return "SocketMultiAgentEnv()"

    @property
    def observation_space(self):
        return {aid: self._observation_space for aid in self.agent_ids}

    @property
    def action_space(self):
        return {aid: self._action_space for aid in self.agent_ids}

    @property
    def reward_space(self):
        return {aid: spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32) for aid in self.agent_ids}
