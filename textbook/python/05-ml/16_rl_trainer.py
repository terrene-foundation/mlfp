# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — ML / RL Trainer
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Train reinforcement learning agents using RLTrainer with
#            EnvironmentRegistry and PolicyRegistry.  Covers environment
#            registration, policy configuration, training, evaluation,
#            and custom Gymnasium environments.
# LEVEL: Advanced
# PARITY: Python-only (kailash-ml has no Rust counterpart)
# VALIDATES: RLTrainer, RLTrainingConfig, RLTrainingResult,
#            EnvironmentRegistry, EnvironmentSpec,
#            PolicyRegistry, PolicySpec, PolicyVersion
#
# NOTE: Running this tutorial requires `pip install kailash-ml[rl]`
#       (stable-baselines3, gymnasium).  Types and registries are
#       validated WITHOUT training when SB3 is not installed.
#
# Run: uv run python textbook/python/05-ml/16_rl_trainer.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from kailash_ml.rl.env_registry import EnvironmentRegistry, EnvironmentSpec
from kailash_ml.rl.policy_registry import PolicyRegistry, PolicySpec, PolicyVersion
from kailash_ml.rl.trainer import RLTrainer, RLTrainingConfig, RLTrainingResult

# ════════════════════════════════════════════════════════════════════════
# Part A: RLTrainingConfig
# ════════════════════════════════════════════════════════════════════════

# ── 1. RLTrainingConfig — defaults and customization ─────────────────

config = RLTrainingConfig()
assert config.algorithm == "PPO"
assert config.policy_type == "MlpPolicy"
assert config.total_timesteps == 100_000
assert config.n_eval_episodes == 10
assert config.eval_freq == 10_000
assert config.seed == 42
assert config.verbose == 0

# Custom config
custom_config = RLTrainingConfig(
    algorithm="DQN",
    policy_type="MlpPolicy",
    total_timesteps=50_000,
    hyperparameters={"learning_rate": 0.001, "buffer_size": 10000},
    n_eval_episodes=5,
    seed=123,
)

assert custom_config.algorithm == "DQN"
assert custom_config.total_timesteps == 50_000
assert custom_config.hyperparameters["learning_rate"] == 0.001

# Serialization
cfg_dict = custom_config.to_dict()
assert cfg_dict["algorithm"] == "DQN"
assert cfg_dict["total_timesteps"] == 50_000
assert cfg_dict["seed"] == 123

# ════════════════════════════════════════════════════════════════════════
# Part B: PolicyRegistry
# ════════════════════════════════════════════════════════════════════════

# ── 2. PolicyRegistry — register specs and versions ──────────────────

policy_reg = PolicyRegistry(root_dir=tempfile.mkdtemp())
assert isinstance(policy_reg, PolicyRegistry)

# Supported algorithms
supported = PolicyRegistry.supported_algorithms()
assert "PPO" in supported
assert "SAC" in supported
assert "DQN" in supported
assert "A2C" in supported
assert "TD3" in supported
assert "DDPG" in supported

# ── 3. Register a policy spec ────────────────────────────────────────

ppo_spec = PolicySpec(
    name="cartpole_ppo",
    algorithm="PPO",
    policy_type="MlpPolicy",
    hyperparameters={"learning_rate": 0.0003},
    description="PPO agent for CartPole-v1",
)

policy_reg.register_spec(ppo_spec)
assert "cartpole_ppo" in policy_reg
assert len(policy_reg) == 1

# Retrieve spec
retrieved_spec = policy_reg.get_spec("cartpole_ppo")
assert retrieved_spec is not None
assert retrieved_spec.algorithm == "PPO"
assert retrieved_spec.description == "PPO agent for CartPole-v1"

# List specs
specs = policy_reg.list_specs()
assert len(specs) == 1
assert specs[0].name == "cartpole_ppo"

# Serialization
spec_dict = ppo_spec.to_dict()
assert spec_dict["name"] == "cartpole_ppo"
assert spec_dict["algorithm"] == "PPO"

# ── 4. Register policy versions (trained artifacts) ──────────────────

v1 = PolicyVersion(
    name="cartpole_ppo",
    version=1,
    algorithm="PPO",
    artifact_path="/tmp/models/cartpole_ppo_v1",
    mean_reward=200.0,
    std_reward=15.0,
    total_timesteps=50_000,
    metadata={"learning_rate": 0.0003},
)

policy_reg.register_version(v1)

v2 = PolicyVersion(
    name="cartpole_ppo",
    version=2,
    algorithm="PPO",
    artifact_path="/tmp/models/cartpole_ppo_v2",
    mean_reward=450.0,
    std_reward=10.0,
    total_timesteps=100_000,
    metadata={"learning_rate": 0.0001},
)

policy_reg.register_version(v2)

# List versions
versions = policy_reg.list_versions("cartpole_ppo")
assert len(versions) == 2

# Get latest version
latest = policy_reg.get_latest_version("cartpole_ppo")
assert latest is not None
assert latest.version == 2
assert latest.mean_reward == 450.0

# Get specific version
specific = policy_reg.get_version("cartpole_ppo", 1)
assert specific is not None
assert specific.version == 1
assert specific.mean_reward == 200.0

# Non-existent version
missing = policy_reg.get_version("cartpole_ppo", 99)
assert missing is None

# PolicyVersion serialization
pv_dict = v2.to_dict()
assert pv_dict["name"] == "cartpole_ppo"
assert pv_dict["version"] == 2
assert pv_dict["mean_reward"] == 450.0

# ── 5. Edge case: invalid algorithm ──────────────────────────────────

try:
    policy_reg.register_spec(PolicySpec(name="bad", algorithm="INVALID"))
    assert False, "Should reject unknown algorithm"
except ValueError as e:
    assert "unknown algorithm" in str(e).lower()

# ════════════════════════════════════════════════════════════════════════
# Part C: EnvironmentRegistry
# ════════════════════════════════════════════════════════════════════════

# ── 6. EnvironmentSpec — environment configuration ───────────────────

env_spec = EnvironmentSpec(
    name="CustomGrid-v0",
    entry_point="gymnasium.envs.classic_control:CartPoleEnv",
    kwargs={},
    max_episode_steps=500,
    reward_threshold=475.0,
    description="Custom grid environment for testing",
)

assert env_spec.name == "CustomGrid-v0"
assert env_spec.max_episode_steps == 500

spec_dict = env_spec.to_dict()
assert spec_dict["name"] == "CustomGrid-v0"
assert spec_dict["max_episode_steps"] == 500

# ── 7. EnvironmentRegistry — manages environments ────────────────────
# Registration requires gymnasium.  We test the registry structure
# without requiring the full dependency.

try:
    import gymnasium  # noqa: F401

    env_reg = EnvironmentRegistry()
    assert isinstance(env_reg, EnvironmentRegistry)
    assert len(env_reg) == 0

    # Register a spec (uses Gymnasium registration)
    env_reg.register(env_spec)
    assert "CustomGrid-v0" in env_reg
    assert len(env_reg) == 1

    # List environments
    envs = env_reg.list_environments()
    assert len(envs) == 1
    assert envs[0].name == "CustomGrid-v0"

    # Get spec
    retrieved_env = env_reg.get_spec("CustomGrid-v0")
    assert retrieved_env is not None
    assert retrieved_env.description == "Custom grid environment for testing"

    gymnasium_available = True
except ImportError:
    gymnasium_available = False

# ════════════════════════════════════════════════════════════════════════
# Part D: RLTrainer
# ════════════════════════════════════════════════════════════════════════

# ── 8. RLTrainer — instantiation ─────────────────────────────────────

with tempfile.TemporaryDirectory() as rl_dir:
    trainer = RLTrainer(
        env_registry=env_reg if gymnasium_available else None,
        policy_registry=policy_reg,
        root_dir=rl_dir,
    )
    assert isinstance(trainer, RLTrainer)

    # Supported algorithms (static method)
    algos = RLTrainer.supported_algorithms()
    assert "PPO" in algos
    assert "DQN" in algos

    # ── 9. RLTrainer.train() — requires SB3 ─────────────────────────
    # Full training requires stable-baselines3 and gymnasium.
    # We demonstrate the API pattern and test with minimal timesteps
    # if available.

    try:
        import stable_baselines3  # noqa: F401

        sb3_available = True
    except ImportError:
        sb3_available = False

    if sb3_available and gymnasium_available:
        # Train with minimal timesteps for speed
        train_config = RLTrainingConfig(
            algorithm="PPO",
            policy_type="MlpPolicy",
            total_timesteps=256,  # Minimal for tutorial
            n_eval_episodes=2,
            seed=42,
            verbose=0,
            save_path=Path(rl_dir) / "test_policy",
        )

        result = trainer.train(
            env_name="CartPole-v1",
            policy_name="tutorial_cartpole",
            config=train_config,
        )

        assert isinstance(result, RLTrainingResult)
        assert result.policy_name == "tutorial_cartpole"
        assert result.algorithm == "PPO"
        assert result.total_timesteps == 256
        assert isinstance(result.mean_reward, float)
        assert isinstance(result.std_reward, float)
        assert result.training_time_seconds > 0
        assert result.artifact_path is not None

        # Serialization
        result_dict = result.to_dict()
        assert result_dict["policy_name"] == "tutorial_cartpole"
        assert result_dict["algorithm"] == "PPO"

        # The trained policy was registered in policy_reg
        trained_versions = policy_reg.list_versions("tutorial_cartpole")
        assert len(trained_versions) >= 1

        # ── 10. Evaluate a trained model ──────────────────────────────

        # Load from policy registry and evaluate
        mean_r, std_r = trainer.load_and_evaluate(
            "tutorial_cartpole",
            env_name="CartPole-v1",
            n_episodes=2,
        )
        assert isinstance(mean_r, float)
        assert isinstance(std_r, float)

    else:
        # Verify that missing SB3 gives a clear ImportError
        try:
            from kailash_ml.rl.trainer import _import_algo

            _import_algo("PPO")
            if not sb3_available:
                assert False, "Should raise ImportError without SB3"
        except ImportError as e:
            assert "stable-baselines3" in str(e)

# ── 11. RLTrainingResult — standalone creation ────────────────────────

manual_result = RLTrainingResult(
    policy_name="demo_policy",
    algorithm="SAC",
    total_timesteps=10_000,
    mean_reward=300.0,
    std_reward=20.0,
    training_time_seconds=45.2,
    artifact_path="/tmp/demo_policy/model",
)

assert manual_result.mean_reward == 300.0
result_dict = manual_result.to_dict()
assert result_dict["algorithm"] == "SAC"

print("PASS: 05-ml/16_rl_trainer")
