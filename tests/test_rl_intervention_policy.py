import unittest
from collections import deque

import numpy as np

from rl_agent_4 import (
    ACTION_COST_ALL_SLA_MET_EXTRA,
    ACTION_COST_CLEAR_VIOLATION,
    ACTION_COST_EXACT_REPEAT_EXTRA,
    ACTION_COST_PERSISTENT_BUILDUP,
    ACTION_COST_RECOVERING,
    ACTION_COST_RECOVERING_REPEAT_EXTRA,
    ACTION_COST_SAME_QUEUE_REPEAT_EXTRA,
    ACTION_COST_STABLE,
    ACTION_DIM,
    DQNAgent,
    DROP_CAP,
    EPS_END,
    EPS_START,
    QIDS,
    QUEUE_FEATURE_DROP_NORM,
    QUEUE_FEATURE_LAT_RATIO,
    QUEUE_FEATURE_STRIDE,
    QUEUE_FEATURE_UTIL_NORM,
    RAW_STATE_DIM,
    SLA_THRESHOLDS,
    STACK_SIZE,
    STATE_DIM,
    TrainingArtifactLogger,
    QoSRoutingEnv,
)


class InterventionPolicyTests(unittest.TestCase):
    @staticmethod
    def _env():
        env = object.__new__(QoSRoutingEnv)
        env.current_traffic_profile = "unit"
        env.global_step = 20
        env._last_action_global_step = 10
        env.last_action_time = 0.0
        env.frame_stack = deque(maxlen=STACK_SIZE)
        env.action_stack = deque(maxlen=STACK_SIZE)
        return env

    @staticmethod
    def _frame(ratios=None, drops=None, utils=None):
        ratios = ratios or {}
        drops = drops or {}
        utils = utils or {}
        frame = np.zeros(RAW_STATE_DIM, dtype=np.float32)
        for qid in QIDS:
            base = tuple(QIDS).index(qid) * QUEUE_FEATURE_STRIDE
            ratio = float(ratios.get(qid, 0.60))
            drop_norm = float(drops.get(qid, 0.0))
            util_norm = float(utils.get(qid, 0.20))
            frame[base + QUEUE_FEATURE_LAT_RATIO] = ratio
            frame[base + QUEUE_FEATURE_DROP_NORM] = drop_norm
            frame[base + QUEUE_FEATURE_UTIL_NORM] = util_norm
            frame[base + 3] = 1.0 if ratio <= 1.0 else 0.0
        return frame

    def _set_qid_series(self, env, qid, ratios, drops=None, utils=None):
        drops = drops or [0.0] * len(ratios)
        utils = utils or [0.20] * len(ratios)
        env.frame_stack.clear()
        for ratio, drop, util in zip(ratios, drops, utils):
            env.frame_stack.append(
                self._frame(
                    ratios={qid: ratio},
                    drops={qid: drop},
                    utils={qid: util},
                )
            )
        while len(env.frame_stack) < STACK_SIZE:
            env.frame_stack.appendleft(self._frame())

    @staticmethod
    def _set_actions(env, actions):
        env.action_stack.clear()
        for _ in range(STACK_SIZE):
            env.action_stack.append(env._action_to_onehot(0))
        for action in actions:
            env.action_stack.append(env._action_to_onehot(action))

    @staticmethod
    def _snapshot(ratios=None, drops=None, utils=None, with_context=True):
        ratios = ratios or {}
        drops = drops or {}
        utils = utils or {}
        snapshot = {}
        for qid in QIDS:
            ratio = float(ratios.get(qid, 0.60))
            drop_norm = float(drops.get(qid, 0.0))
            util_norm = float(utils.get(qid, 0.20))
            snapshot[qid] = {
                "lat_p95": SLA_THRESHOLDS[qid] * ratio,
                "drop_p95": DROP_CAP * drop_norm,
                "util_p95": 100.0 * util_norm,
                "data_valid": True,
                "hot_src_ip": f"10.0.{qid}.1",
                "hot_dst_ip": f"10.0.{qid}.2",
                "bottleneck_sid": qid + 10 if with_context else None,
                "alternatives": [{"name": "alt0"}, {"name": "alt1"}],
            }
        return snapshot

    def test_state_dimension_unchanged(self):
        self.assertEqual(STATE_DIM, (RAW_STATE_DIM * STACK_SIZE) + (ACTION_DIM * STACK_SIZE))

    def test_agent_uses_instance_epsilon_decay_window(self):
        agent = DQNAgent(
            state_dim=4,
            action_dim=2,
            device="cpu",
            eps_decay_steps=10,
        )

        for _ in range(5):
            agent.update_epsilon()

        expected = EPS_END + (EPS_START - EPS_END) * 0.5
        self.assertAlmostEqual(agent.eps, expected)

    def test_resume_epsilon_maps_to_stage_decay_window(self):
        agent = DQNAgent(
            state_dim=4,
            action_dim=2,
            device="cpu",
            eps_decay_steps=20_000,
        )

        agent.set_epsilon_for_decay(0.50)

        self.assertEqual(agent.eps_step_count, 0)
        self.assertAlmostEqual(agent.eps, 0.50)
        self.assertAlmostEqual(agent.eps_decay_start, 0.50)

        for _ in range(20_000):
            agent.update_epsilon()

        self.assertAlmostEqual(agent.eps, EPS_END)

    def test_context_classifies_stable_stack(self):
        env = self._env()
        self._set_qid_series(env, 0, [0.60] * STACK_SIZE)

        ctx = env._queue_intervention_context(0, self._snapshot({0: 0.60}))

        self.assertEqual(ctx["intervention_context"], "stable")

    def test_context_classifies_one_frame_spike(self):
        env = self._env()
        self._set_qid_series(env, 0, [0.60] * STACK_SIZE)

        ctx = env._queue_intervention_context(0, self._snapshot({0: 0.98}))

        self.assertEqual(ctx["intervention_context"], "short_spike")
        self.assertEqual(ctx["target_persistence_5"], 1)

    def test_context_classifies_persistent_buildup(self):
        env = self._env()
        self._set_qid_series(env, 0, [0.82, 0.88, 0.91, 0.96, 1.02, 1.06])

        ctx = env._queue_intervention_context(0, self._snapshot({0: 1.06}))

        self.assertEqual(ctx["intervention_context"], "persistent_buildup")
        self.assertGreater(ctx["target_slope_5"], 0.0)

    def test_context_classifies_clear_violation(self):
        env = self._env()
        self._set_qid_series(env, 0, [0.60] * STACK_SIZE)

        ctx = env._queue_intervention_context(0, self._snapshot({0: 1.12}))

        self.assertEqual(ctx["intervention_context"], "clear_violation")

    def test_network_classifies_two_clear_violations_as_severe(self):
        env = self._env()
        self._set_qid_series(env, 0, [0.60] * STACK_SIZE)

        ctx = env._network_intervention_context(
            self._snapshot({0: 1.20, 1: 1.18, 7: 0.70})
        )

        self.assertTrue(ctx["severe_multi_queue"])
        self.assertEqual(ctx["network_intervention_context"], "severe_multi_queue")

    def test_persistent_buildup_gets_medium_cost(self):
        env = self._env()
        self._set_actions(env, [])
        self._set_qid_series(env, 0, [0.82, 0.88, 0.91, 0.96, 1.02, 1.06])

        info = env._contextual_action_cost_info(1, self._snapshot({0: 1.06}))

        self.assertEqual(info["intervention_context"], "persistent_buildup")
        self.assertAlmostEqual(info["final_action_cost"], ACTION_COST_PERSISTENT_BUILDUP)

    def test_clear_violation_gets_low_cost(self):
        env = self._env()
        self._set_actions(env, [])
        self._set_qid_series(env, 0, [0.60] * STACK_SIZE)

        info = env._contextual_action_cost_info(1, self._snapshot({0: 1.20}))

        self.assertEqual(info["intervention_context"], "clear_violation")
        self.assertAlmostEqual(info["final_action_cost"], ACTION_COST_CLEAR_VIOLATION)

    def test_all_sla_reroute_gets_extra_penalty_unless_persistent(self):
        env = self._env()
        self._set_actions(env, [])
        self._set_qid_series(env, 0, [0.60] * STACK_SIZE)

        stable = env._contextual_action_cost_info(1, self._snapshot({0: 0.60}))

        self.assertEqual(stable["intervention_context"], "stable")
        self.assertAlmostEqual(
            stable["final_action_cost"],
            ACTION_COST_STABLE + ACTION_COST_ALL_SLA_MET_EXTRA,
        )

        self._set_qid_series(env, 0, [0.82, 0.88, 0.91, 0.96, 1.02, 1.06])
        persistent = env._contextual_action_cost_info(1, self._snapshot({0: 1.06}))

        self.assertEqual(persistent["intervention_context"], "persistent_buildup")
        self.assertEqual(persistent["all_sla_penalty"], 0.0)

    def test_same_queue_repeat_is_not_penalized_in_clear_violation(self):
        env = self._env()
        self._set_actions(env, [1])
        self._set_qid_series(env, 0, [0.60] * STACK_SIZE)

        info = env._contextual_action_cost_info(1, self._snapshot({0: 1.20}))

        self.assertTrue(info["same_queue_repeat"])
        self.assertEqual(info["repeat_penalty"], 0.0)
        self.assertAlmostEqual(info["final_action_cost"], ACTION_COST_CLEAR_VIOLATION)

    def test_same_queue_repeat_gets_only_small_penalty_while_recovering(self):
        env = self._env()
        self._set_actions(env, [1])
        self._set_qid_series(env, 0, [0.80, 1.24, 1.18, 1.04])

        info = env._contextual_action_cost_info(1, self._snapshot({0: 1.04}))

        self.assertEqual(info["intervention_context"], "recovering")
        self.assertTrue(info["same_queue_repeat"])
        self.assertAlmostEqual(info["repeat_penalty"], ACTION_COST_RECOVERING_REPEAT_EXTRA)
        self.assertAlmostEqual(
            info["final_action_cost"],
            ACTION_COST_RECOVERING + ACTION_COST_RECOVERING_REPEAT_EXTRA,
        )

    def test_same_queue_repeat_is_penalized_in_stable_context(self):
        env = self._env()
        self._set_actions(env, [2])
        self._set_qid_series(env, 0, [0.60] * STACK_SIZE)

        info = env._contextual_action_cost_info(1, self._snapshot({0: 0.60}))

        self.assertTrue(info["same_queue_repeat"])
        self.assertFalse(info["exact_action_repeat"])
        self.assertAlmostEqual(info["repeat_penalty"], ACTION_COST_SAME_QUEUE_REPEAT_EXTRA)

    def test_exact_repeat_is_penalized_more_than_same_queue_repeat(self):
        env = self._env()
        self._set_actions(env, [1])
        self._set_qid_series(env, 0, [0.60] * STACK_SIZE)

        info = env._contextual_action_cost_info(1, self._snapshot({0: 0.60}))

        self.assertTrue(info["exact_action_repeat"])
        self.assertAlmostEqual(info["repeat_penalty"], ACTION_COST_EXACT_REPEAT_EXTRA)

    def test_single_reroutes_remain_valid_in_stable_context(self):
        env = self._env()

        mask = env._get_valid_actions(self._snapshot({0: 0.60, 1: 0.60, 7: 0.60}))

        self.assertTrue(mask[0])
        for action in range(1, 7):
            self.assertTrue(mask[action])
        self.assertFalse(mask[7])

    def test_step_csv_header_includes_intervention_fields(self):
        for field in (
            "intervention_context",
            "target_ratio",
            "target_persistence_3",
            "target_slope_3",
            "target_drop_trend",
            "same_queue_repeat",
            "repeat_penalty",
            "all_sla_penalty",
            "final_action_cost",
            "outcome_class",
            "outcome_shaping",
        ):
            self.assertIn(field, TrainingArtifactLogger.STEP_FIELDS)


if __name__ == "__main__":
    unittest.main()
