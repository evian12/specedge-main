import json
import os

import torch

import util


class _ConfigMeta(type):
    _initialized = False

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        return cls

    def __getattr__(cls, name):
        if name.startswith("_"):
            return super().__getattribute__(name)

        if not cls._initialized:
            cls._initialize()

        if name in cls.__dict__:
            return cls.__dict__[name]
        else:
            raise AttributeError(f"'{cls.__name__}' has no attribute '{name}'")

    def __setattr__(cls, name, value):
        super().__setattr__(name, value)

    def _initialize(cls):
        raise NotImplementedError("Subclasses must implement _initialize()")

    def _from_env(cls, key: str):
        value = os.getenv(key)

        if value is None or value == "null":
            raise ValueError(f"Environment variable '{key}' is not set")

        return value

    def _from_env_default(cls, key: str, default: str):
        value = os.getenv(key)
        return default if value is None or value == "null" else value

    def reset(cls):
        cls._initialized = False
        cls._initialize()


class SpecEdgeClientConfig(metaclass=_ConfigMeta):
    """
    Configuration for the SpecEdge client

    Results and logs are stored in the directory
    "result_path/exp_name/process_name/seed"

    Attributes:
        optimization (int): Optimization level for the model
        result_path (str): Path to the directory where the results will be stored
        exp_name (str): Name of the experiment
        process_name (str): Name of the process

        seed (int): Seed for the random number generator

        draft_model (str): Path to the draft model
        device (torch.device): Device to run the model on
        dtype (torch.dtype): Data type to use for the model

        dataset (str): Name of the dataset

        max_n_beams (int): Maximum number of beams to generate
        max_beam_len (int): Maximum length of a beam
        max_branch_width (int): Maximum width of a branch
        max_budget (int): Maximum budget for the SpecExec algorithm

        proactive_type (str): Type of proactive draft
        proactive_max_n_beams (int): Maximum number of beams to generate proactively
        proactive_max_beam_len (int): Maximum length of a beam for proactive draft
        proactive_max_branch_width (int): Maximum width of a branch for proactive draft
        proactive_max_budget (int): Maximum budget for the proactive draft

        max_new_tokens (int): Maximum number of new tokens to generate
        max_request_num (int): Maximum number of requests to send

        host (str): Hostname of the server
        req_idx (int): Index of the request
    """

    @classmethod
    def _initialize(cls):
        # experiment configuration
        cls.result_path = cls._from_env("SPECEDGE_RESULT_PATH")
        cls.exp_name = cls._from_env("SPECEDGE_EXP_NAME")
        cls.process_name = cls._from_env("SPECEDGE_PROCESS_NAME")
        cls.seed = int(cls._from_env("SPECEDGE_SEED"))
        cls.optimization = int(cls._from_env("SPECEDGE_OPTIMIZATION"))
        cls.max_len = int(cls._from_env("SPECEDGE_MAX_LEN"))

        # model configuration
        cls.draft_model = cls._from_env("SPECEDGE_DRAFT_MODEL")
        cls.target_model = cls._from_env("SPECEDGE_TARGET_MODEL")
        cls.device = torch.device(cls._from_env("SPECEDGE_DEVICE"))
        cls.dtype = util.convert_dtype(cls._from_env("SPECEDGE_DTYPE"))
        cls.reasoning = cls._from_env("SPECEDGE_REASONING") == "True"

        # dataset configuration
        cls.dataset = cls._from_env("SPECEDGE_DATASET")

        # SpecExec configuration
        cls.max_n_beams = int(cls._from_env("SPECEDGE_MAX_N_BEAMS"))
        cls.max_beam_len = int(cls._from_env("SPECEDGE_MAX_BEAM_LEN"))
        cls.max_branch_width = int(cls._from_env("SPECEDGE_MAX_BRANCH_WIDTH"))
        cls.max_budget = int(cls._from_env("SPECEDGE_MAX_BUDGET"))

        # initial draft depth configuration
        cls.initial_draft_mode = cls._from_env_default(
            "SPECEDGE_INITIAL_DRAFT_MODE", "fixed"
        )
        cls.initial_draft_structure = cls._from_env_default(
            "SPECEDGE_INITIAL_DRAFT_STRUCTURE", "tree"
        )
        cls.initial_draft_candidate_depths = [
            int(value)
            for value in json.loads(
                cls._from_env_default(
                    "SPECEDGE_INITIAL_DRAFT_CANDIDATE_DEPTHS",
                    f"[{cls.max_beam_len}]",
                )
            )
        ]
        cls.initial_draft_warmup_per_depth = int(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_WARMUP_PER_DEPTH", "4"
            )
        )
        cls.initial_draft_exploration_weight = float(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_EXPLORATION_WEIGHT", "0.3"
            )
        )
        cls.initial_draft_forced_exploration_interval = int(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_FORCED_EXPLORATION_INTERVAL", "32"
            )
        )
        cls.initial_draft_ridge_lambda = float(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_RIDGE_LAMBDA", "1.0"
            )
        )
        cls.initial_draft_reward_clip = float(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_REWARD_CLIP", "20.0"
            )
        )
        cls.initial_draft_ewma_alpha = float(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_EWMA_ALPHA", "0.2"
            )
        )
        cls.initial_draft_local_initial_depth = int(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_LOCAL_INITIAL_DEPTH",
                str(cls.max_beam_len),
            )
        )
        cls.initial_draft_local_min_depth = int(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_LOCAL_MIN_DEPTH",
                "1",
            )
        )
        cls.initial_draft_local_max_depth = int(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_LOCAL_MAX_DEPTH",
                str(cls.max_beam_len),
            )
        )
        cls.initial_draft_local_increase_streak = int(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_LOCAL_INCREASE_STREAK",
                "2",
            )
        )
        cls.initial_draft_local_decrease_streak = int(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_LOCAL_DECREASE_STREAK",
                "2",
            )
        )
        cls.initial_draft_local_high_score = float(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_LOCAL_HIGH_SCORE",
                "2.0",
            )
        )
        cls.initial_draft_local_low_penalty = float(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_LOCAL_LOW_PENALTY",
                "1.0",
            )
        )
        cls.initial_draft_local_increase_score_threshold = float(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_LOCAL_INCREASE_SCORE_THRESHOLD",
                "3.0",
            )
        )
        cls.initial_draft_local_decrease_score_threshold = float(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_LOCAL_DECREASE_SCORE_THRESHOLD",
                "3.0",
            )
        )
        cls.initial_draft_local_protect_window = int(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_LOCAL_PROTECT_WINDOW",
                "5",
            )
        )
        cls.initial_draft_local_protect_avg_accepted_depth = float(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_LOCAL_PROTECT_AVG_ACCEPTED_DEPTH",
                "2.0",
            )
        )
        cls.initial_draft_local_neutral_score_decay = float(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_LOCAL_NEUTRAL_SCORE_DECAY",
                "0.8",
            )
        )
        cls.initial_draft_local_state_window_size = int(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_LOCAL_STATE_WINDOW_SIZE",
                "5",
            )
        )
        cls.initial_draft_local_very_slow_depth = int(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_LOCAL_VERY_SLOW_DEPTH",
                "1",
            )
        )
        cls.initial_draft_local_slow_depth = int(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_LOCAL_SLOW_DEPTH",
                "2",
            )
        )
        cls.initial_draft_local_mid_depth = int(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_LOCAL_MID_DEPTH",
                "3",
            )
        )
        cls.initial_draft_local_fast_depth = int(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_LOCAL_FAST_DEPTH",
                "4",
            )
        )
        cls.initial_draft_local_very_slow_accept_threshold = float(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_LOCAL_VERY_SLOW_ACCEPT_THRESHOLD",
                "1.2",
            )
        )
        cls.initial_draft_local_very_slow_depth_threshold = float(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_LOCAL_VERY_SLOW_DEPTH_THRESHOLD",
                "0.2",
            )
        )
        cls.initial_draft_local_very_slow_exit_accept_threshold = float(
            cls._from_env_default(
                (
                    "SPECEDGE_INITIAL_DRAFT_LOCAL_"
                    "VERY_SLOW_EXIT_ACCEPT_THRESHOLD"
                ),
                "1.4",
            )
        )
        cls.initial_draft_local_enter_very_slow_votes = int(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_LOCAL_ENTER_VERY_SLOW_VOTES",
                "2",
            )
        )
        cls.initial_draft_local_fast_accept_threshold = float(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_LOCAL_FAST_ACCEPT_THRESHOLD",
                "2.4",
            )
        )
        cls.initial_draft_local_fast_depth_threshold = float(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_LOCAL_FAST_DEPTH_THRESHOLD",
                "2.0",
            )
        )
        cls.initial_draft_local_slow_accept_threshold = float(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_LOCAL_SLOW_ACCEPT_THRESHOLD",
                "1.6",
            )
        )
        cls.initial_draft_local_slow_depth_threshold = float(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_LOCAL_SLOW_DEPTH_THRESHOLD",
                "0.6",
            )
        )
        cls.initial_draft_local_enter_fast_votes = int(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_LOCAL_ENTER_FAST_VOTES",
                "2",
            )
        )
        cls.initial_draft_local_enter_slow_votes = int(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_LOCAL_ENTER_SLOW_VOTES",
                "2",
            )
        )
        cls.initial_draft_local_fast_exit_accept_threshold = float(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_LOCAL_FAST_EXIT_ACCEPT_THRESHOLD",
                "2.0",
            )
        )
        cls.initial_draft_local_slow_exit_accept_threshold = float(
            cls._from_env_default(
                "SPECEDGE_INITIAL_DRAFT_LOCAL_SLOW_EXIT_ACCEPT_THRESHOLD",
                "1.8",
            )
        )

        # proactive draft configuration
        cls.proactive_type = cls._from_env("SPECEDGE_PROACTIVE_TYPE")
        cls.proactive_max_n_beams = int(cls._from_env("SPECEDGE_PROACTIVE_MAX_N_BEAMS"))
        cls.proactive_max_beam_len = int(
            cls._from_env("SPECEDGE_PROACTIVE_MAX_BEAM_LEN")
        )
        cls.proactive_max_branch_width = int(
            cls._from_env("SPECEDGE_PROACTIVE_MAX_BRANCH_WIDTH")
        )
        cls.proactive_max_budget = int(cls._from_env("SPECEDGE_PROACTIVE_MAX_BUDGET"))
        cls.proactive_mode = cls._from_env_default(
            "SPECEDGE_PROACTIVE_MODE", "baseline"
        )
        cls.proactive_path_policy = cls._from_env_default(
            "SPECEDGE_PROACTIVE_PATH_POLICY", "single_best"
        )
        cls.proactive_multi_max_deepest_leaves = int(
            cls._from_env_default(
                "SPECEDGE_PROACTIVE_MULTI_MAX_DEEPEST_LEAVES", "8"
            )
        )
        cls.proactive_multi_min_bonus_per_leaf = int(
            cls._from_env_default(
                "SPECEDGE_PROACTIVE_MULTI_MIN_BONUS_PER_LEAF", "1"
            )
        )
        cls.proactive_multi_max_bonus_per_leaf = int(
            cls._from_env_default(
                "SPECEDGE_PROACTIVE_MULTI_MAX_BONUS_PER_LEAF", "4"
            )
        )
        cls.proactive_multi_max_roots = int(
            cls._from_env_default(
                "SPECEDGE_PROACTIVE_MULTI_MAX_ROOTS", "8"
            )
        )
        cls.proactive_multi_min_root_probability = float(
            cls._from_env_default(
                "SPECEDGE_PROACTIVE_MULTI_MIN_ROOT_PROBABILITY", "0.01"
            )
        )
        cls.proactive_multi_leaf_temperature = float(
            cls._from_env_default(
                "SPECEDGE_PROACTIVE_MULTI_LEAF_TEMPERATURE", "1.0"
            )
        )
        cls.proactive_multi_full_depth_prior = float(
            cls._from_env_default(
                "SPECEDGE_PROACTIVE_MULTI_FULL_DEPTH_PRIOR", "0.5"
            )
        )
        cls.proactive_multi_acceptance_ewma_alpha = float(
            cls._from_env_default(
                "SPECEDGE_PROACTIVE_MULTI_ACCEPTANCE_EWMA_ALPHA", "0.1"
            )
        )
        cls.proactive_multi_depth_probability_coverage = [
            float(value)
            for value in json.loads(
                cls._from_env_default(
                    "SPECEDGE_PROACTIVE_MULTI_DEPTH_PROBABILITY_COVERAGE",
                    "[1.0, 0.8, 0.5]",
                )
            )
        ]
        cls.proactive_sequence_acceptance_survival = [
            float(value)
            for value in json.loads(
                cls._from_env_default(
                    "SPECEDGE_PROACTIVE_SEQUENCE_ACCEPTANCE_SURVIVAL",
                    "[1.0, 0.5, 0.3, 0.2, 0.1]",
                )
            )
        ]
        cls.proactive_sequence_max_bonus_per_depth = int(
            cls._from_env_default(
                "SPECEDGE_PROACTIVE_SEQUENCE_MAX_BONUS_PER_DEPTH",
                "4",
            )
        )
        cls.proactive_sequence_max_roots = int(
            cls._from_env_default(
                "SPECEDGE_PROACTIVE_SEQUENCE_MAX_ROOTS",
                "8",
            )
        )
        cls.proactive_sequence_min_root_probability = float(
            cls._from_env_default(
                "SPECEDGE_PROACTIVE_SEQUENCE_MIN_ROOT_PROBABILITY",
                "0.0",
            )
        )
        cls.proactive_sequence_min_bonus_probability = float(
            cls._from_env_default(
                "SPECEDGE_PROACTIVE_SEQUENCE_MIN_BONUS_PROBABILITY",
                "0.0",
            )
        )
        cls.proactive_sequence_selection_score = cls._from_env_default(
            "SPECEDGE_PROACTIVE_SEQUENCE_SELECTION_SCORE",
            "joint",
        )
        cls.proactive_sequence_reuse_depth_bonus = float(
            cls._from_env_default(
                "SPECEDGE_PROACTIVE_SEQUENCE_REUSE_DEPTH_BONUS",
                "0.0",
            )
        )
        cls.proactive_sequence_depth_probability_coverage = [
            float(value)
            for value in json.loads(
                cls._from_env_default(
                    "SPECEDGE_PROACTIVE_SEQUENCE_DEPTH_PROBABILITY_COVERAGE",
                    "[1.0, 0.8, 0.5]",
                )
            )
        ]
        cls.proactive_adaptive_ewma_alpha = float(
            cls._from_env_default("SPECEDGE_PROACTIVE_ADAPTIVE_EWMA_ALPHA", "0.2")
        )
        cls.proactive_adaptive_min_alignment_rate = float(
            cls._from_env_default(
                "SPECEDGE_PROACTIVE_ADAPTIVE_MIN_ALIGNMENT_RATE", "0.1"
            )
        )
        cls.proactive_adaptive_low_alignment_depth = int(
            cls._from_env_default(
                "SPECEDGE_PROACTIVE_ADAPTIVE_LOW_ALIGNMENT_DEPTH", "0"
            )
        )
        cls.proactive_adaptive_warmup_cycles = int(
            cls._from_env_default(
                "SPECEDGE_PROACTIVE_ADAPTIVE_WARMUP_CYCLES", "4"
            )
        )
        cls.proactive_adaptive_exploration_interval = int(
            cls._from_env_default(
                "SPECEDGE_PROACTIVE_ADAPTIVE_EXPLORATION_INTERVAL", "8"
            )
        )
        cls.proactive_adaptive_safety_margin_ms = float(
            cls._from_env_default(
                "SPECEDGE_PROACTIVE_ADAPTIVE_SAFETY_MARGIN_MS", "2.0"
            )
        )
        cls.proactive_adaptive_uncertainty_scale = float(
            cls._from_env_default(
                "SPECEDGE_PROACTIVE_ADAPTIVE_UNCERTAINTY_SCALE", "1.0"
            )
        )

        # token generation configuration
        cls.max_new_tokens = int(cls._from_env("SPECEDGE_MAX_NEW_TOKENS"))
        cls.max_request_num = int(cls._from_env("SPECEDGE_MAX_REQUEST_NUM"))
        cls.req_offset = int(cls._from_env("SPECEDGE_REQ_OFFSET"))
        cls.sample_req_cnt = int(cls._from_env("SPECEDGE_SAMPLE_REQ_CNT"))

        # server configuration
        cls.host = cls._from_env("SPECEDGE_HOST")
        cls.client_idx = int(cls._from_env("SPECEDGE_CLIENT_IDX"))

        cls._initialized = True


class SpecEdgeBatchClientConfig(metaclass=_ConfigMeta):
    @classmethod
    def _initialize(cls):
        # experiment configuration
        cls.result_path = cls._from_env("SPECEDGE_RESULT_PATH")
        cls.exp_name = cls._from_env("SPECEDGE_EXP_NAME")
        cls.process_name = cls._from_env("SPECEDGE_PROCESS_NAME")
        cls.seed = int(cls._from_env("SPECEDGE_SEED"))
        cls.max_len = int(cls._from_env("SPECEDGE_MAX_LEN"))

        # model configuration
        cls.draft_model = cls._from_env("SPECEDGE_DRAFT_MODEL")
        cls.device = torch.device(cls._from_env("SPECEDGE_CLIENT_DEVICE"))
        cls.dtype = util.convert_dtype(cls._from_env("SPECEDGE_DTYPE"))

        # dataset configuration
        cls.dataset = cls._from_env("SPECEDGE_DATASET")

        # SpecExec configuration
        cls.max_n_beams = int(cls._from_env("SPECEDGE_MAX_N_BEAMS"))
        cls.max_beam_len = int(cls._from_env("SPECEDGE_MAX_BEAM_LEN"))
        cls.max_branch_width = int(cls._from_env("SPECEDGE_MAX_BRANCH_WIDTH"))
        cls.max_budget = int(cls._from_env("SPECEDGE_MAX_BUDGET"))

        # token generation configuration
        cls.max_batch_size = int(cls._from_env("SPECEDGE_MAX_BATCH_SIZE"))
        cls.max_new_tokens = int(cls._from_env("SPECEDGE_MAX_NEW_TOKENS"))
        cls.max_request_num = int(cls._from_env("SPECEDGE_MAX_REQUEST_NUM"))

        # server configuration
        cls.host = cls._from_env("SPECEDGE_HOST")
        cls.req_idx = int(cls._from_env("SPECEDGE_REQ_IDX"))
        cls.sample_req_cnt = int(cls._from_env("SPECEDGE_SAMPLE_REQ_CNT"))

        cls._initialized = True


class SpecEdgeServerConfig(metaclass=_ConfigMeta):
    @classmethod
    def _initialize(cls):
        cls.result_path = cls._from_env("SPECEDGE_RESULT_PATH")
        cls.exp_name = cls._from_env("SPECEDGE_EXP_NAME")
        cls.process_name = cls._from_env("SPECEDGE_PROCESS_NAME")
        cls.seed = int(cls._from_env("SPECEDGE_SEED"))
        cls.optimization = int(cls._from_env("SPECEDGE_OPTIMIZATION"))
        cls.max_len = int(cls._from_env("SPECEDGE_MAX_LEN"))

        # model configuration
        cls.target_model = cls._from_env("SPECEDGE_TARGET_MODEL")
        cls.device = torch.device(cls._from_env("SPECEDGE_DEVICE"))
        cls.dtype = util.convert_dtype(cls._from_env("SPECEDGE_DTYPE"))
        cls.temperature = float(cls._from_env("SPECEDGE_TEMPERATURE"))

        # engine configuration
        cls.max_n_beams = int(cls._from_env("SPECEDGE_MAX_N_BEAMS"))

        cls._initialized = True


class SpecEdgeBatchServerConfig(metaclass=_ConfigMeta):
    @classmethod
    def _initialize(cls):
        cls.result_path = cls._from_env("SPECEDGE_RESULT_PATH")
        cls.exp_name = cls._from_env("SPECEDGE_EXP_NAME")
        cls.process_name = cls._from_env("SPECEDGE_PROCESS_NAME")
        cls.seed = int(cls._from_env("SPECEDGE_SEED"))
        cls.max_len = int(cls._from_env("SPECEDGE_MAX_LEN"))
        cls.batch_type = cls._from_env("SPECEDGE_BATCH_TYPE")
        cls.dataset = cls._from_env("SPECEDGE_DATASET")
        cls.sample_req_cnt = int(cls._from_env("SPECEDGE_SAMPLE_REQ_CNT"))
        cls.req_offset = int(cls._from_env("SPECEDGE_REQ_OFFSET"))

        # model configuration
        cls.target_model = cls._from_env("SPECEDGE_TARGET_MODEL")
        cls.device = torch.device(cls._from_env("SPECEDGE_SERVER_DEVICE"))
        cls.dtype = util.convert_dtype(cls._from_env("SPECEDGE_DTYPE"))
        cls.temperature = float(cls._from_env("SPECEDGE_TEMPERATURE"))

        # engine configuration
        cls.max_batch_size = int(cls._from_env("SPECEDGE_MAX_BATCH_SIZE"))
        cls.max_n_beams = int(cls._from_env("SPECEDGE_MAX_N_BEAMS"))
        cls.max_budget = int(cls._from_env("SPECEDGE_MAX_BUDGET"))
        cls.num_clients = int(cls._from_env("SPECEDGE_NUM_CLIENTS"))
        cls.cache_prefill = cls._from_env("SPECEDGE_CACHE_PREFILL") == "True"
        cls.simulated_latency_ms = float(
            cls._from_env_default("SPECEDGE_SIMULATED_LATENCY_MS", "0.0")
        )
        cls.simulated_decode_latency_ms = float(
            cls._from_env_default(
                "SPECEDGE_SIMULATED_DECODE_LATENCY_MS",
                "0.0",
            )
        )

        cls._initialized = True


class AutoregressiveBatchConfig(metaclass=_ConfigMeta):
    @classmethod
    def _initialize(cls):
        cls.result_path = cls._from_env("SPECEDGE_RESULT_PATH")
        cls.exp_name = cls._from_env("SPECEDGE_EXP_NAME")
        cls.process_name = cls._from_env("SPECEDGE_PROCESS_NAME")
        cls.seed = int(cls._from_env("SPECEDGE_SEED"))

        cls.model = cls._from_env("SPECEDGE_MODEL")
        cls.device = torch.device(cls._from_env("SPECEDGE_DEVICE"))
        cls.dtype = util.convert_dtype(cls._from_env("SPECEDGE_DTYPE"))
        cls.temperature = float(cls._from_env("SPECEDGE_TEMPERATURE"))

        cls.dataset = cls._from_env("SPECEDGE_DATASET")

        cls.max_len = int(cls._from_env("SPECEDGE_MAX_LEN"))
        cls.max_new_tokens = int(cls._from_env("SPECEDGE_MAX_NEW_TOKENS"))
        cls.max_request_num = int(cls._from_env("SPECEDGE_MAX_REQUEST_NUM"))
        cls.batch_size = int(cls._from_env("SPECEDGE_BATCH_SIZE"))
        cls.sample_req_cnt = int(cls._from_env("SPECEDGE_SAMPLE_REQ_CNT"))

        cls._initialized = True
