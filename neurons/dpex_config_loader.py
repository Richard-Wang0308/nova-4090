"""
DPEX Configuration Loader (FIXED)
=================================

Loads pool sizes and parameters from YAML config file.
Provides validation and default fallbacks.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """Pool size configuration."""
    pop_A_size: int = 1200
    pop_B_size: int = 1000
    elite_pool_size: int = 700
    pop_B_start_offset: int = 100


@dataclass
class TabuConfig:
    """Tabu list configuration."""
    tabu_A_maxlen: int = 150
    tabu_B_maxlen: int = 150
    tabu_C_maxlen: int = 150


@dataclass
class CandidateConfig:
    """Candidate generation configuration."""
    seed_budget: int = 80
    early_budget: int = 100
    normal_budget: int = 80
    stagnation_budget: int = 60
    dja_ratio_early: float = 0.60
    tabu_ratio_early: float = 0.40
    dja_ratio_stagnation: float = 0.30
    tabu_ratio_stagnation: float = 0.70


@dataclass
class QualityFilterConfig:
    """Quality filtering configuration."""
    min_similarity_to_elite: float = 0.55
    diversity_threshold: float = 0.65
    duplication_tolerance: float = 0.05


@dataclass
class StagnationConfig:
    """Stagnation detection configuration."""
    window_size: int = 3
    improvement_threshold: float = 0.001
    exploit_mode_threshold: int = 3


@dataclass
class ScoringConfig:
    """Scoring configuration."""
    cost_per_molecule: float = 30.0
    batch_size: int = 10
    max_molecules_per_iteration: int = 100


@dataclass
class DJAConfig:
    """DJA algorithm configuration."""
    best_adoption_probability: float = 0.60
    escape_probability: float = 0.50
    use_elite_diversity: bool = True
    elite_percentage: float = 0.10


@dataclass
class TabuAlgoConfig:
    """Tabu algorithm configuration (SEPARATE from TabuConfig)."""
    min_similarity_for_neighbors: float = 0.50
    num_similar_components: int = 5
    aspiration_threshold: float = 0.97


@dataclass
class ExploitConfig:
    """Exploit mode configuration."""
    min_similarity_threshold: float = 0.70
    num_similar_to_explore: int = 10
    budget_percentage: float = 0.20


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    stats_interval: int = 5
    verbose_generation: bool = False
    verbose_filtering: bool = True


@dataclass
class DPEXConfig:
    """Complete DPEX configuration."""
    pool: PoolConfig = field(default_factory=PoolConfig)
    tabu: TabuConfig = field(default_factory=TabuConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    quality_filter: QualityFilterConfig = field(default_factory=QualityFilterConfig)
    stagnation: StagnationConfig = field(default_factory=StagnationConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    dja: DJAConfig = field(default_factory=DJAConfig)
    tabu_algo: TabuAlgoConfig = field(default_factory=TabuAlgoConfig)
    exploit: ExploitConfig = field(default_factory=ExploitConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy access."""
        return {
            'pool': self.pool.__dict__,
            'tabu': self.tabu.__dict__,
            'candidates': self.candidates.__dict__,
            'quality_filter': self.quality_filter.__dict__,
            'stagnation': self.stagnation.__dict__,
            'scoring': self.scoring.__dict__,
            'dja': self.dja.__dict__,
            'tabu_algo': self.tabu_algo.__dict__,
            'exploit': self.exploit.__dict__,
            'logging': self.logging.__dict__,
        }


class DPEXConfigLoader:
    """Loads and validates DPEX configuration from YAML."""
    
    def __init__(self, config_path: str = None):
        """Initialize config loader."""
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__),
                'dpex_pool_config.yaml'
            )
        
        self.config_path = config_path
        self.config: Optional[DPEXConfig] = None
    
    def load(self) -> DPEXConfig:
        """Load configuration from YAML file."""
        
        if not os.path.exists(self.config_path):
            logger.warning(
                f"Config file not found at {self.config_path}, "
                f"using defaults"
            )
            self.config = DPEXConfig()
            return self.config
        
        try:
            with open(self.config_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
            
            if yaml_data is None:
                logger.warning("Config file is empty, using defaults")
                self.config = DPEXConfig()
                return self.config
            
            # Parse each section with proper error handling
            pool_data = yaml_data.get('pool', {})
            tabu_data = yaml_data.get('tabu', {})
            candidates_data = yaml_data.get('candidates', {})
            quality_filter_data = yaml_data.get('quality_filter', {})
            stagnation_data = yaml_data.get('stagnation', {})
            scoring_data = yaml_data.get('scoring', {})
            dja_data = yaml_data.get('dja', {})
            tabu_algo_data = yaml_data.get('tabu_algo', {})  # FIXED: separate from tabu
            exploit_data = yaml_data.get('exploit', {})
            logging_data = yaml_data.get('logging', {})
            
            # Create config object with filtered parameters
            self.config = DPEXConfig(
                pool=PoolConfig(**self._filter_dict(pool_data, PoolConfig)),
                tabu=TabuConfig(**self._filter_dict(tabu_data, TabuConfig)),
                candidates=CandidateConfig(**self._filter_dict(candidates_data, CandidateConfig)),
                quality_filter=QualityFilterConfig(**self._filter_dict(quality_filter_data, QualityFilterConfig)),
                stagnation=StagnationConfig(**self._filter_dict(stagnation_data, StagnationConfig)),
                scoring=ScoringConfig(**self._filter_dict(scoring_data, ScoringConfig)),
                dja=DJAConfig(**self._filter_dict(dja_data, DJAConfig)),
                tabu_algo=TabuAlgoConfig(**self._filter_dict(tabu_algo_data, TabuAlgoConfig)),
                exploit=ExploitConfig(**self._filter_dict(exploit_data, ExploitConfig)),
                logging=LoggingConfig(**self._filter_dict(logging_data, LoggingConfig)),
            )
            
            logger.info(f"✅ Config loaded from {self.config_path}")
            self._log_config_summary()
            
            return self.config
        
        except yaml.YAMLError as e:
            logger.error(f"❌ YAML parsing error: {e}")
            logger.warning("Using default configuration")
            self.config = DPEXConfig()
            return self.config
        
        except TypeError as e:
            logger.error(f"❌ Config parsing error: {e}")
            logger.warning("Using default configuration")
            self.config = DPEXConfig()
            return self.config
        
        except Exception as e:
            logger.error(f"❌ Unexpected error loading config: {e}")
            logger.warning("Using default configuration")
            self.config = DPEXConfig()
            return self.config
    
    @staticmethod
    def _filter_dict(data: Dict, target_class) -> Dict:
        """Filter dictionary to only include valid fields for target class."""
        if not data:
            return {}
        
        # Get valid field names from dataclass
        valid_fields = set()
        if hasattr(target_class, '__dataclass_fields__'):
            valid_fields = set(target_class.__dataclass_fields__.keys())
        
        # Filter data to only include valid fields
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        
        return filtered
    
    def _log_config_summary(self) -> None:
        """Log configuration summary."""
        if not self.config:
            return
        
        logger.info(
            f"📋 DPEX Configuration Summary:"
            f"\n   Pool A size: {self.config.pool.pop_A_size}"
            f"\n   Pool B size: {self.config.pool.pop_B_size}"
            f"\n   Elite pool size: {self.config.pool.elite_pool_size}"
            f"\n   Tabu A maxlen: {self.config.tabu.tabu_A_maxlen}"
            f"\n   Tabu B maxlen: {self.config.tabu.tabu_B_maxlen}"
            f"\n   Seed budget: {self.config.candidates.seed_budget}"
            f"\n   Early budget: {self.config.candidates.early_budget}"
            f"\n   Normal budget: {self.config.candidates.normal_budget}"
            f"\n   Stagnation budget: {self.config.candidates.stagnation_budget}"
            f"\n   Scoring cost/molecule: {self.config.scoring.cost_per_molecule}s"
        )
    
    def validate(self) -> bool:
        """Validate configuration values."""
        if not self.config:
            return False
        
        errors = []
        
        # Validate pool sizes
        if self.config.pool.pop_A_size <= 0:
            errors.append("pop_A_size must be > 0")
        if self.config.pool.pop_B_size <= 0:
            errors.append("pop_B_size must be > 0")
        if self.config.pool.elite_pool_size <= 0:
            errors.append("elite_pool_size must be > 0")
        
        # Validate tabu sizes
        if self.config.tabu.tabu_A_maxlen <= 0:
            errors.append("tabu_A_maxlen must be > 0")
        
        # Validate thresholds (0-1 range)
        if not (0.0 <= self.config.quality_filter.min_similarity_to_elite <= 1.0):
            errors.append("min_similarity_to_elite must be in [0.0, 1.0]")
        if not (0.0 <= self.config.quality_filter.diversity_threshold <= 1.0):
            errors.append("diversity_threshold must be in [0.0, 1.0]")
        
        # Validate probabilities
        if not (0.0 <= self.config.dja.best_adoption_probability <= 1.0):
            errors.append("best_adoption_probability must be in [0.0, 1.0]")
        
        # Validate ratios
        if not (0.0 <= self.config.candidates.dja_ratio_early <= 1.0):
            errors.append("dja_ratio_early must be in [0.0, 1.0]")
        
        if errors:
            for error in errors:
                logger.error(f"❌ Validation error: {error}")
            return False
        
        logger.info("✅ Configuration validation passed")
        return True
    
    def get_config(self) -> DPEXConfig:
        """Get loaded configuration."""
        if self.config is None:
            return self.load()
        return self.config


# ════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ════════════════════════════════════════════════════════════════

def load_dpex_config(config_path: str = None) -> DPEXConfig:
    """Load DPEX configuration from YAML file."""
    loader = DPEXConfigLoader(config_path)
    config = loader.load()
    loader.validate()
    return config


def get_default_config() -> DPEXConfig:
    """Get default configuration."""
    return DPEXConfig()