from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import numpy as np
import yaml
import pickle
from typing import Callable, Optional

@dataclass
class GenerationConfig:
    ngen_mc: int = 10**5
    ngen_true: int = 10**4
    of_niter: int = 5
    use_poisson_fluctuations_around_total: bool = True

@dataclass
class TrainingConfig:
    learning_rate: float = 0.0005
    epochs: int = 50
    patience: int = -1
    restore_best_weights: bool = False
    n_models_to_ensemble: int = 10
    save_step2_model: bool = False
    do_bootstrap: bool = False

@dataclass
class RegularizationConfig:
    l1reg_kernel: float = 0.0
    l2reg_kernel: float = 0.0
    l1reg_activation: float = 0.0
    l2reg_activation: float = 0.0
    l1reg_bias: float = 0.0
    l2reg_bias: float = 0.0

@dataclass
class ModelConfig:
    units_per_layer: int = 50
    number_of_layers: int = 3
    dropout: float = 0.0
    use_batch_norm: bool = False
    activation: Callable = None
    beta: Optional[float] = None

@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    generation: GenerationConfig
    training: TrainingConfig
    regularization: RegularizationConfig
    model: ModelConfig
    ndim: int
    random_seed: int
    
    @property
    def batch_size(self) -> int:
        """Batch size equals ngen_true"""
        return self.generation.ngen_true

def generate_samples(config: ExperimentConfig, rng, mc_mu, mc_cov, true_mu, true_cov, resolution):
    """Generate MC and true samples with detector resolution effects"""
    
    # Generate base samples
    mc_pts = rng.multivariate_normal(mc_mu, mc_cov, size=config.generation.ngen_mc)
    true_pts = rng.multivariate_normal(true_mu, true_cov, size=config.generation.ngen_true)
    
    # Add detector resolution
    mc_det_pts = rng.normal(mc_pts, resolution)
    true_det_pts = rng.normal(true_pts, resolution)
    
    # Generate larger sample with weights (40x larger)
    scale_factor = 40
    true_pts_large = rng.multivariate_normal(
        true_mu, true_cov, 
        size=config.generation.ngen_true * scale_factor
    )
    true_det_pts_large = rng.normal(true_pts_large, resolution)
    true_weights_large = np.full(
        config.generation.ngen_true * scale_factor, 
        1.0 / scale_factor
    )
    
    return {
        'mc_pts': mc_pts,
        'mc_det_pts': mc_det_pts,
        'true_pts': true_pts,
        'true_det_pts': true_det_pts,
        'true_pts_large': true_pts_large,
        'true_det_pts_large': true_det_pts_large,
        'true_weights_large': true_weights_large,
    }

def save_experiment_data(output_dir: Path, samples: dict, parameters: dict, config: ExperimentConfig):
    """Save all experiment data and configuration"""
    
    # Save sample data
    np.savez(
        output_dir / 'mc-and-true-samples.npz',
        mc_pts=samples['mc_pts'],
        mc_det_pts=samples['mc_det_pts'],
        true_pts=samples['true_pts'],
        true_det_pts=samples['true_det_pts'],
        true_pts10x=samples['true_pts_large'],
        true_det_pts10x=samples['true_det_pts_large'],
        true_pts10x_weights=samples['true_weights_large'],
    )
    
    # Save parameters as pickle
    with open(output_dir / 'config-pars.pkl', 'wb') as f:
        pickle.dump(parameters, f)
    
    # Prepare YAML configuration
    config_data = {
        'timestamp': datetime.now().isoformat(),
        'generation': asdict(config.generation),
        'training': asdict(config.training),
        'regularization': asdict(config.regularization),
        'model': {
            **asdict(config.model),
            'activation': config.model.activation.__name__ if config.model.activation else None
        },
        'experiment': {
            'ndim': config.ndim,
            'random_seed': config.random_seed,
            'batch_size': config.batch_size
        },
        'parameters': {
            'mc': {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in parameters.items() if 'mc' in k},
            'true': {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in parameters.items() if 'true' in k},
            'resolution': {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in parameters.items() if 'resolution' in k}
        }
    }
    
    # Save YAML configuration
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False, indent=2)

# Usage example:
def run_experiment(output_dir, rng, mc_mu, mc_cov, true_mu, true_cov, resolution, 
                   ndim, random_seed, parametric_gelu, beta, resolution_sf, 
                   mc_rho, mc_sig, true_rho, true_sig):
    """Run the complete experiment with clean configuration"""
    
    # Create configuration
    config = ExperimentConfig(
        generation=GenerationConfig(),
        training=TrainingConfig(),
        regularization=RegularizationConfig(),
        model=ModelConfig(activation=parametric_gelu, beta=beta),
        ndim=ndim,
        random_seed=random_seed
    )
    
    # Generate samples
    samples = generate_samples(config, rng, mc_mu, mc_cov, true_mu, true_cov, resolution)
    
    # Prepare parameters
    parameters = {
        'mc_mu': mc_mu,
        'mc_rho': mc_rho,
        'mc_sig': mc_sig,
        'true_mu': true_mu,
        'true_rho': true_rho,
        'true_sig': true_sig,
        'resolution': resolution,
        'resolution_sf': resolution_sf,
        'mc_cov': mc_cov,
        'true_cov': true_cov,
    }
    
    # Save everything
    save_experiment_data(output_dir, samples, parameters, config)
    
    return config, samples

# Alternative: If you want to customize specific parameters
def create_custom_config(**overrides):
    """Create configuration with custom overrides"""
    config = ExperimentConfig(
        generation=GenerationConfig(),
        training=TrainingConfig(),
        regularization=RegularizationConfig(),
        model=ModelConfig(),
        ndim=2,  # default
        random_seed=42  # default
    )
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            # Handle nested overrides like training.epochs=100
            parts = key.split('.')
            if len(parts) == 2 and hasattr(config, parts[0]):
                section = getattr(config, parts[0])
                if hasattr(section, parts[1]):
                    setattr(section, parts[1], value)
    
    return config