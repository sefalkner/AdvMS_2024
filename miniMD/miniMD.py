
import numpy as np
from numba import jit

@jit(nopython=True) 
def update_positions(current_x : np.ndarray, force : np.ndarray, beta : float, dt : float, diffusion_coeff : float) -> np.ndarray:
    """
    Update the positions using overdamped langevin dynamics:

        x_(i+1) = x_i + D * beta * force * dt + sqrt(2 * D * dt) * g

    where x_i are the positions at i (current_x) and x_(i+1) are the updated positions (next_x). 
    The diffusion coefficient D (diffusion_coeff), the timestep dt, the temperature in form of 
    beta and the force are needed for the propagation. The factor g is a random number from a 
    standard normal distribution.

    Parameters
    ----------
    current_x : np.ndarray
        Current configuration to be propagated. The shape of the array(current_x.shape) can vary depending on the system which is simulated.
        
    force : np.ndarray
        The force corresponding to current_x. This has to be of the same shape as current_x. 

    beta : float
        Beta determines the simulation temperature, it is equivalent to 1/kT. Must be greater than 0. 

    dt : float
        The simulation timestep for the propagation of current_x. Must be greater than 0. Decrease this or diffusion_coeff if you experience an unstable configuration.
    
    diffusion_coeff : float
        The diffusion coefficient for the propagation of current_x determining the magnitude of random "bumps". Must be greater than 0. Decrease this or diffusion_coeff if you experience an unstable configuration.
        
    Returns
    -------
    new_x : np.ndarray
        Updated Configuration.

    """

    assert dt > 0, "Timestep must be positive."
    assert diffusion_coeff > 0, "Diffusion coefficient must be positive."
    assert beta > 0, "Temperature must be positive."
    assert current_x.shape == force.shape, "Force and position vector must be of the same size, check your force function."

    # Draw random number from a standard normal distribution
    gauss_rand = np.random.randn(*current_x.shape).astype(np.float32)
    
    # Calculate displacement
    prefactor = np.sqrt(2 * diffusion_coeff * dt)
    dx = diffusion_coeff * dt * force * beta + prefactor * gauss_rand

    # Update the configuration x
    next_x = current_x + dx
    
    return next_x