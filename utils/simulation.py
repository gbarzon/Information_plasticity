import numpy as np
from numba import njit
from utils.numba_utils import numba_random_normal, set_seed

def create_info_system(N, w, k, tau_ei, r, tau_p, p, D):
    A = w * np.array([[1, -k], [1, -k]])
    return {'N': N, 'w': w, 'k': k, 'tau_ei': tau_ei, 'r': r, 'tau_p': tau_p, 'p': p, 'D': D, 'A': A}
    
def create_info_input(W, hs):
    return {'W': W, 'hs': hs}

def create_info_simulation(steps, dt):
    return {'steps': steps, 'dt': dt}

def simulate_coupled_system(info_system, info_input, info_simulation, linear=True, seed=None):
    if seed is not None:
        set_seed(seed)
    ### Simulate input
    inputs = simulate_rates(info_simulation['steps'], info_input['W'], info_simulation['dt'], 0)
    
    ### Simulate system
    #hs = np.array([info_input['hs'][inputs], np.zeros(info_simulation['steps'])]).T
    inputs_values = (info_input['hs'].T)[inputs]
    if linear:
        states = simulate_plastic_single(info_simulation['steps'], info_system['A'], info_system['r'], info_system['tau_ei'], info_system['p'], info_system['tau_p'],  info_system['D'], inputs_values, info_simulation['dt'])
    else:
        states = simulate_nonlinear_plastic(info_simulation['steps'], info_system['A'], info_system['r'], info_system['tau_ei'], info_system['p'], info_system['tau_p'],  info_system['D'], inputs_values, info_simulation['dt'])
    
    return (inputs, states)

def create_transition_matrix_star_graph(M, wup, wdown):
    K = M-1
    W = np.zeros((M,M))
    W[:,0] = wup
    np.fill_diagonal(W, -wdown)
    W[0] = np.array( [-K*wup] + [wdown]*K )
    
    return W

@njit
def step_linear(x, A, r, tau, D, h, dt):
    return x +  (-r * x + A @ x + h) * dt / tau + np.sqrt(2*D) * numba_random_normal(x.size) * np.sqrt(dt/tau) #/ tau

@njit
def step_nonlinear(x, A, r, tau, D, h, dt):
    #return x +  (-r * x + np.tanh( A @ x + h) ) * dt / tau + np.sqrt(2*D) * numba_random_normal(x.size) * np.sqrt(dt) / tau
    return x +  (-r * x + A @ np.tanh(x) + h ) * dt / tau + np.sqrt(2*D) * numba_random_normal(x.size) * np.sqrt(dt/tau) #/ tau

@njit
def step_plasticity(x, K, p, tau_p, dt):
    #return x +  (-r * x + np.tanh( A @ x + h) ) * dt / tau + np.sqrt(2*D) * numba_random_normal(x.size) * np.sqrt(dt) / tau
    return K + (-K + p * np.outer(x, x) ) * dt / tau_p

@njit
def step_plasticity_single(x, K, p, tau_p, dt):
    #return x +  (-r * x + np.tanh( A @ x + h) ) * dt / tau + np.sqrt(2*D) * numba_random_normal(x.size) * np.sqrt(dt) / tau
    return K + (-K + p * x[0]*x[1] ) * dt / tau_p #+ 0.0001 * numba_random_normal(2)[0]

@njit
def step_plasticity_nonlinear(x, K, p, tau_p, dt):
    #return x +  (-r * x + np.tanh( A @ x + h) ) * dt / tau + np.sqrt(2*D) * numba_random_normal(x.size) * np.sqrt(dt) / tau
    return K + (-K + p * np.outer(np.tanh(x), np.tanh(x)) ) * dt / tau_p

@njit
def simulate(steps, A, r, tau, D, h, dt):
    states = np.zeros((steps, A.shape[0]))
    
    for t in range(1,steps):
        states[t] = step_linear(states[t-1], A, r, tau, D, h[t], dt)
        
    return states

@njit
def simulate_plastic_single(steps, A, r, tau_mu, p, tau_p, D, h, dt):
    states = np.zeros((steps, A.shape[0]))
    plasticity = np.zeros((steps))
    
    #states[0] += 1.
    plasticity[0] = 6.
    
    for t in range(1,steps):
        states[t] = step_linear(states[t-1], A*np.exp(plasticity[t-1]), r, tau_mu, D, h[t], dt)
        plasticity[t] = step_plasticity_single(states[t], plasticity[t-1], p, tau_p, dt)
        
    return states, plasticity

@njit
def simulate_plastic(steps, A, r, tau_mu, p, tau_p, D, h, dt):
    states = np.zeros((steps, A.shape[0]))
    plasticity = np.zeros((steps, A.shape[0], A.shape[0]))
    
    for t in range(1,steps):
        states[t] = step_linear(states[t-1], A*np.exp(plasticity[t-1][0,1]), r, tau_mu, D, h[t], dt)
        plasticity[t] = step_plasticity(states[t], plasticity[t-1], p, tau_p, dt)
        
    return states, plasticity

@njit
def simulate_nonlinear(steps, A, r, tau, D, h, dt):
    states = np.zeros((steps, A.shape[0]))
    
    for t in range(1,steps):
        states[t] = step_nonlinear(states[t-1], A, r, tau, D, h[t], dt)
        
    return states

@njit
def simulate_nonlinear_plastic(steps, A, r, tau_mu, p, tau_p, D, h, dt):
    states = np.zeros((steps, A.shape[0]))
    plasticity = np.zeros((steps, A.shape[0], A.shape[0]))
    
    for t in range(1,steps):
        states[t] = step_nonlinear(states[t-1], A*np.exp(plasticity[t-1][0,1]), r, tau_mu, D, h[t], dt)
        plasticity[t] = step_plasticity_nonlinear(states[t], plasticity[t-1], p, tau_p, dt)
        
    return states, plasticity

@njit
def simulate_master_equation(steps, W, dt, p0):
    states = np.zeros((steps, W.shape[0]))
    states[0] = p0
    
    for t in range(1,steps):
        states[t] = states[t-1] + dt * W @ states[t-1]
        
    return states

@njit
def simulate_rates(steps, W, dt, h0):
    states = np.zeros(steps, dtype=np.int64)
    states[0] = h0
    
    for t in range(1,steps):
        ### Write rate equation
        probs = W[:,states[t-1]]*dt
        probs[states[t-1]] = 0
        probs[states[t-1]] = 1 - probs.sum()
        
        states[t] = np.where( np.random.random() < np.cumsum(probs) )[0][0]
        
    return states