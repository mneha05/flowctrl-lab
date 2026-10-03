from dataclasses import dataclass
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def _mlp(seed, max_iter, hidden=(64, 64)):
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=hidden, activation="tanh", solver="adam",
                     learning_rate_init=2e-3, max_iter=max_iter, batch_size=256,
                     random_state=seed, early_stopping=True, validation_fraction=0.1,
                     n_iter_no_change=8, tol=1e-4)
    )

class BehaviorCloningPolicy:
    name = "behavior_cloning"
    def __init__(self, max_action, seed=0, max_iter=100):
        self.max_action = max_action
        self.model = _mlp(seed, max_iter)
    def fit(self, states, actions):
        self.model.fit(states, actions/self.max_action)
        return self
    def act(self, state, rng):
        x = self.model.predict(np.asarray(state).reshape(1,-1))[0]
        return np.clip(x,-1,1)*self.max_action

@dataclass
class DiffusionSchedule:
    steps: int = 16
    beta_start: float = 0.02
    beta_end: float = 0.18
    def arrays(self):
        beta=np.linspace(self.beta_start,self.beta_end,self.steps)
        alpha=1-beta
        return beta, alpha, np.cumprod(alpha)

class DiffusionPolicy:
    name = "diffusion"
    def __init__(self, max_action, seed=0, max_iter=100, steps=16):
        self.max_action=max_action
        self.seed=seed
        self.schedule=DiffusionSchedule(steps=steps)
        self.model=_mlp(seed,max_iter)
    def fit(self, states, actions):
        rng=np.random.default_rng(self.seed)
        beta,alpha,abar=self.schedule.arrays()
        n=len(states)
        tidx=rng.integers(0,len(beta),size=n)
        eps=rng.normal(0,1,size=actions.shape)
        clean=actions/self.max_action
        ab=abar[tidx][:,None]
        noisy=np.sqrt(ab)*clean+np.sqrt(1-ab)*eps
        t=(tidx/max(1,len(beta)-1))[:,None]
        self.model.fit(np.concatenate([states,noisy,t],1),eps)
        return self
    def act(self, state, rng):
        beta,alpha,abar=self.schedule.arrays()
        x=rng.normal(0,1,2)
        s=np.asarray(state).reshape(1,-1)
        for i in range(len(beta)-1,-1,-1):
            t=np.array([[i/max(1,len(beta)-1)]])
            eps=self.model.predict(np.concatenate([s,x.reshape(1,-1),t],1))[0]
            mean=(x-(beta[i]/np.sqrt(1-abar[i]+1e-8))*eps)/np.sqrt(alpha[i])
            x=mean+(np.sqrt(beta[i])*rng.normal(0,1,2) if i>0 else 0)
            x=np.clip(x,-2,2)
        return np.clip(x,-1,1)*self.max_action

class FlowMatchingPolicy:
    name = "flow_matching"
    def __init__(self, max_action, seed=0, max_iter=100, steps=16):
        self.max_action=max_action
        self.seed=seed
        self.steps=steps
        self.model=_mlp(seed,max_iter)
    def fit(self, states, actions):
        rng=np.random.default_rng(self.seed)
        clean=actions/self.max_action
        z=rng.normal(0,1,clean.shape)
        t=rng.uniform(0,1,(len(clean),1))
        xt=(1-t)*z+t*clean
        self.model.fit(np.concatenate([states,xt,t],1),clean-z)
        return self
    def act(self, state, rng):
        x=rng.normal(0,1,2)
        s=np.asarray(state).reshape(1,-1)
        dt=1/self.steps
        for i in range(self.steps):
            t=np.array([[(i+0.5)/self.steps]])
            v=self.model.predict(np.concatenate([s,x.reshape(1,-1),t],1))[0]
            x=np.clip(x+dt*v,-2,2)
        return np.clip(x,-1,1)*self.max_action
