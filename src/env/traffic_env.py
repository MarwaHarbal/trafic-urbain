import numpy as np

class TrafficEnv:
    def __init__(self, lambda_rates=None, max_queue=4, flow_rate=2):
        if lambda_rates is None:
            lambda_rates = {'N': 0.2, 'S': 0.2, 'E': 0.2, 'W': 0.2}
        self.lambda_rates = lambda_rates
        self.max_queue = max_queue # Niveau de discrétisation maximal
        self.flow_rate = flow_rate # Véhicules s'écoulant par pas de temps au vert
        
        # State: queues (N, S, E, W) and phase
        self.queues = {'N': 0, 'S': 0, 'E': 0, 'W': 0}
        self.phase = 0 # 0 pour N/S vert, 1 pour E/W vert
        self.time_step = 0
        
    def reset(self):
        self.queues = {'N': 0, 'S': 0, 'E': 0, 'W': 0}
        self.phase = 0
        self.time_step = 0
        return self._get_state()
        
    def _get_state(self):
        # Discrétisation pour le Q-Learning
        q_N = min(self.queues['N'], self.max_queue)
        q_S = min(self.queues['S'], self.max_queue)
        q_E = min(self.queues['E'], self.max_queue)
        q_W = min(self.queues['W'], self.max_queue)
        return (q_N, q_S, q_E, q_W, self.phase)
        
    def step(self, action):
        """
        action: 0 (Maintenir la phase), 1 (Changer la phase)
        """
        self.time_step += 1
        
        # 1. Arrivées stochastiques (Processus de Poisson)
        arrivals = {dir: np.random.poisson(rate) for dir, rate in self.lambda_rates.items()}
        for dir in self.queues:
            self.queues[dir] += arrivals[dir]
            
        # 2. Application de l'action et écoulement
        if action == 1:
            # Transition de phase (représente l'orange/rouge intégral)
            # Aucun véhicule ne passe pendant ce pas de temps.
            self.phase = 1 - self.phase
        else:
            # Écoulement du trafic selon la phase verte
            if self.phase == 0: # N/S Vert
                departures = {
                    'N': min(self.queues['N'], self.flow_rate),
                    'S': min(self.queues['S'], self.flow_rate),
                    'E': 0,
                    'W': 0
                }
            else: # E/W Vert
                departures = {
                    'N': 0,
                    'S': 0,
                    'E': min(self.queues['E'], self.flow_rate),
                    'W': min(self.queues['W'], self.flow_rate)
                }
                
            for dir in self.queues:
                self.queues[dir] -= departures[dir]
                
        # 3. Calcul de la récompense
        # Pénalité basée sur le nombre REEL de véhicules en attente
        total_waiting = sum(self.queues.values())
        reward = -total_waiting
        
        next_state = self._get_state()
        done = False # L'environnement est continu
        
        return next_state, reward, done, {'queues': self.queues.copy()}
        
    def render(self):
        print(f"\n--- Temps : {self.time_step} ---")
        phase_str = "Nord/Sud (VERT)  |  Est/Ouest (ROUGE)" if self.phase == 0 else "Nord/Sud (ROUGE) |  Est/Ouest (VERT)"
        print(f"Phase : {phase_str}")
        print(f"          Nord ({self.queues['N']})")
        print(f"           |")
        print(f"Ouest ({self.queues['W']}) --+-- Est ({self.queues['E']})")
        print(f"           |")
        print(f"          Sud ({self.queues['S']})")
