class FixedDurationAgent:
    def __init__(self, cycle_length=10):
        """
        Agent baseline naïf qui alterne les feux à intervalle régulier,
        indépendamment de l'état du trafic.
        """
        self.cycle_length = cycle_length
        self.timer = 0
        
    def act(self, state, evaluate=True):
        self.timer += 1
        # Si le temps écoulé correspond à la durée du cycle, on change de phase
        if self.timer >= self.cycle_length:
            self.timer = 0
            return 1 # Action: Changer
        return 0 # Action: Maintenir
